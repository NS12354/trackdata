#!/usr/bin/env python3
"""Measure whether footage LABELING is improving — a frozen accuracy benchmark.

Throughput tells you how fast/cheap; this tells you how GOOD. It scores any
labeling config (provider/model/prompt) against a FROZEN set of frames with
human-written reference descriptions, using an LLM-as-judge for the per-frame
score, and logs the result so "improving" becomes a number that goes up (and
regressions become visible).

Workflow
--------
1) Build the eval set ONCE (pre-filled with draft references you then correct):

       python scripts/eval_labeling.py bootstrap data/anonymized --frames-per-clip 4

   Edit data/eval/labels/manifest.json: fix each `reference` to the TRUE activity
   and set a `category` (e.g. "loading", "walking", "lock"). This is your frozen
   yardstick — don't change it once you start comparing.

2) Score a config (re-run after ANY change to see if it improved):

       # current local model
       python scripts/eval_labeling.py score

       # a different model / provider (overrides settings for this run)
       python scripts/eval_labeling.py score --provider ollama --model moondream
       python scripts/eval_labeling.py score --provider claude

Each `score` prints overall accuracy, a per-category breakdown, coverage, cost,
and the DELTA vs the previous run — and appends the run to a versioned log so you
can track the trend. The judge defaults to your local llama3.1:8b (free); pass
--judge claude to use the API. Calibrate the judge once by hand-checking a few of
its verdicts before you trust the number.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config import settings  # noqa: E402
from pipeline.video_meta import probe  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
DEFAULT_SET = EVAL_DIR / "labels"
RUNS_LOG = EVAL_DIR / "labeling_runs.jsonl"


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------
def _encode_frame(bgr, max_dim: int) -> bytes:
    h, w = bgr.shape[:2]
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _sample_frames(video: Path, n: int, max_dim: int):
    """Yield (timestamp_s, jpeg_bytes) for n evenly-spaced frames."""
    meta = probe(video)
    dur = meta.duration_seconds or (meta.frame_count / (meta.fps or 30.0))
    cap = cv2.VideoCapture(str(video))
    try:
        for i in range(n):
            t = dur * (i + 0.5) / n  # centers of n equal slices
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            yield round(t, 2), _encode_frame(frame, max_dim)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Labeler (the thing under test) + Judge
# ---------------------------------------------------------------------------
def _caption(provider, jpeg: bytes) -> tuple[str, float]:
    """Run the labeler on one frame -> (description, cost_usd)."""
    label, cost = provider.classify(jpeg)
    return (label.description or "").strip(), cost


_JUDGE_PROMPT = (
    "You are grading an AI's one-sentence description of a single video frame "
    "against a human reference (the ground truth of what the person is doing).\n\n"
    "REFERENCE (truth): {ref}\n"
    "AI DESCRIPTION: {cand}\n\n"
    "Score how well the AI captured the MAIN activity and the object handled:\n"
    "  1   = same activity AND object (wording may differ)\n"
    "  0.5 = partially right (correct activity OR object, or vague/incomplete)\n"
    "  0   = wrong, unrelated, or empty\n\n"
    'Respond ONLY with compact JSON: {{"score": 0|0.5|1, "reason": "<few words>"}}'
)


def _judge_ollama(ref: str, cand: str, model: str) -> tuple[float, str]:
    if not cand:
        return 0.0, "empty caption"
    prompt = _JUDGE_PROMPT.format(ref=ref, cand=cand)
    r = requests.post(
        settings.ollama_base_url.rstrip("/") + "/api/chat",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "stream": False, "format": "json", "options": {"temperature": 0.0}},
        timeout=settings.ollama_timeout_seconds,
    )
    r.raise_for_status()
    return _parse_score(r.json().get("message", {}).get("content", ""))


def _judge_claude(ref: str, cand: str, model: str) -> tuple[float, str]:
    if not cand:
        return 0.0, "empty caption"
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=model, max_tokens=100,
        messages=[{"role": "user", "content": _JUDGE_PROMPT.format(ref=ref, cand=cand)}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return _parse_score(text)


def _parse_score(text: str) -> tuple[float, str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            s = float(obj.get("score", 0))
            s = min(1.0, max(0.0, round(s * 2) / 2))  # snap to {0, 0.5, 1}
            return s, str(obj.get("reason", ""))[:80]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 0.0, "unparseable judge reply"


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def cmd_bootstrap(args):
    clips: list[Path] = []
    for raw in args.clips:
        p = Path(raw)
        if p.is_dir():
            clips += sorted(p.glob("*.mp4"))
        elif p.exists():
            clips.append(p)
    if not clips:
        raise SystemExit("no .mp4 clips found in the given paths")

    out = Path(args.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Use the current labeler to PRE-FILL draft references (you then correct them).
    from pipeline.segmentation_providers import get_segmentation_provider
    provider = get_segmentation_provider()

    items = []
    print(f"Bootstrapping eval set from {len(clips)} clip(s) -> {out}\n")
    for clip in clips:
        for ts, jpeg in _sample_frames(clip, args.frames_per_clip, settings.segmentation_frame_max_dim):
            name = f"{clip.stem}_{ts:06.2f}.jpg"
            (frames_dir / name).write_bytes(jpeg)
            draft, _ = _caption(provider, jpeg)
            items.append({
                "frame": f"frames/{name}",
                "source_clip": clip.name,
                "timestamp_s": ts,
                "reference": draft,        # <-- EDIT THIS to the true activity
                "draft_reference": draft,  # kept so you can see what the model said
                "category": "",            # <-- SET THIS (e.g. loading, walking, lock)
            })
            print(f"  {name}: \"{draft[:70]}\"")

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(items, indent=2))
    print(f"\nWrote {len(items)} frames to {manifest}")
    print("NEXT: open manifest.json, correct each `reference` to the TRUE activity,")
    print("      and fill in `category`. Then: python scripts/eval_labeling.py score")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def cmd_score(args):
    set_dir = Path(args.set)
    manifest = set_dir / "manifest.json" if set_dir.is_dir() else set_dir
    if not manifest.exists():
        raise SystemExit(f"eval set not found: {manifest} (run `bootstrap` first)")
    base = manifest.parent
    items = json.loads(manifest.read_text())
    if not items:
        raise SystemExit("eval set is empty")

    # Apply config overrides for THIS run (so you can compare models without
    # editing .env). These mutate the in-memory settings only.
    if args.provider:
        settings.segmentation_provider = args.provider
    if args.model:
        settings.ollama_vlm_model = args.model
        settings.claude_vlm_model = args.model
    settings.segmentation_mode = "open"  # labeling = open-vocab description

    from pipeline.segmentation_providers import get_segmentation_provider
    provider = get_segmentation_provider()
    judge_fn = _judge_claude if args.judge == "claude" else _judge_ollama
    judge_model = args.judge_model or (settings.claude_vlm_model if args.judge == "claude"
                                       else settings.chat_model)

    cfg = {
        "provider": settings.segmentation_provider,
        "model": (settings.claude_vlm_model if settings.segmentation_provider == "claude"
                  else settings.ollama_vlm_model),
        "frame_max_dim": settings.segmentation_frame_max_dim,
        "judge": f"{args.judge}:{judge_model}",
    }
    print(f"Scoring {len(items)} frames | labeler={cfg['provider']}/{cfg['model']} "
          f"| judge={cfg['judge']}\n")

    total, cov, cost = 0.0, 0, 0.0
    by_cat: dict[str, list] = {}
    rows = []
    for it in items:
        ref = (it.get("reference") or "").strip()
        if not ref:
            continue  # skip un-annotated frames
        jpeg = (base / it["frame"]).read_bytes()
        cand, c = _caption(provider, jpeg)
        cost += c
        score, reason = judge_fn(ref, cand, judge_model)
        total += score
        cov += 1 if cand else 0
        cat = it.get("category") or "uncategorized"
        by_cat.setdefault(cat, []).append(score)
        rows.append((it["frame"], score, cand[:48], reason[:32]))
        mark = {1.0: "✓", 0.5: "~", 0.0: "✗"}.get(score, "?")
        print(f"  {mark} {score:>3} {Path(it['frame']).name:<28} \"{cand[:46]}\"")

    n = sum(len(v) for v in by_cat.values())
    if n == 0:
        raise SystemExit("no annotated frames (every `reference` is blank) — edit the manifest")
    acc = total / n

    print("\n" + "=" * 60)
    print(f"ACCURACY: {acc*100:5.1f}%   ({total:.1f}/{n} points over {n} frames)")
    print(f"COVERAGE: {cov}/{n} frames captioned non-empty")
    print(f"COST:     ${cost:.4f} for this eval run "
          f"({'$0 local' if cost == 0 else f'~${cost/n*1000:.2f}/1000 frames'})")
    print("\nPer-category:")
    for cat, scores in sorted(by_cat.items()):
        print(f"  {cat:<20} {sum(scores)/len(scores)*100:5.1f}%  (n={len(scores)})")

    # Trend vs the previous run with the SAME eval set.
    prev = _last_run(set_dir.name)
    if prev is not None:
        delta = (acc - prev["accuracy"]) * 100
        arrow = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "=")
        print(f"\nvs previous run ({prev['config']['provider']}/{prev['config']['model']}): "
              f"{prev['accuracy']*100:.1f}% -> {acc*100:.1f}%  {arrow} {delta:+.1f}pts")
    else:
        print("\n(first run on this eval set — future runs will show the delta)")
    print("=" * 60)

    _append_run({
        "ts": time.time(),
        "set": set_dir.name,
        "config": cfg,
        "accuracy": round(acc, 4),
        "coverage": round(cov / n, 4),
        "cost_usd": round(cost, 6),
        "n": n,
        "per_category": {k: round(sum(v) / len(v), 4) for k, v in by_cat.items()},
    })
    print(f"\nLogged to {RUNS_LOG}")


def _last_run(set_name: str):
    if not RUNS_LOG.exists():
        return None
    last = None
    for line in RUNS_LOG.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("set") == set_name:
            last = rec
    return last


def _append_run(rec: dict):
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="build a frozen eval set (draft references pre-filled)")
    b.add_argument("clips", nargs="+", help="clip files or folders of .mp4")
    b.add_argument("--out", default=str(DEFAULT_SET), help="output eval-set dir")
    b.add_argument("--frames-per-clip", type=int, default=4)
    b.set_defaults(func=cmd_bootstrap)

    s = sub.add_parser("score", help="score a config against the eval set + log the result")
    s.add_argument("--set", default=str(DEFAULT_SET), help="eval-set dir or manifest.json")
    s.add_argument("--provider", default=None, help="override: ollama | claude")
    s.add_argument("--model", default=None, help="override labeler model")
    s.add_argument("--judge", default="ollama", choices=["ollama", "claude"])
    s.add_argument("--judge-model", default=None, help="override judge model")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
