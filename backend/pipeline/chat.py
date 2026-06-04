"""Chatbot (Phase 6) — answers questions grounded in the activity commentary
(open-mode segmentation) plus the operational events/metrics.

The model only sees the timestamped descriptions and summary stats we assemble —
it does not re-watch video. Local LLM via Ollama by default (free, no data
egress); optional Claude provider.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import requests
from sqlalchemy import select

from config import settings
from db import session_scope
from models import Video
from storage import get_storage
from .events import summarize_video

log = logging.getLogger("revisent.chat")

_SYSTEM = (
    "You are an assistant that answers questions about what happened in a worker's "
    "processed camera footage. You are given timestamped activity descriptions and "
    "summary metrics. Answer ONLY from this data. If it doesn't contain the answer, "
    "say you can't tell from the footage. Be concise and cite timestamps (mm:ss) "
    "when relevant. Do not invent details."
)


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _video_context(video_id: str) -> Optional[str]:
    storage = get_storage()
    seg_key = f"processed/{video_id}/segments.json"
    if not storage.exists(seg_key):
        return None
    data = json.loads(storage.local_path(seg_key).read_text())
    segs = data.get("segments", [])[: settings.chat_max_segments]
    with session_scope() as s:
        v = s.get(Video, video_id)
        meta = f"file={v.original_filename}, property={v.property_tag or 'n/a'}, " \
               f"duration={v.duration_seconds or 0:.0f}s" if v else video_id
    summary = summarize_video(video_id)

    lines = [f"VIDEO [{video_id[:8]}] {meta}"]
    lines.append(f"  duration_active_hands={summary.get('active_hand_seconds')}s, "
                 f"service_events={summary['service_event_count']}, "
                 f"downtime={summary['downtime_seconds']}s, "
                 f"contamination={summary['contamination_event_count']}")
    lines.append("  timeline (what the worker was doing):")
    for sg in segs:
        desc = (sg.get("description") or sg.get("task_label") or "").strip()
        if desc:
            lines.append(f"    {_clock(sg['start_time'])}–{_clock(sg['end_time'])}: {desc}")
    return "\n".join(lines)


def build_context(video_id: Optional[str] = None) -> str:
    """Assemble the grounding context: one video, or all that have commentary."""
    if video_id:
        ctx = _video_context(video_id)
        return ctx or f"(no processed commentary available for video {video_id})"
    with session_scope() as s:
        ids = [v.id for v in s.execute(select(Video)).scalars().all()]
    blocks = [c for vid in ids if (c := _video_context(vid))]
    return "\n\n".join(blocks) if blocks else "(no processed footage with commentary yet)"


def _ask_ollama(question: str, context: str) -> str:
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": settings.chat_model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"FOOTAGE DATA:\n{context}\n\nQUESTION: {question}"},
        ],
        "stream": False,
        "options": {"temperature": settings.chat_temperature},
    }
    resp = requests.post(url, json=payload, timeout=settings.ollama_timeout_seconds)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def _ask_claude(question: str, context: str) -> str:
    if not settings.anthropic_api_key:
        raise RuntimeError("chat_provider=claude requires ANTHROPIC_API_KEY")
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=settings.claude_vlm_model,
        max_tokens=600,
        system=_SYSTEM,
        messages=[{"role": "user", "content": f"FOOTAGE DATA:\n{context}\n\nQUESTION: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def derive_scene(descriptions: List[str]) -> str:
    """Name the setting/scene in 1-3 words from the activity descriptions (local LLM)."""
    joined = " ".join(d for d in descriptions if d)[:1500]
    if not joined:
        return ""
    prompt = (
        "These describe moments in a worker's first-person video:\n" + joined +
        "\n\nIn 1-3 words, name the overall setting/scene (e.g. 'Car service', "
        "'Warehouse', 'Kitchen'). Reply with ONLY the scene name."
    )
    try:
        url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        r = requests.post(url, json={
            "model": settings.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.1},
        }, timeout=settings.ollama_timeout_seconds)
        r.raise_for_status()
        ans = r.json().get("message", {}).get("content", "").strip()
        return ans.split("\n")[0].strip(' ".').strip()[:60]
    except Exception:
        return ""


def answer_question(question: str, video_id: Optional[str] = None) -> dict:
    context = build_context(video_id)
    provider = settings.chat_provider.lower()
    answer = _ask_claude(question, context) if provider == "claude" else _ask_ollama(question, context)
    log.info("chat (%s/%s): %r", provider, settings.chat_model, question[:80])
    return {"answer": answer, "provider": provider, "model": settings.chat_model,
            "grounded_on_videos": 1 if video_id else None}
