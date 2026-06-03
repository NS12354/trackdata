"""Vision-model providers for task segmentation.

Two interchangeable backends classify a single frame against the waste-services
task taxonomy:

  * OllamaVLMProvider  — a LOCAL vision model (Ollama). Free, and no frame ever
    leaves the machine. Default.
  * ClaudeVLMProvider  — the Anthropic API. Higher quality, paid; opt-in only.

Each returns a FrameLabel (task, confidence, description) plus a per-call USD
cost (0.0 for the local model). Keeping a small interface lets us swap providers
by config without touching the segmentation logic.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

from config import settings

# Waste-services task taxonomy (from the project spec).
TASK_TAXONOMY: List[str] = [
    "approaching property",
    "moving container",
    "opening gate/enclosure",
    "manipulating lock or latch",
    "handling overflow/contamination",
    "loading/unloading",
    "transit/walking",
    "idle/waiting",
]
_TAXONOMY_LOWER = {t.lower(): t for t in TASK_TAXONOMY}

# Claude follows structured-output instructions well, so we ask it for JSON.
_JSON_PROMPT = (
    "You are analyzing a single frame from a waste-services field worker's "
    "chest-mounted camera. Decide which ONE of these tasks best matches what is "
    "happening in the frame:\n"
    + "\n".join(f"- {t}" for t in TASK_TAXONOMY)
    + "\n\nRespond ONLY with compact JSON of the form "
    '{"task": "<one task from the list, verbatim>", '
    '"confidence": <0.0-1.0>, "description": "<one short sentence>"}.'
)

# Small local VLMs are weak at structured output but good at captioning, so we
# ask for a description that mentions the activity, then map it to the taxonomy
# locally (see OllamaVLMProvider).
_CAPTION_PROMPT = (
    "This is one frame from a waste-services worker's chest-mounted camera. "
    "In one sentence, describe what the worker is doing and what objects they are "
    "handling. The activity is likely one of: "
    + ", ".join(TASK_TAXONOMY) + "."
)


@dataclass
class FrameLabel:
    task: str
    confidence: float
    description: str
    raw: str = ""


def _match_taxonomy(value: str) -> Optional[str]:
    """Map a model's free-text task back onto a canonical taxonomy label."""
    if not value:
        return None
    v = value.strip().lower()
    if v in _TAXONOMY_LOWER:
        return _TAXONOMY_LOWER[v]
    # Loose containment / keyword match.
    for low, canon in _TAXONOMY_LOWER.items():
        if low in v or v in low:
            return canon
    keywords = {
        "approach": "approaching property", "container": "moving container",
        "bin": "moving container", "gate": "opening gate/enclosure",
        "enclosure": "opening gate/enclosure", "lock": "manipulating lock or latch",
        "latch": "manipulating lock or latch", "overflow": "handling overflow/contamination",
        "contamin": "handling overflow/contamination", "load": "loading/unloading",
        "unload": "loading/unloading", "walk": "transit/walking",
        "transit": "transit/walking", "idle": "idle/waiting", "wait": "idle/waiting",
    }
    for kw, canon in keywords.items():
        if kw in v:
            return canon
    return None


def _parse_label(text: str) -> FrameLabel:
    """Parse a model response (ideally JSON) into a FrameLabel, defensively."""
    task, conf, desc = None, 0.5, ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            task = _match_taxonomy(str(obj.get("task", "")))
            desc = str(obj.get("description", "")).strip()
            try:
                conf = float(obj.get("confidence", conf))
            except (TypeError, ValueError):
                pass
        except json.JSONDecodeError:
            pass
    if task is None:
        task = _match_taxonomy(text) or "idle/waiting"
        conf = min(conf, 0.3)  # low confidence when we had to guess
    conf = max(0.0, min(1.0, conf))
    return FrameLabel(task=task, confidence=round(conf, 3), description=desc, raw=text[:500])


class OllamaVLMProvider:
    """Local vision model via the Ollama HTTP API. Free; no data egress."""

    name = "ollama"

    def __init__(self):
        self.model = settings.ollama_vlm_model
        self.url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        self.timeout = settings.ollama_timeout_seconds

    def classify(self, jpeg_bytes: bytes) -> tuple[FrameLabel, float]:
        b64 = base64.b64encode(jpeg_bytes).decode()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _CAPTION_PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0},
        }
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        caption = resp.json().get("message", {}).get("content", "").strip()
        # Map the free-text caption onto the taxonomy locally.
        task = _match_taxonomy(caption)
        if task is None:
            label = FrameLabel(task="idle/waiting", confidence=0.25,
                               description=caption, raw=caption[:500])
        else:
            label = FrameLabel(task=task, confidence=0.55,
                               description=caption, raw=caption[:500])
        return label, 0.0  # local inference is free

    def healthcheck(self) -> bool:
        try:
            tags = requests.get(
                settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=5
            ).json().get("models", [])
            names = {m.get("name", "").split(":")[0] for m in tags}
            return self.model.split(":")[0] in names
        except Exception:
            return False


class ClaudeVLMProvider:
    """Anthropic vision model. Higher quality, paid — opt-in only."""

    name = "claude"
    # Approx Sonnet pricing (USD per token); used only to log an estimate.
    _IN_PER_TOK = 3.0 / 1_000_000
    _OUT_PER_TOK = 15.0 / 1_000_000

    def __init__(self):
        if not settings.anthropic_api_key:
            raise RuntimeError("segmentation_provider=claude requires ANTHROPIC_API_KEY")
        import anthropic
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_vlm_model

    def classify(self, jpeg_bytes: bytes) -> tuple[FrameLabel, float]:
        b64 = base64.b64encode(jpeg_bytes).decode()
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": _JSON_PROMPT},
                ],
            }],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        cost = (msg.usage.input_tokens * self._IN_PER_TOK
                + msg.usage.output_tokens * self._OUT_PER_TOK)
        return _parse_label(text), cost

    def healthcheck(self) -> bool:
        return bool(settings.anthropic_api_key)


def get_segmentation_provider():
    provider = settings.segmentation_provider.lower()
    if provider == "ollama":
        return OllamaVLMProvider()
    if provider == "claude":
        return ClaudeVLMProvider()
    raise ValueError(f"unknown segmentation_provider {settings.segmentation_provider!r}")
