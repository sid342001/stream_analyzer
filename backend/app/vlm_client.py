"""Tier-2/3 calls to the VLM (vllm or the llama.cpp standby) — both speak the
same OpenAI-compatible /v1/chat/completions API (see model_servers/README.md
§1.1/§1.2), so this client doesn't need to know which one is actually running.
"""

from __future__ import annotations

import base64
import logging

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


def _encode_jpeg_data_url(crop: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("failed to encode crop as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def describe_crop(base_url: str, model: str, crop: np.ndarray, concept: str, timeout_s: float) -> str:
    """Tier-2: one terse grounded line about a single detection crop.

    Kept deliberately short — per docs/Architecture_and_Technology.md, event
    cards want a one-line description, not a paragraph.
    """
    image_url = _encode_jpeg_data_url(crop)
    prompt = f"One short grounded sentence: what is this {concept} doing?"
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": 64,
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        logger.warning("VLM call failed, falling back to a plain label: %s", exc)
        return f"{concept.capitalize()} detected."
