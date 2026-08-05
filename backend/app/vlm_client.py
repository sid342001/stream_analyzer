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

# One pooled client rather than a fresh connection (and TCP handshake) per
# call. Shared across the VLM thread pool — httpx.Client is thread-safe.
_client = httpx.Client(timeout=30.0)


def _encode_jpeg_data_url(crop: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("failed to encode crop as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_data_url(data_url: str) -> np.ndarray:
    """Inverse of _encode_jpeg_data_url — returns BGR (cv2's native order),
    since the only caller draws a cv2.rectangle on the result immediately."""
    _, _, b64data = data_url.partition(",")
    buf = np.frombuffer(base64.b64decode(b64data), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode image data URL")
    return bgr


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


_MAX_CHAT_HISTORY_TURNS = 8
_MAX_EXEMPLARS_PER_CALL = 3

# Sent as the first message on every Tier-2/Tier-3 call. Establishes the
# domain (degraded aerial footage is normal, not a defect) and — the part
# that actually matters for triage — instructs the model to say when it
# can't tell rather than fabricate a confident-sounding guess. No structured
# output contract on top of this (no required "CONFIRMED:"/"UNCERTAIN:"
# prefix): VLMs don't reliably hold to strict formats, and the plain-language
# version of this ("too blurry to determine...") already worked in testing.
_SYSTEM_PROMPT = (
    "You are analyzing real-time UAV/drone surveillance footage (EO or "
    "thermal). Expect blur, low resolution, distance, and partial occlusion "
    "— that's normal for aerial footage, not a defect. Only state a "
    "specific finding if the visual evidence clearly supports it; if the "
    "image is too degraded or ambiguous to tell, say so plainly and "
    "describe what can and can't be determined, rather than guessing. Focus "
    "on anything that stands out as unusual or worth an operator's "
    "attention."
)


def _exemplar_content_blocks(concept: str, exemplars: list | None) -> list[dict]:
    """Reference-image blocks shown *before* the live crop/question, so the
    model has a concrete example of what "<concept>" means here instead of
    guessing from the label alone — this is what lets it recognize a
    composite, relationship-based thing (e.g. "people covering a solar
    panel") that no single object class captures. `exemplars` is
    `ConceptRecord.exemplars` (see app/store.py) — a list of objects with an
    `.image` attribute (base64 JPEG data URL); duck-typed rather than
    type-imported to keep this module decoupled from the store's internals.
    Capped since each image costs real tokens/bandwidth per call.
    """
    if not exemplars:
        logger.info("no exemplars for concept=%r, no reference images attached", concept)
        return []
    logger.info("attaching %d/%d exemplar reference image(s) for concept=%r",
               min(len(exemplars), _MAX_EXEMPLARS_PER_CALL), len(exemplars), concept)
    blocks: list[dict] = [
        {"type": "text", "text": f"Reference example(s) of what '{concept}' looks like:"}
    ]
    for ex in exemplars[:_MAX_EXEMPLARS_PER_CALL]:
        image = getattr(ex, "image", None)
        if image:
            blocks.append({"type": "image_url", "image_url": {"url": image}})
    return blocks


def answer_question(base_url: str, model: str, image_data_url: str | None, box: list[int],
                    color_hex: str, question: str, history: list[dict], timeout_s: float,
                    concept: str = "target", exemplars: list | None = None) -> str:
    """Tier-3: grounded follow-up chat about a specific event.

    `history` is prior turns as [{role, content}] (no image — only text),
    already accumulated client-side; this endpoint is stateless. The marked
    image (and any exemplar reference images) are attached to the *first*
    user turn across (history + this question) only — re-attaching them on
    every turn burns context for no gain once the model already has them
    in-conversation.
    """
    marked_url = None
    if image_data_url:
        try:
            bgr = _decode_data_url(image_data_url)
            if box and len(box) == 4:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), _hex_to_bgr(color_hex), 2)
            marked_url = _encode_jpeg_data_url(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            logger.exception("failed to prepare marked image for chat, continuing without it")

    turns = list(history[-_MAX_CHAT_HISTORY_TURNS:]) + [{"role": "user", "content": question}]
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    image_attached = False
    for turn in turns:
        role, content = turn.get("role", "user"), turn.get("content", "")
        if role == "user" and not image_attached and marked_url is not None:
            blocks = _exemplar_content_blocks(concept, exemplars) + [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": marked_url}},
            ]
            messages.append({"role": "user", "content": blocks})
            image_attached = True
        else:
            messages.append({"role": role, "content": content})

    try:
        resp = _client.post(
            f"{base_url}/chat/completions",
            json={"model": model, "messages": messages, "max_tokens": 256},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        logger.warning("VLM chat call failed: %s", exc)
        return "Sorry, I couldn't reach the analysis model just now. Please try again."


def describe_scene(base_url: str, model: str, frame: np.ndarray, timeout_s: float) -> str:
    """Tier-2: a short general overview of the whole frame, run periodically
    (see pipeline.py's _queue_scene_analysis) instead of once per detected
    object — not scoped to any single concept, so no exemplar reference
    images here (there's no one concept to show an example of).

    Kept short (a few sentences, not a paragraph) — per
    docs/Architecture_and_Technology.md, event cards want a compact
    description.
    """
    image_url = _encode_jpeg_data_url(frame)
    prompt = (
        "Give a short general overview (2-4 sentences) of what's visible and happening "
        "in this frame as a whole — notable objects, activity, and anything that stands "
        "out as worth an operator's attention. Not a play-by-play of every object, just "
        "the overall picture."
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    try:
        resp = _client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 150,
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        logger.warning("VLM scene call failed: %s", exc)
        return "No notable activity."
