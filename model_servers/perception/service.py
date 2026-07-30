"""Debug-only FastAPI wrapper around Perception.

This is NOT the live path — ../../backend/app/pipeline.py imports `Perception`
in-process instead (see ../README.md §1.3). This exists so you can `curl`
SAM 3 + DINOv3 directly, independent of the full ingest/tracking/VLM pipeline.

Start with:  docker compose --profile debug up -d perception-api
"""

from __future__ import annotations

import io
import json
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from interface import Concept, Detection, Perception

app = FastAPI(title="uav-perception-debug")
_percept: Optional[Perception] = None


@app.on_event("startup")
def _load() -> None:
    global _percept
    _percept = Perception.from_env()


@app.get("/health")
def health():
    return {"status": "ok" if _percept else "loading", "backend": _percept.backend if _percept else None}


def _read_frame(image: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(image)).convert("RGB"))


@app.post("/detect_track")
async def detect_track(image: UploadFile = File(...), concepts: str = Form(...)):
    """`concepts` is a JSON list of Concept fields, e.g.
    '[{"label": "person", "text_prompt": "person"}]'
    """
    frame = _read_frame(await image.read())
    concept_list = [Concept(**c) for c in json.loads(concepts)]
    dets = _percept.detect_track(frame, concept_list)
    return {"detections": [d.__dict__ for d in dets]}


@app.post("/embed")
async def embed(image: UploadFile = File(...), detections: str = Form(...)):
    """`detections` is a JSON list of Detection fields (as returned by /detect_track)."""
    frame = _read_frame(await image.read())
    det_list = [Detection(**d) for d in json.loads(detections)]
    out = _percept.embed(frame, det_list)
    return {"detections": [d.__dict__ for d in out]}
