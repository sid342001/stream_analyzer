"""FastAPI orchestrator: ingest + perception + VLM -> WebSocket, and the
context-authoring REST API. See ../README.md for what's real vs. not yet
wired into the frontend.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel

from app.config import settings
from app.pipeline import Pipeline
from app.schemas import Concept
from app.store import concepts as concept_store
from perception import Detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

pipeline = Pipeline(settings, concept_store)
_message_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1000)
_stop_event = threading.Event()
_connections: set[WebSocket] = set()


def _on_pipeline_message(message: dict[str, Any]) -> None:
    """Called from the background pipeline thread — must stay non-blocking
    and thread-safe. Drop-oldest under backpressure rather than block the
    inference loop on a slow WebSocket consumer."""
    try:
        _message_queue.put_nowait(message)
    except queue.Full:
        try:
            _message_queue.get_nowait()
        except queue.Empty:
            pass
        _message_queue.put_nowait(message)


async def _broadcast_loop() -> None:
    while True:
        message = await asyncio.to_thread(_message_queue.get)
        if not _connections:
            continue
        text = json.dumps(message)
        dead = set()
        for ws in _connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.add(ws)
        _connections.difference_update(dead)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()
    concept_store.seed(["person", "vehicle", "tent", "fire"])

    pipeline_thread = threading.Thread(
        target=pipeline.run, args=(_on_pipeline_message, _stop_event), daemon=True
    )
    pipeline_thread.start()
    broadcast_task = asyncio.create_task(_broadcast_loop())

    yield

    _stop_event.set()
    broadcast_task.cancel()


app = FastAPI(title="uav-stream-analyzer-backend", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """Streams {"type": "tracks"|"telemetry"|"event", "data": ...} messages —
    see app/pipeline.py's module docstring for the exact wire shape. Consumed
    by frontend/src/lib/wsFeed.ts's `WsFeed`, the `Feed` implementation wired
    into App.tsx.
    """
    await websocket.accept()
    _connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # no client->server messages defined yet; just detect disconnect
    except WebSocketDisconnect:
        pass
    finally:
        _connections.discard(websocket)


# `/api` prefix matches frontend/vite.config.ts's dev-server proxy (`/api` ->
# this service, `/ws` -> the websocket above, proxied separately since it
# needs the `ws: true` upgrade flag).
api = APIRouter(prefix="/api")


@api.get("/concepts", response_model=list[Concept])
def list_concepts() -> list[Concept]:
    return [c.to_public() for c in concept_store.list()]


class CreateConceptRequest(BaseModel):
    label: str
    priority: str = "watch"


@api.post("/concepts", response_model=Concept)
def create_concept(body: CreateConceptRequest) -> Concept:
    record = concept_store.add(label=body.label, priority=body.priority)
    return record.to_public()


class UpdateConceptRequest(BaseModel):
    enabled: bool | None = None
    priority: str | None = None


@api.patch("/concepts/{concept_id}", response_model=Concept)
def update_concept(concept_id: str, body: UpdateConceptRequest) -> Concept:
    """Backs the watch-list's enable toggle and priority cycle (see
    frontend/src/components/LeftRail.tsx) — the pipeline reads
    `concept_store.enabled()` fresh every sampled frame, so a toggle here
    takes effect on the next detection pass with no extra signaling needed.
    """
    record = concept_store.get(concept_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown concept_id")
    if body.enabled is not None:
        record = concept_store.set_enabled(concept_id, body.enabled)
    if body.priority is not None:
        record = concept_store.set_priority(concept_id, body.priority)
    return record.to_public()


@api.post("/concepts/{concept_id}/exemplars", response_model=Concept)
async def add_exemplar(
    concept_id: str,
    image: UploadFile = File(...),
    box: str = Form(..., description='JSON [x1, y1, x2, y2] in the image\'s own pixel coordinates'),
) -> Concept:
    """Context authoring, live: draw a box on a reference image, get back the
    updated concept with its exemplar count bumped. Mirrors
    model_servers/perception/tools/embed_concept.py's logic, but against
    live-loaded weights instead of a one-off script — same DINOv3-embedding
    approach, see that tool's docstring for why (SAM 3's own exemplar-box
    prompting API isn't confirmed yet).
    """
    record = concept_store.get(concept_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown concept_id")

    try:
        box_coords = tuple(json.loads(box))
        if len(box_coords) != 4:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="box must be a JSON [x1, y1, x2, y2] array")

    frame = np.array(Image.open(io.BytesIO(await image.read())).convert("RGB"))
    fake_detection = Detection(track_id=0, concept=record.label, box=box_coords, mask_rle="", score=1.0)
    [embedded] = pipeline.perception.embed(frame, [fake_detection])

    updated = concept_store.add_exemplar_embedding(concept_id, embedded.embedding)
    assert updated is not None
    return updated.to_public()


app.include_router(api)
