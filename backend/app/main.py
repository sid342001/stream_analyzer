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
from collections import deque
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
_stop_event = threading.Event()
_connections: set[WebSocket] = set()

# Message types whose payload is a full-state snapshot: a newer one totally
# supersedes an older one, so an undelivered old one is worthless and is
# coalesced away rather than queued. `event` is deliberately NOT in here —
# events are the product, and dropping one is a correctness failure.
_COALESCING = ("frame", "tracks", "telemetry")
# Drain order: events first, video last. A flood of ~100KB frames must never
# delay, let alone evict, an event — which the previous single shared queue
# allowed.
_DRAIN_ORDER = ("event", "telemetry", "tracks", "frame")


class _Outbox:
    """Per-type outbound buffering, replacing the old single shared queue.

    The old design had one Queue(maxsize=1000) with type-blind drop-oldest, so
    a burst of display frames could silently evict an event. Here snapshot
    types get a one-slot coalescing buffer and events get a real queue.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, str] = {}
        self._events: deque[str] = deque(maxlen=1000)
        self._wakeup = threading.Event()

    def publish(self, message: dict[str, Any]) -> None:
        """Called from pipeline threads. Never blocks."""
        kind = message.get("type", "")
        # Serialize on the producer thread so the event loop does no CPU work
        # per message.
        text = json.dumps(message)
        with self._lock:
            if kind in _COALESCING:
                self._latest[kind] = text
            else:
                if len(self._events) == self._events.maxlen:
                    logger.error("event backlog full — dropping oldest event")
                self._events.append(text)
        self._wakeup.set()

    def drain(self) -> list[str]:
        with self._lock:
            out = list(self._events)
            self._events.clear()
            for kind in _DRAIN_ORDER:
                if kind == "event":
                    continue
                text = self._latest.pop(kind, None)
                if text is not None:
                    out.append(text)
            return out

    def wait(self, timeout: float) -> bool:
        got = self._wakeup.wait(timeout)
        if got:
            self._wakeup.clear()
        return got


_outbox = _Outbox()


def _on_pipeline_message(message: dict[str, Any]) -> None:
    _outbox.publish(message)


async def _broadcast_loop() -> None:
    while True:
        # Block in a worker thread so the event loop stays free; one wait per
        # batch rather than the old one-threadpool-dispatch-per-message.
        got = await asyncio.to_thread(_outbox.wait, 0.5)
        if not got:
            continue
        messages = _outbox.drain()
        if not messages or not _connections:
            continue
        dead = set()
        for ws in list(_connections):
            try:
                for text in messages:
                    await ws.send_text(text)
            except Exception:
                dead.add(ws)
        _connections.difference_update(dead)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()
    concept_store.seed(["person", "vehicle", "tent", "fire"])

    threads = pipeline.start(_on_pipeline_message, _stop_event)
    broadcast_task = asyncio.create_task(_broadcast_loop())

    yield

    _stop_event.set()
    # Wake any thread blocked waiting for a frame so it can see the stop event.
    pipeline.shutdown()
    broadcast_task.cancel()
    # Join rather than relying on daemon=True: the ingest thread must return
    # under its own power so its generator's `with av.open(...)` unwinds and
    # frees the UDP socket. Bounded by ingest.py's read timeout.
    for t in threads:
        t.join(timeout=15)
        if t.is_alive():
            logger.warning("thread %s did not exit cleanly", t.name)


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
    if pipeline.perception is None:
        raise HTTPException(status_code=503, detail="perception models still loading")

    try:
        # int(): Perception._crop slices the frame with these, and numpy
        # rejects float slice indices.
        box_coords = tuple(int(v) for v in json.loads(box))
        if len(box_coords) != 4:
            raise ValueError
    except (json.JSONDecodeError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="box must be a JSON [x1, y1, x2, y2] array")

    raw = await image.read()

    def _embed_job() -> list[float]:
        # Runs on the detection thread — the only thread allowed to touch
        # CUDA. Image decoding is in here too so it never lands on the event
        # loop.
        frame = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        det = Detection(track_id=0, concept=record.label, box=box_coords, mask_rle="", score=1.0)
        [embedded] = pipeline.perception.embed(frame, [det])
        return embedded.embedding

    try:
        future = pipeline.submit_gpu(_embed_job)
    except queue.Full:
        raise HTTPException(status_code=503, detail="GPU queue saturated, retry shortly")

    try:
        embedding = await asyncio.wait_for(asyncio.wrap_future(future), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="embedding timed out")

    updated = concept_store.add_exemplar_embedding(concept_id, embedding)
    assert updated is not None
    return updated.to_public()


app.include_router(api)
