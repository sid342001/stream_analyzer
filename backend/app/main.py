"""FastAPI orchestrator: ingest + perception + VLM -> WebSocket, and the
context-authoring REST API. See ../README.md for what's real vs. not yet
wired into the frontend.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import queue
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field

from app import vlm_client
from app.config import settings
from app.pipeline import Pipeline, context_crop, encode_rgb_jpeg_data_url
from app.schemas import Camel, Concept
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
_COALESCING = ("frame", "tracks", "telemetry", "status")
# Drain order: events first, video last. A flood of ~100KB frames must never
# delay, let alone evict, an event — which the previous single shared queue
# allowed.
#
# `status` must appear here too, not just in _COALESCING above — drain()
# only pops kinds it finds in this tuple, so a kind stored in _latest but
# missing from _DRAIN_ORDER is buffered forever and never delivered, with no
# error anywhere to reveal the mistake. (Caught in review before this ever
# shipped — see the plan's "Corrections made in review" section.)
_DRAIN_ORDER = ("event", "status", "telemetry", "tracks", "frame")

_MAX_EXEMPLAR_UPLOAD_BYTES = 15 * 1024 * 1024
_MAX_EXEMPLAR_UPLOAD_WIDTH = 2048


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

# Status messages only fire on state *transitions* (see pipeline.py), so a
# client that connects mid-stream — the common case, since the stream is
# usually already live before any browser tab opens — would otherwise never
# learn the current state and get stuck showing "unknown" forever. Track the
# latest one here so ws_endpoint can replay it to every new connection.
_last_status: dict[str, Any] | None = None


def _on_pipeline_message(message: dict[str, Any]) -> None:
    global _last_status
    if message.get("type") == "status":
        _last_status = message
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
    if _last_status is not None:
        await websocket.send_text(json.dumps(_last_status))
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


class SetStreamRequest(BaseModel):
    url: str


@api.post("/stream")
def set_stream(body: SetStreamRequest) -> dict[str, str]:
    """Swap the live UDP source at runtime — see pipeline.request_reconnect()
    for the mechanics. The frontend watches the `status` WS message for the
    outcome rather than this response, since connecting is asynchronous.
    """
    pipeline.request_reconnect(body.url)
    return {"status": "reconnecting"}


class HiresFrameResponse(Camel):
    image: str


@api.get("/frame/hires", response_model=HiresFrameResponse)
async def get_hires_frame() -> HiresFrameResponse:
    """One-shot full-ingest-resolution frame for context authoring.

    Deliberately not the display stream: that's downscaled to
    display_max_width for bandwidth, but exemplars are embedded against
    full-resolution frames (see pipeline.hires_slot's docstring), so boxing
    the smaller display frame would create exemplars from a measurably
    blurrier source than what live detections are compared against.
    """
    item = await asyncio.to_thread(pipeline.hires_slot.take, 2.0)
    if item is None:
        raise HTTPException(status_code=503,
                            detail="no live frame available; is the stream connected?")
    bgr = cv2.cvtColor(item.rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode frame")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return HiresFrameResponse(image="data:image/jpeg;base64," + b64)


class TrackSnapshotResponse(Camel):
    image: str
    box: list[int]
    concept: str
    concept_color: str = Field(alias="conceptColor")


@api.get("/tracks/{track_id}/snapshot", response_model=TrackSnapshotResponse)
async def get_track_snapshot(track_id: int) -> TrackSnapshotResponse:
    """On-demand query support for the Objects tab. Individual objects
    aren't automatically analyzed anymore (see pipeline.py's
    _queue_scene_analysis — Tier-2 is now a periodic whole-frame narrative),
    so clicking one needs a *fresh* crop captured right now, not a stored
    one from the past.

    Frame fetched *before* the box lookup, not after: the hi-res capture can
    take up to 2s to arrive, so reading the box afterward — as close as
    possible to when it's actually used — gives the freshest alignment
    between "where the box says the object is" and "what's in the frame."
    """
    item = await asyncio.to_thread(pipeline.hires_slot.take, 2.0)
    if item is None:
        raise HTTPException(status_code=503,
                            detail="no live frame available; is the stream connected?")

    hit = pipeline.tracker.get_box(track_id)
    if hit is None:
        raise HTTPException(status_code=404, detail="track not currently live")
    concept, box = hit

    try:
        crop, box_in_crop = context_crop(item.rgb, box, settings.context_crop_margin)
        image = encode_rgb_jpeg_data_url(crop)
    except Exception:
        raise HTTPException(status_code=500, detail="failed to prepare snapshot")

    record = next((c for c in concept_store.list() if c.label == concept), None)
    color = record.color if record else "#94a3b8"

    return TrackSnapshotResponse(image=image, box=box_in_crop, concept=concept, concept_color=color)


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
    if len(raw) > _MAX_EXEMPLAR_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large (max 15MB)")

    def _embed_job() -> tuple[list[float], str, list[int]]:
        # Runs on the detection thread — the only thread allowed to touch
        # CUDA. Image decoding is in here too so it never lands on the event
        # loop. The thumbnail build (context_crop/encode) is cheap CPU work
        # piggybacked on the same job rather than a second round-trip.
        frame = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        coords = box_coords
        if frame.shape[1] > _MAX_EXEMPLAR_UPLOAD_WIDTH:
            # An operator dragging in a huge photo would otherwise stall
            # detection for the whole decode+embed; DINOv3 resizes to a
            # fixed input size anyway, so this costs no accuracy.
            scale = _MAX_EXEMPLAR_UPLOAD_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
            coords = tuple(int(v * scale) for v in box_coords)
        det = Detection(track_id=0, concept=record.label, box=coords, mask_rle="", score=1.0)
        [embedded] = pipeline.perception.embed(frame, [det])
        # Small margin + small max_side: this is "what does the concept look
        # like" for the VLM/YOLOE, a thumbnail — not evidence-quality like
        # the Tier-2/3 context crop (see pipeline.context_crop's docstring).
        thumb_crop, thumb_box = context_crop(frame, coords, margin=0.3, max_side=320)
        thumb_image = encode_rgb_jpeg_data_url(thumb_crop)
        return embedded.embedding, thumb_image, thumb_box

    try:
        future = pipeline.submit_gpu(_embed_job)
    except queue.Full:
        raise HTTPException(status_code=503, detail="GPU queue saturated, retry shortly")

    try:
        embedding, thumb_image, thumb_box = await asyncio.wait_for(asyncio.wrap_future(future), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="embedding timed out")

    updated = concept_store.add_exemplar(concept_id, thumb_image, thumb_box, embedding)
    assert updated is not None
    return updated.to_public()


class ChatTurn(Camel):
    role: str
    content: str


class ChatRequest(Camel):
    """No event id in the path: the backend keeps no server-side event
    record (consistent with "no DB"), so an id it can't look anything up
    with would be decoration on the API surface. `image`/`box` are the
    values the frontend already has from the event it's chatting about.
    `concept` is the event's concept label — used to look up that concept's
    stored exemplar images (see app/store.py's Exemplar) so the VLM gets
    "what this concept looks like" reference images, not just the live crop.
    """
    question: str
    image: str | None = None
    box: list[int] = Field(default_factory=list)
    concept: str = ""
    concept_color: str = Field(default="#94a3b8", alias="conceptColor")
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(Camel):
    answer: str


@api.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    """Tier-3: grounded follow-up chat about a specific event's evidence
    image. Pure CPU + an HTTP call, no CUDA, so this goes through the VLM
    pool (shared admission control with Tier-2 descriptions) rather than
    submit_gpu, which is CUDA-only.
    """
    history = [{"role": t.role, "content": t.content} for t in body.history]
    record = concept_store.get_by_label(body.concept) if body.concept else None
    exemplars = record.exemplars if record is not None else None

    def _answer() -> str:
        return vlm_client.answer_question(
            settings.vlm_base_url, settings.vlm_model, body.image, body.box,
            body.concept_color, body.question, history, settings.vlm_timeout_s,
            concept=body.concept or "target", exemplars=exemplars,
        )

    try:
        future = pipeline.submit_vlm(_answer)
    except AssertionError:
        raise HTTPException(status_code=503, detail="VLM pool not ready")
    try:
        answer = await asyncio.wait_for(asyncio.wrap_future(future), timeout=settings.vlm_timeout_s + 5)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="chat timed out")
    return ChatResponse(answer=answer)


app.include_router(api)
