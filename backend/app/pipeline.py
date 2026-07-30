"""Tier-1 (every sampled frame) + Tier-2 (first appearance) pipeline.

Runs synchronously in a background thread (started from app/main.py's
lifespan) — PyAV decoding and SAM3/DINOv3 inference are both blocking/GPU-bound
and gain nothing from asyncio. Results are handed to `on_message` as plain
dicts; app/main.py's WebSocket layer is the only thing that knows about
asyncio or connected clients.

Wire protocol (one JSON object per WebSocket message, matching the `Feed`
handlers frontend/src/lib/feed.ts declares):
    {"type": "frame",     "data": "data:image/jpeg;base64,..."}
    {"type": "tracks",    "data": [TrackedObject, ...]}
    {"type": "telemetry", "data": Telemetry}
    {"type": "event",     "data": EventItem}
`frame` is broadcast at settings.display_fps, independently of
settings.sample_fps (the detection rate) — video plays smoothly even though
SAM3/DINOv3 only run a few times a second. See frontend/src/lib/wsFeed.ts.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from app import ingest, vlm_client
from app.config import Settings
from app.schemas import EventItem, Telemetry, TrackedObject
from app.store import ConceptStore
from app.tracker import Tracker
from perception import Concept, Perception

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict[str, Any]], None]


def _cosine(a: list[float], b: list[float]) -> float:
    av, bv = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


def _crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    return frame[y1:y2, x1:x2]


class Pipeline:
    """Owns the perception model and the per-stream tracker/telemetry state."""

    def __init__(self, settings: Settings, store: ConceptStore) -> None:
        self.settings = settings
        self.store = store
        self.tracker = Tracker(
            iou_threshold=settings.track_iou_threshold,
            max_missed_frames=settings.track_max_missed_frames,
        )
        self.last_telemetry = Telemetry(lat=0.0, lon=0.0, alt=0.0, hdg=0.0, speed=0.0)
        # Loaded once at startup — see app/main.py's lifespan. Not created
        # here so failures during model loading surface before the server
        # reports itself healthy.
        self.perception: Perception | None = None

    def load(self) -> None:
        logger.info("loading SAM 3 + DINOv3 (backend=%s)...", self.settings.perception_backend)
        self.perception = Perception(
            backend=self.settings.perception_backend,
            sam_weights=self.settings.sam_weights,
            dinov3_weights=self.settings.dinov3_weights,
        )
        logger.info("perception ready")

    def run(self, on_message: OnMessage, stop: threading.Event) -> None:
        """Blocking — call this from a background thread, not the event loop."""
        assert self.perception is not None, "call load() before run()"

        while not stop.is_set():
            try:
                self._ingest_and_process(on_message)
            except Exception:
                logger.exception("ingest/pipeline loop crashed, retrying in 3s")
                time.sleep(3)

    def _ingest_and_process(self, on_message: OnMessage) -> None:
        detect_interval = 1.0 / self.settings.sample_fps
        display_interval = 1.0 / self.settings.display_fps
        last_detect = 0.0
        last_display = 0.0

        for frame, telemetry in ingest.frames(self.settings.stream_url):
            if telemetry is not None:
                self.last_telemetry = Telemetry(**telemetry)
                on_message({"type": "telemetry", "data": self.last_telemetry.model_dump(by_alias=True)})
                continue

            assert frame is not None
            now = time.monotonic()

            if now - last_display >= display_interval:
                last_display = now
                self._broadcast_frame(frame, on_message)

            if now - last_detect >= detect_interval:
                last_detect = now
                self._process_frame(frame, on_message)

    def _broadcast_frame(self, frame: np.ndarray, on_message: OnMessage) -> None:
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.settings.display_jpeg_quality],
        )
        if not ok:
            return
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        on_message({"type": "frame", "data": data_url})

    def _process_frame(self, frame: np.ndarray, on_message: OnMessage) -> None:
        h, w = frame.shape[:2]
        enabled = self.store.enabled()
        if not enabled:
            return

        concepts = [Concept(label=c.label, text_prompt=c.text_prompt) for c in enabled]
        detections = self.perception.detect_track(frame, concepts)
        if not detections:
            self._broadcast_tracks([], w, h, on_message)
            return
        detections = self.perception.embed(frame, detections)
        detections = self._filter_by_exemplars(detections, enabled)
        if not detections:
            self._broadcast_tracks([], w, h, on_message)
            return

        raw = [(d.concept, d.box, d.score) for d in detections]
        tracks = self.tracker.update(raw, w, h)
        self._broadcast_tracks(tracks, w, h, on_message)

        by_label = {c.label: c for c in enabled}
        for track in tracks:
            if not track.is_new:
                continue
            concept_record = by_label.get(track.concept)
            if concept_record is None:
                continue
            self._fire_first_appearance(frame, track, concept_record.color, on_message)

    def _filter_by_exemplars(self, detections: list, enabled: list) -> list:
        """DINOv3 similarity gate — see config.match_threshold's docstring.
        Concepts with no authored exemplars pass every SAM 3 text-prompt
        candidate through unfiltered."""
        by_label = {c.label: c for c in enabled}
        kept = []
        for det in detections:
            record = by_label.get(det.concept)
            if record is None or not record.reference_embeddings:
                kept.append(det)
                continue
            best = max(_cosine(det.embedding, ref) for ref in record.reference_embeddings)
            if best >= self.settings.match_threshold:
                kept.append(det)
        return kept

    def _broadcast_tracks(self, tracks: list, frame_w: int, frame_h: int, on_message: OnMessage) -> None:
        by_label_color = {c.label: c.color for c in self.store.enabled()}
        payload = []
        for t in tracks:
            x1, y1, x2, y2 = t.box
            payload.append(
                TrackedObject(
                    id=t.track_id,
                    concept=t.concept,
                    color=by_label_color.get(t.concept, "#94a3b8"),
                    x=t.norm_x,
                    y=t.norm_y,
                    w=(x2 - x1) / frame_w,
                    h=(y2 - y1) / frame_h,
                    vx=t.vx,
                    vy=t.vy,
                    score=t.score,
                    trail=[{"x": px, "y": py} for px, py in t.trail],
                ).model_dump(by_alias=True)
            )
        on_message({"type": "tracks", "data": payload})

    def _fire_first_appearance(self, frame: np.ndarray, track, color: str, on_message: OnMessage) -> None:
        crop = _crop(frame, track.box)
        description = vlm_client.describe_crop(
            self.settings.vlm_base_url, self.settings.vlm_model, crop, track.concept, self.settings.vlm_timeout_s
        )
        event = EventItem(
            id=f"ev_{track.track_id}_{int(time.time() * 1000)}",
            ts=int(time.time() * 1000),
            concept=track.concept,
            concept_color=color,
            severity="info",
            kind="first appearance",
            description=description,
            verdict="pending",
            tier=2,
            deep_analyzed=False,
            track_id=track.track_id,
            telemetry=self.last_telemetry,
        )
        on_message({"type": "event", "data": event.model_dump(by_alias=True)})
