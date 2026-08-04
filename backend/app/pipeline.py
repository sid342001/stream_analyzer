"""Tier-1 (detection) + Tier-2 (first appearance) pipeline.

## Threading model — read this before changing anything

Four workers, and the split is load-bearing rather than cosmetic:

- **ingest**   owns the PyAV container and does nothing but drain the socket,
               decode, and hand frames off. Any variable-cost work here stalls
               the UDP read, which overruns FFmpeg's circular buffer and
               corrupts the stream — that failure was observed in production
               logs when decode, JPEG encode and inference all shared a thread.
- **encode**   colour-converts, downscales, JPEG-encodes and publishes display
               frames. Split out because encode cost is variable (~3-15ms
               depending on resolution) and would eat the ingest thread's
               per-frame budget.
- **detect**   the *sole* owner of the GPU: SAM 3 -> DINOv3 -> tracker. Also
               drains an inbox of out-of-band GPU jobs (see `submit_gpu`) so
               that exactly one thread ever touches CUDA.
- **vlm pool** Tier-2 description calls. A dead VLM used to freeze video for
               30s per new track because the HTTP call was inline here.

Threads (not asyncio) because PyAV, OpenCV and torch are blocking C extensions
that release the GIL — so these genuinely run in parallel. asyncio cannot
overlap them at all. Please don't "modernise" this to async.

Frames move between threads through `frame_slot.DemandSlot`, a one-slot
rendezvous that cannot build a backlog. See that module for why not a queue.

## Wire protocol (one JSON object per WebSocket message)

    {"type": "frame",     "data": "data:image/jpeg;base64,..."}
    {"type": "tracks",    "data": [TrackedObject, ...]}
    {"type": "telemetry", "data": Telemetry}
    {"type": "event",     "data": EventItem}

`frame` is published at `display_fps`, independently of detection, so video
stays smooth while SAM 3 manages only a few frames a second. Consumed by
frontend/src/lib/wsFeed.ts.
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from app import ingest, vlm_client
from app.config import Settings
from app.frame_slot import DemandSlot, FrameItem
from app.schemas import EventItem, Telemetry, TrackedObject
from app.store import ConceptStore
from app.tracker import Tracker
from perception import Concept, Perception

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict[str, Any]], None]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class _Stats:
    """Approximate throughput/latency counters for the perf log.

    `+= 1` is not atomic under the GIL, so these can undercount slightly under
    contention. That's acceptable for a log line — just don't build logic on
    them.
    """

    def __init__(self) -> None:
        self.decoded = 0
        self.converted = 0
        self.displayed = 0
        self.detected = 0
        self.detect_ms: list[float] = []
        self.encode_ms: list[float] = []
        self.vlm_skipped = 0
        self._last_log = time.monotonic()

    def maybe_log(self, interval_s: float) -> None:
        if interval_s <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_log
        if elapsed < interval_s:
            return

        def _p(values: list[float], pct: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]

        logger.info(
            "perf: decode %.1f/s converted %.1f/s display %.1f/s detect %.1f/s | "
            "detect_ms p50=%.0f p95=%.0f | encode_ms p50=%.0f p95=%.0f | vlm_skipped=%d",
            self.decoded / elapsed, self.converted / elapsed,
            self.displayed / elapsed, self.detected / elapsed,
            _p(self.detect_ms, 0.5), _p(self.detect_ms, 0.95),
            _p(self.encode_ms, 0.5), _p(self.encode_ms, 0.95),
            self.vlm_skipped,
        )
        self.decoded = self.converted = self.displayed = self.detected = 0
        self.detect_ms.clear()
        self.encode_ms.clear()
        self.vlm_skipped = 0
        self._last_log = now


class Pipeline:
    """Owns the perception model, the per-stream tracker, and the worker threads."""

    def __init__(self, settings: Settings, store: ConceptStore) -> None:
        self.settings = settings
        self.store = store
        self.tracker = Tracker(
            iou_threshold=settings.track_iou_threshold,
            max_missed_s=settings.track_max_missed_s,
        )
        self.last_telemetry = Telemetry(lat=0.0, lon=0.0, alt=0.0, hdg=0.0, speed=0.0)
        # Loaded once at startup by main.py's lifespan, before the server
        # reports itself healthy — so a model-loading failure surfaces there
        # rather than as a mysterious 500 later.
        self.perception: Perception | None = None

        self.display_slot = DemandSlot()
        self.detect_slot = DemandSlot()
        self.stats = _Stats()

        # Out-of-band GPU work (currently: exemplar embedding from the REST
        # API). Routed through the detection thread so exactly one thread ever
        # issues CUDA work — see submit_gpu().
        self._gpu_inbox: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue(maxsize=8)
        self._vlm_pool: ThreadPoolExecutor | None = None
        self._vlm_pending = 0
        self._vlm_lock = threading.Lock()

    def load(self) -> None:
        logger.info("loading SAM 3 + DINOv3 (backend=%s)...", self.settings.perception_backend)
        self.perception = Perception(
            backend=self.settings.perception_backend,
            sam_weights=self.settings.sam_weights,
            dinov3_weights=self.settings.dinov3_weights,
        )
        logger.info("perception ready")

    # ---- out-of-band GPU jobs -------------------------------------------------

    def submit_gpu(self, fn: Callable[[], Any]) -> Future:
        """Queue GPU work to run on the detection thread.

        The detection thread is the only CUDA owner, so callers on other
        threads (notably the asyncio event loop serving the exemplar-upload
        endpoint) must come through here rather than touching `perception`
        directly — otherwise two threads race on one CUDA context.

        Raises queue.Full if the inbox is saturated; the caller should surface
        that as a 503 rather than let a client stall detection.
        """
        fut: Future = Future()
        self._gpu_inbox.put_nowait((fn, fut))
        return fut

    def _drain_gpu_inbox(self) -> None:
        while True:
            try:
                fn, fut = self._gpu_inbox.get_nowait()
            except queue.Empty:
                return
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except Exception as exc:  # noqa: BLE001 - propagate to the awaiting caller
                fut.set_exception(exc)

    def _fail_pending_gpu_jobs(self) -> None:
        """Never leave an awaiting request hanging when the worker exits."""
        while True:
            try:
                _fn, fut = self._gpu_inbox.get_nowait()
            except queue.Empty:
                return
            if not fut.done():
                fut.set_exception(RuntimeError("detection worker stopped"))

    # ---- ingest thread --------------------------------------------------------

    def run_ingest(self, on_message: OnMessage, stop: threading.Event) -> None:
        """Drain the stream and hand frames off. Blocking — run in a thread."""
        backoff = 1.0
        while not stop.is_set():
            try:
                self._ingest_once(on_message, stop)
                backoff = 1.0
            except Exception:
                logger.exception("ingest loop failed, retrying in %.0fs", backoff)
                # stop.wait() rather than sleep() so shutdown doesn't have to
                # wait out the full backoff.
                stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
        logger.info("ingest thread exiting")

    def _ingest_once(self, on_message: OnMessage, stop: threading.Event) -> None:
        # A fresh stream is a fresh scene: stale tracks would IOU-match
        # unrelated content and suppress the first-appearance events that
        # genuinely new objects should fire.
        self.tracker.reset()

        display_interval = 1.0 / max(self.settings.display_fps, 0.1)
        last_display = 0.0
        seq = 0

        gen = ingest.frames(self.settings.stream_url)
        try:
            for video_frame, telemetry in gen:
                if stop.is_set():
                    return

                if telemetry is not None:
                    self.last_telemetry = Telemetry(**telemetry)
                    on_message({"type": "telemetry",
                                "data": self.last_telemetry.model_dump(by_alias=True)})
                    continue

                self.stats.decoded += 1
                now = time.monotonic()

                # Decide *before* converting — to_ndarray() is a full swscale
                # pass plus an allocation, and at a 25-30fps source most frames
                # are wanted by nobody.
                want_display = (now - last_display >= display_interval) and self.display_slot.wanted()
                want_detect = self.detect_slot.wanted()
                if not (want_display or want_detect):
                    continue

                rgb = video_frame.to_ndarray(format="rgb24")
                rgb.flags.writeable = False  # enforce the shared-read-only contract
                self.stats.converted += 1
                item = FrameItem(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0],
                                 seq=seq, captured=now)
                seq += 1

                if want_display and self.display_slot.offer(item):
                    # Stamp only on a successful handoff: if the encoder is
                    # busy we retry on the next frame rather than skipping a
                    # whole interval, so display degrades smoothly.
                    last_display = now
                if want_detect:
                    self.detect_slot.offer(item)

                self.stats.maybe_log(self.settings.perf_log_interval_s)
        finally:
            # Deterministic finalisation: raises GeneratorExit at the yield so
            # ingest.frames()' `with av.open(...)` unwinds and releases the UDP
            # socket. Relying on refcounting here previously leaked the socket
            # and caused "Address already in use" on reconnect.
            #
            # Only safe from the thread that owns the generator — never call
            # this from the lifespan while this thread sits in demux().
            gen.close()

    # ---- display-encode thread ------------------------------------------------

    def run_encoder(self, on_message: OnMessage, stop: threading.Event) -> None:
        while not stop.is_set():
            item = self.display_slot.take(0.25)
            if item is None:
                continue
            started = time.perf_counter()
            try:
                self._encode_and_publish(item, on_message)
            except Exception:
                logger.exception("display encode failed")
                continue
            self.stats.encode_ms.append((time.perf_counter() - started) * 1000)
            self.stats.displayed += 1
        logger.info("encoder thread exiting")

    def _encode_and_publish(self, item: FrameItem, on_message: OnMessage) -> None:
        bgr = cv2.cvtColor(item.rgb, cv2.COLOR_RGB2BGR)

        # The console's video panel is much smaller than a 720/1080p frame, so
        # downscaling before encode costs nothing visually and saves both
        # encode time and WebSocket bandwidth (base64 inflates by 33%).
        max_w = self.settings.display_max_width
        if max_w and item.width > max_w:
            scale = max_w / item.width
            bgr = cv2.resize(bgr, (max_w, int(item.height * scale)), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY,
                                             self.settings.display_jpeg_quality])
        if not ok:
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        on_message({"type": "frame", "data": "data:image/jpeg;base64," + b64})

    # ---- detection thread -----------------------------------------------------

    def run_detection(self, on_message: OnMessage, stop: threading.Event) -> None:
        assert self.perception is not None, "call load() before run_detection()"
        # Optional ceiling only. The thread is self-pacing (it always takes the
        # newest frame and loops), so this exists purely to leave GPU headroom
        # if you want it — 0 disables.
        min_interval = 1.0 / self.settings.sample_fps if self.settings.sample_fps > 0 else 0.0
        last_captured: float | None = None

        try:
            while not stop.is_set():
                self._drain_gpu_inbox()

                item = self.detect_slot.take(0.25)
                if item is None:
                    continue

                started = time.perf_counter()
                try:
                    dt = (item.captured - last_captured) if last_captured is not None else None
                    self._process_frame(item, dt, on_message)
                    last_captured = item.captured
                except Exception:
                    logger.exception("detection pass failed")
                elapsed = time.perf_counter() - started
                self.stats.detect_ms.append(elapsed * 1000)
                self.stats.detected += 1

                # Post-work pacing, never a pre-work "is it time yet" gate —
                # the latter silently collapses into "run on every frame" as
                # soon as the work exceeds the interval.
                if min_interval:
                    stop.wait(max(0.0, min_interval - elapsed))
        finally:
            self._fail_pending_gpu_jobs()
            logger.info("detection thread exiting")

    def _process_frame(self, item: FrameItem, dt: float | None, on_message: OnMessage) -> None:
        frame = item.rgb
        enabled = self.store.enabled()
        if not enabled:
            # Still broadcast: otherwise disabling every concept leaves the
            # frontend rendering the last track set forever.
            on_message({"type": "tracks", "data": []})
            return

        concepts = [Concept(label=c.label, text_prompt=c.text_prompt) for c in enabled]
        detections = self.perception.detect_track(frame, concepts, want_masks=False)

        # DINOv3 only earns its keep once a concept actually has exemplars to
        # compare against — otherwise _filter_by_exemplars passes everything
        # through and the embeddings are computed for nothing.
        if detections and any(c.reference_embeddings for c in enabled):
            detections = self.perception.embed(frame, detections)
            detections = self._filter_by_exemplars(detections, enabled)

        raw = [(d.concept, d.box, d.score) for d in detections]
        tracks = self.tracker.update(raw, item.width, item.height, dt)
        self._broadcast_tracks(tracks, item.width, item.height, enabled, on_message)

        by_label = {c.label: c for c in enabled}
        for track in tracks:
            if not track.is_new:
                continue
            concept_record = by_label.get(track.concept)
            if concept_record is not None:
                self._queue_first_appearance(frame, track, concept_record.color, on_message)

    def _filter_by_exemplars(self, detections: list, enabled: list) -> list:
        """DINOv3 similarity gate — see config.match_threshold's docstring.
        Concepts with no authored exemplars pass every SAM 3 text-prompt
        candidate through unfiltered."""
        by_label = {c.label: c for c in enabled}
        kept = []
        for det in detections:
            record = by_label.get(det.concept)
            refs = record.reference_embeddings if record else None
            if not refs:
                kept.append(det)
                continue
            emb = np.asarray(det.embedding)
            if max(_cosine(emb, np.asarray(r)) for r in refs) >= self.settings.match_threshold:
                kept.append(det)
        return kept

    def _broadcast_tracks(self, tracks: list, frame_w: int, frame_h: int,
                          enabled: list, on_message: OnMessage) -> None:
        colors = {c.label: c.color for c in enabled}
        payload = []
        for t in tracks:
            x1, y1, x2, y2 = t.box
            payload.append(
                TrackedObject(
                    id=t.track_id,
                    concept=t.concept,
                    color=colors.get(t.concept, "#94a3b8"),
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

    # ---- Tier 2: VLM description, off the detection thread --------------------

    def _queue_first_appearance(self, frame: np.ndarray, track, color: str,
                                on_message: OnMessage) -> None:
        if self._vlm_pool is None:
            return

        with self._vlm_lock:
            if self._vlm_pending >= self.settings.vlm_max_pending:
                # Better a plain label now than a growing pile of 30s requests
                # against one vLLM instance.
                self.stats.vlm_skipped += 1
                self._emit_event(track, color, f"{track.concept.capitalize()} detected.", on_message)
                return
            self._vlm_pending += 1

        x1, y1, x2, y2 = track.box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        # .copy() is required, not hygiene: a slice is a *view* that pins the
        # whole ~2.7MB frame, and this crop can sit in the queue for up to
        # vlm_timeout_s.
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])

        # Capture identity/time/telemetry NOW. Reading them when the call
        # returns would stamp the event with the VLM's completion time, which
        # could be tens of seconds late.
        concept, track_id = track.concept, track.track_id
        telemetry = self.last_telemetry

        def _describe() -> None:
            try:
                description = vlm_client.describe_crop(
                    self.settings.vlm_base_url, self.settings.vlm_model, crop,
                    concept, self.settings.vlm_timeout_s,
                )
            except Exception:
                logger.exception("VLM description failed")
                description = f"{concept.capitalize()} detected."
            finally:
                with self._vlm_lock:
                    self._vlm_pending -= 1
            self._emit_event_raw(track_id, concept, color, description, telemetry, on_message)

        self._vlm_pool.submit(_describe)

    def _emit_event(self, track, color: str, description: str, on_message: OnMessage) -> None:
        self._emit_event_raw(track.track_id, track.concept, color, description,
                             self.last_telemetry, on_message)

    def _emit_event_raw(self, track_id: int, concept: str, color: str, description: str,
                        telemetry: Telemetry, on_message: OnMessage) -> None:
        event = EventItem(
            id=f"ev_{track_id}_{int(time.time() * 1000)}",
            ts=int(time.time() * 1000),
            concept=concept,
            concept_color=color,
            severity="info",
            kind="first appearance",
            description=description,
            verdict="pending",
            tier=2,
            deep_analyzed=False,
            track_id=track_id,
            telemetry=telemetry,
        )
        on_message({"type": "event", "data": event.model_dump(by_alias=True)})

    # ---- lifecycle ------------------------------------------------------------

    def start(self, on_message: OnMessage, stop: threading.Event) -> list[threading.Thread]:
        self._vlm_pool = ThreadPoolExecutor(
            max_workers=self.settings.vlm_max_concurrency, thread_name_prefix="vlm"
        )
        threads = [
            threading.Thread(target=self.run_ingest, args=(on_message, stop),
                             name="ingest", daemon=True),
            threading.Thread(target=self.run_encoder, args=(on_message, stop),
                             name="encode", daemon=True),
            threading.Thread(target=self.run_detection, args=(on_message, stop),
                             name="detect", daemon=True),
        ]
        for t in threads:
            t.start()
        return threads

    def shutdown(self) -> None:
        """Wake every blocked consumer so the threads can observe their stop event."""
        self.display_slot.close()
        self.detect_slot.close()
        if self._vlm_pool is not None:
            self._vlm_pool.shutdown(wait=False, cancel_futures=True)
