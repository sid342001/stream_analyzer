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
    {"type": "status",    "data": {"connected": bool, "url": str, "error": str | None}}

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
from app.tracker import Tracker, _iou
from perception import Concept, Perception

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict[str, Any]], None]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# Public (no leading underscore) — shared with main.py's exemplar-upload
# endpoint, which needs the exact same crop/resize/encode pattern for
# exemplar thumbnails, just at a smaller max_side. Kept in this module since
# it's conceptually "how we turn a frame+box into something a VLM/model can
# be shown," which pipeline.py already owns for detection crops.
CONTEXT_CROP_MAX_SIDE = 640

# Downscale cap for the whole-frame scene analysis image (_queue_scene_analysis)
# — a full-res 720p/1080p frame costs more tokens/bandwidth for no benefit to
# a general "what's happening" description. Wider than CONTEXT_CROP_MAX_SIDE
# since it's the whole scene, not one object's crop — matches display_max_width's
# default, a resolution already judged "plenty for a human to read at a glance."
SCENE_IMAGE_MAX_WIDTH = 960


def context_crop(frame: np.ndarray, box: tuple[int, int, int, int], margin: float,
                 max_side: int = CONTEXT_CROP_MAX_SIDE) -> tuple[np.ndarray, list[int]]:
    """Slice a crop around `box` expanded by `margin` on each side (e.g.
    margin=0.6 -> the crop is ~2.2x the box's width/height), clamped to
    frame bounds, longer side capped at `max_side`. Returns the crop as an
    ndarray plus `box` re-expressed in the crop's own pixel coordinates, so
    nothing downstream needs offset math against the source frame.

    Deliberately wider than a tight box crop: a description/answer/exemplar
    grounded in the object *and its surroundings* (what it's near, what it's
    next to) is what these are for — a tight crop of the object alone
    starves the VLM of exactly that.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * margin, bh * margin
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    crop = np.ascontiguousarray(frame[cy1:cy2, cx1:cx2])

    # box, translated into the (still full-res) crop's own coordinates,
    # clamped since padding can be asymmetric near frame edges.
    box_in_crop = [
        max(0, x1 - cx1), max(0, y1 - cy1),
        min(cx2 - cx1, x2 - cx1), min(cy2 - cy1, y2 - cy1),
    ]

    crop_h, crop_w = crop.shape[:2]
    long_side = max(crop_w, crop_h)
    if long_side > max_side:
        # Trades object detail for context: a bigger margin packs more
        # surrounding scene into the same pixel budget, so the object itself
        # renders smaller. max_side is the knob if that tradeoff needs to
        # move the other way.
        scale = max_side / long_side
        crop = cv2.resize(crop, (max(1, int(crop_w * scale)), max(1, int(crop_h * scale))),
                          interpolation=cv2.INTER_AREA)
        box_in_crop = [int(v * scale) for v in box_in_crop]

    return crop, box_in_crop


def encode_rgb_jpeg_data_url(rgb: np.ndarray, quality: int = 85) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("failed to encode crop as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


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
        # One-shot, full-resolution frame capture for context authoring (draw
        # a box on a paused frame). Deliberately separate from display_slot:
        # that one is downscaled to display_max_width for bandwidth, but
        # exemplars are embedded against full ingest-resolution frames
        # (_process_frame uses item.rgb, never downscaled) — boxing the
        # smaller display frame would create exemplars from a measurably
        # blurrier source than what live detections are compared against.
        self.hires_slot = DemandSlot()
        self.stats = _Stats()

        # Out-of-band GPU work (currently: exemplar embedding from the REST
        # API). Routed through the detection thread so exactly one thread ever
        # issues CUDA work — see submit_gpu().
        self._gpu_inbox: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue(maxsize=8)
        self._vlm_pool: ThreadPoolExecutor | None = None
        self._vlm_pending = 0
        self._vlm_lock = threading.Lock()
        # Set by start() — see the comment there for why this needs to be an
        # instance attribute rather than staying a thread-local parameter.
        self._on_message: OnMessage | None = None
        # Set by request_reconnect(); polled by _ingest_once's frame loop and
        # _wait_backoff so a source swap takes effect promptly instead of
        # waiting out the current read or backoff.
        self._reconnect = threading.Event()

        # Tier-2 scene analysis: last time a whole-frame VLM call fired,
        # same clock as FrameItem.captured — interval-gated in
        # _process_frame, see _queue_scene_analysis.
        self._last_scene_analysis: float = 0.0

    def load(self) -> None:
        logger.info("loading SAM 3 + DINOv3 (backend=%s, exemplar_recall=%s)...",
                    self.settings.perception_backend, self.settings.exemplar_recall_backend)
        self.perception = Perception(
            backend=self.settings.perception_backend,
            sam_weights=self.settings.sam_weights,
            dinov3_weights=self.settings.dinov3_weights,
            exemplar_recall_backend=self.settings.exemplar_recall_backend,
            yoloe_weights=self.settings.yoloe_weights,
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

    def submit_vlm(self, fn: Callable[[], Any]) -> Future:
        """Queue non-CUDA work (currently: Tier-3 chat) onto the VLM thread
        pool. Shares admission control (vlm_max_concurrency) with Tier-2
        descriptions rather than stacking uncontrolled load on top of a vLLM
        instance already measured as this system's GPU-contention bottleneck.
        """
        assert self._vlm_pool is not None, "call start() before submit_vlm()"
        return self._vlm_pool.submit(fn)

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
            except Exception as exc:
                logger.exception("ingest loop failed, retrying in %.0fs", backoff)
                on_message({"type": "status", "data": {
                    "connected": False, "url": self.settings.stream_url, "error": str(exc),
                }})
                self._wait_backoff(backoff, stop)
                backoff = min(backoff * 2, 15.0)
        logger.info("ingest thread exiting")

    def _wait_backoff(self, seconds: float, stop: threading.Event) -> None:
        """Interruptible reconnect backoff — wakes early on shutdown *or* a
        reconnect request, so switching away from an already-dead source
        doesn't have to wait out the old exponential backoff (up to 15s).
        Polls in short slices since threading.Event can't wait on two events
        at once; 0.2s is well under the granularity anyone would notice.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if stop.wait(0.2) or self._reconnect.is_set():
                return

    def request_reconnect(self, url: str) -> None:
        """Swap the live source without restarting the process — called from
        the REST handler on the event loop, which has no thread of its own
        to receive on_message as a parameter, hence self._on_message (set in
        start()). tracker.reset() already runs unconditionally at the top of
        every _ingest_once entry, so stale tracks from the old source are
        cleared for free once the swap takes effect.
        """
        self.settings.stream_url = url
        self._reconnect.set()
        if self._on_message is not None:
            self._on_message({"type": "status",
                              "data": {"connected": False, "url": url, "error": None}})

    def _ingest_once(self, on_message: OnMessage, stop: threading.Event) -> None:
        # A fresh stream is a fresh scene: stale tracks would IOU-match
        # unrelated content and suppress the first-appearance events that
        # genuinely new objects should fire.
        self.tracker.reset()

        display_interval = 1.0 / max(self.settings.display_fps, 0.1)
        last_display = 0.0
        seq = 0
        announced = False

        gen = ingest.frames(self.settings.stream_url)
        try:
            for video_frame, telemetry in gen:
                if stop.is_set():
                    return
                if self._reconnect.is_set():
                    self._reconnect.clear()
                    return

                if not announced:
                    # Only reachable once the generator has actually yielded
                    # something, i.e. av.open() succeeded — a failed open
                    # raises out of the `for` statement's implicit next()
                    # before this point, straight into run_ingest's except.
                    announced = True
                    on_message({"type": "status", "data": {
                        "connected": True, "url": self.settings.stream_url, "error": None,
                    }})

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
                want_hires = self.hires_slot.wanted()
                if not (want_display or want_detect or want_hires):
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
                if want_hires:
                    self.hires_slot.offer(item)

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

        # Recall routing (see config.exemplar_recall_backend's docstring):
        # concepts with no exemplars have nothing for YOLOE to bake a visual
        # prompt from, so they always go through SAM 3's text prompt
        # regardless of the setting. Concepts *with* exemplars are routed by
        # the setting — "sam3" keeps this identical to pre-Phase-3 behavior.
        backend = self.settings.exemplar_recall_backend
        sam3_concepts = [c for c in enabled if not c.exemplars or backend in ("sam3", "both")]
        yoloe_concepts = [c for c in enabled if c.exemplars and backend in ("yoloe26", "both")]

        detections: list = []
        if sam3_concepts:
            concepts = [Concept(label=c.label, text_prompt=c.text_prompt) for c in sam3_concepts]
            detections.extend(self.perception.detect_track(frame, concepts, want_masks=False))
        for c in yoloe_concepts:
            detections.extend(self.perception.detect_by_exemplar(frame, c.label, c.exemplars))

        if yoloe_concepts:
            # Only when both sources could have run for some concept —
            # otherwise this is a no-op pass that risks changing SAM3-only
            # behavior in a way "sam3" is supposed to never do (e.g. SAM 3
            # occasionally proposing two overlapping boxes on its own).
            detections = self._dedupe_by_iou(detections)

        # DINOv3 only earns its keep once a concept actually has exemplars to
        # compare against — otherwise _filter_by_exemplars passes everything
        # through and the embeddings are computed for nothing.
        if detections and any(c.exemplars for c in enabled):
            detections = self.perception.embed(frame, detections)
            detections = self._filter_by_exemplars(detections, enabled)

        raw = [(d.concept, d.box, d.score) for d in detections]
        tracks = self.tracker.update(raw, item.width, item.height, dt)
        self._broadcast_tracks(tracks, item.width, item.height, enabled, on_message)

        # Tier-2 is now a periodic whole-frame narrative, not one call per
        # newly-appeared object — see config.scene_interval_s's docstring.
        # Individual objects stay fully queryable on demand (see main.py's
        # GET /api/tracks/{id}/snapshot); they just don't each trigger an
        # automatic VLM call anymore, so cost is a flat rate regardless of
        # how many objects are on screen.
        if enabled and item.captured - self._last_scene_analysis >= self.settings.scene_interval_s:
            self._last_scene_analysis = item.captured
            self._queue_scene_analysis(item, on_message)

    @staticmethod
    def _dedupe_by_iou(detections: list, threshold: float = 0.6) -> list:
        """Collapse near-duplicate boxes for the same concept — e.g. the
        same physical object found by both SAM 3's text-prompt pass and
        YOLOE-26's exemplar pass — keeping the higher-scoring one of each
        overlapping pair. O(n^2) but n is a handful of detections per frame."""
        kept: list = []
        for det in sorted(detections, key=lambda d: d.score, reverse=True):
            if any(det.concept == k.concept and _iou(det.box, k.box) >= threshold for k in kept):
                continue
            kept.append(det)
        return kept

    def _filter_by_exemplars(self, detections: list, enabled: list) -> list:
        """DINOv3 similarity gate — see config.match_threshold's docstring.
        Concepts with no authored exemplars pass every SAM 3 text-prompt
        candidate through unfiltered."""
        by_label = {c.label: c for c in enabled}
        kept = []
        for det in detections:
            record = by_label.get(det.concept)
            refs = [e.embedding for e in record.exemplars] if record else []
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

    # Neutral color for scene notes — not tied to a concept, so no concept
    # color applies. Same fallback gray _broadcast_tracks already uses for
    # an unmatched concept.
    _SCENE_COLOR = "#94a3b8"

    def _queue_scene_analysis(self, item: FrameItem, on_message: OnMessage) -> None:
        """One whole-frame VLM call, gated to run at most once every
        settings.scene_interval_s (see _process_frame) — replaces the old
        per-object first-appearance analysis. Cost is now a flat rate
        regardless of how many objects are on screen; individual objects
        stay queryable via main.py's GET /api/tracks/{id}/snapshot instead
        of each triggering an automatic call.
        """
        if self._vlm_pool is None:
            return

        frame = item.rgb
        if frame.shape[1] > SCENE_IMAGE_MAX_WIDTH:
            scale = SCENE_IMAGE_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (SCENE_IMAGE_MAX_WIDTH, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)

        try:
            image = encode_rgb_jpeg_data_url(frame)
        except Exception:
            logger.exception("failed to encode scene image")
            image = None

        telemetry = self.last_telemetry

        with self._vlm_lock:
            if self._vlm_pending >= self.settings.vlm_max_pending:
                # Skip this tick rather than posting a hollow fallback note —
                # unlike a missed object detection, there's no "detected but
                # couldn't describe" case here worth logging as an event; the
                # next interval tries again shortly on its own.
                self.stats.vlm_skipped += 1
                return
            self._vlm_pending += 1

        def _describe() -> None:
            try:
                description = vlm_client.describe_scene(
                    self.settings.vlm_base_url, self.settings.vlm_model, frame,
                    self.settings.vlm_timeout_s,
                )
            except Exception:
                logger.exception("VLM scene description failed")
                description = "No notable activity."
            finally:
                with self._vlm_lock:
                    self._vlm_pending -= 1
            self._emit_event_raw(None, "scene", self._SCENE_COLOR, description, telemetry,
                                 on_message, image, [])

        self._vlm_pool.submit(_describe)

    def _emit_event_raw(self, track_id: int | None, concept: str, color: str, description: str,
                        telemetry: Telemetry, on_message: OnMessage,
                        image: str | None = None, box: list[int] | None = None) -> None:
        event = EventItem(
            id=f"ev_{track_id if track_id is not None else 'scene'}_{int(time.time() * 1000)}",
            ts=int(time.time() * 1000),
            concept=concept,
            concept_color=color,
            severity="info",
            kind="scene overview",
            image=image,
            box=box or [],
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
        # Retained so out-of-band callers (REST handlers on the event loop,
        # e.g. request_reconnect()) can publish too — everything else gets
        # on_message as a parameter, but those callers have no thread of
        # their own to thread it through.
        self._on_message = on_message
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
        self.hires_slot.close()
        if self._vlm_pool is not None:
            self._vlm_pool.shutdown(wait=False, cancel_futures=True)
