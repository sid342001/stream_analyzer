"""Backend configuration — all overridable via environment variables.

There is no top-level .env file wiring this yet (see ../README.md); for now
these defaults assume: perception weights mounted the same way perception-api
mounts them, and vLLM reachable at localhost:8080 (adjust VLM_BASE_URL if the
backend itself runs inside a container — e.g. `http://host.docker.internal:8080`
on Docker Desktop, or the vllm service name if joined to its compose network).
"""

from __future__ import annotations

import os


class Settings:
    # ---- ingest: UDP MPEG-TS + KLV source (see uav_udp_stream/) --------------
    # Bind address, not a peer to connect to — 0.0.0.0 so it accepts packets
    # from anywhere (the host, another container, etc.), not just localhost.
    # host.docker.internal is wrong here: that resolves to the host's address
    # for a container to *connect out* to, and can't be bound as a local
    # listen address from inside the container.
    stream_url: str = os.environ.get("STREAM_URL", "udp://0.0.0.0:5600")

    # ---- perception: SAM 3 + DINOv3, imported in-process (see model_servers/) -
    perception_backend: str = os.environ.get("PERCEPTION_BACKEND", "sam3")
    sam_weights: str = os.environ.get("SAM_WEIGHTS", "/models/sam3")
    dinov3_weights: str = os.environ.get("DINOV3_WEIGHTS", "/models/dinov3-vitl16")
    # Upper bound on how often detection runs, NOT a pacing mechanism — the
    # detection thread is self-pacing (it always takes the newest frame and
    # loops), so this only exists to deliberately leave GPU headroom. Set it
    # high to let detection run flat out.
    sample_fps: float = float(os.environ.get("SAMPLE_FPS", "5"))

    # ---- display: video frames sent to the frontend for CenterFeed's canvas ---
    # Separate from sample_fps (the detection rate) on purpose, and served by a
    # separate thread — the video plays smoothly even though SAM3/DINOv3 only
    # manage a few frames a second.
    display_fps: float = float(os.environ.get("DISPLAY_FPS", "15"))
    display_jpeg_quality: int = int(os.environ.get("DISPLAY_JPEG_QUALITY", "70"))
    # Downscale before JPEG encoding. The console's video panel is far smaller
    # than a 1280px-wide frame, so shrinking here costs nothing visually and
    # saves both encode time and WebSocket bandwidth. 0 disables downscaling.
    display_max_width: int = int(os.environ.get("DISPLAY_MAX_WIDTH", "960"))

    # ---- VLM: OpenAI-compatible endpoint (vllm or the llama.cpp standby) -----
    vlm_base_url: str = os.environ.get("VLM_BASE_URL", "http://localhost:8080/v1")
    vlm_model: str = os.environ.get("VLM_MODEL_NAME", "uav-vlm")
    vlm_timeout_s: float = float(os.environ.get("VLM_TIMEOUT_S", "30"))
    # Tier-2 description calls run on their own small thread pool so a slow or
    # dead VLM can never stall detection (let alone video). If more than
    # `vlm_max_pending` jobs are already queued we skip the call and use the
    # fallback label rather than piling 30s requests onto one vLLM instance.
    #
    # Concurrency of 1 is deliberate and measured: vLLM and SAM 3 share one
    # GPU, and concurrent VLM inference is what dominates detection tail
    # latency. Observed on an RTX 5000 Ada with 4 concepts — detection p50
    # 485ms / p95 ~3500ms while VLM calls were in flight, versus p50 169ms /
    # p95 191ms with the VLM idle. Raising this trades detection smoothness
    # for faster event descriptions.
    vlm_max_concurrency: int = int(os.environ.get("VLM_MAX_CONCURRENCY", "1"))
    vlm_max_pending: int = int(os.environ.get("VLM_MAX_PENDING", "3"))
    # How far the on-demand object snapshot (GET /api/tracks/{id}/snapshot)
    # and Tier-3 chat evidence crop expand past the tracked box, as a
    # fraction of the box's own width/height — 0.6 means the crop is ~2.2x
    # the box's size (see pipeline.py's context_crop). Raising it gives the
    # model more of the surrounding scene at the cost of the object itself
    # rendering smaller within the fixed CONTEXT_CROP_MAX_SIDE pixel budget.
    context_crop_margin: float = float(os.environ.get("CONTEXT_CROP_MARGIN", "0.6"))

    # ---- Tier-2: periodic whole-frame scene analysis ---------------------------
    # Replaces per-object first-appearance analysis: one VLM call per this
    # many seconds, describing the whole frame, instead of one call per
    # newly-appeared object. This number *is* the Tier-2 cost/freshness
    # knob now — flat and predictable regardless of how many objects are on
    # screen (see pipeline.py's _queue_scene_analysis). Individual objects
    # stay queryable on demand via GET /api/tracks/{id}/snapshot.
    scene_interval_s: float = float(os.environ.get("SCENE_INTERVAL_S", "8.0"))

    # ---- Perception backend routing — see model_servers/perception/
    # interface.py's detect_track/detect_by_text/detect_by_exemplar and
    # pipeline.py's _process_frame.
    #   "sam3"    -> SAM 3 text-prompt recall only, for every concept
    #                (pre-Phase-3 behavior; YOLOE-26 is never loaded or
    #                called). Rollback switch: if YOLOE-26 integration
    #                doesn't verify cleanly, set this and behavior matches
    #                exactly what existed before YOLOE-26 was added.
    #   "yoloe26" -> YOLOE-26 only; SAM 3 is not loaded at all (saves GPU
    #                memory). YOLOE-26 is open-vocabulary, so it covers
    #                both cases itself: plain text-prompt concepts go
    #                through its text path (detect_by_text, same
    #                set_classes()/get_text_pe() mechanism Ultralytics uses
    #                for YOLOE's own open-vocab text mode — confirmed via
    #                inspect.getsource, see interface.py's module
    #                docstring), concepts with exemplars go through its
    #                visual-prompt path (detect_by_exemplar) as before.
    #   "both"    -> default. SAM 3 text-prompt recall for every concept,
    #                plus YOLOE-26 visual-prompt recall as a second source
    #                for concepts that have exemplars (deduped by IOU).
    exemplar_recall_backend: str = os.environ.get("EXEMPLAR_RECALL_BACKEND", "both")
    yoloe_weights: str = os.environ.get("YOLOE_WEIGHTS", "/models/yoloe-26/yoloe-26s-seg.pt")

    # ---- tracker (backend/app/tracker.py) -------------------------------------
    track_iou_threshold: float = float(os.environ.get("TRACK_IOU_THRESHOLD", "0.3"))
    # Wall-clock, not a frame count: the detection interval is variable now
    # (the detection thread is self-pacing), so "N missed frames" would mean a
    # different amount of real time at every rate.
    track_max_missed_s: float = float(os.environ.get("TRACK_MAX_MISSED_S", "1.0"))

    # ---- exemplar matching: DINOv3 cosine similarity gate ----------------------
    # SAM 3's text prompt casts a wide net; a concept with authored exemplars
    # (see app/store.py) is only confirmed if a detection's DINOv3 embedding is
    # at least this similar to one of them — same mechanism as
    # model_servers/perception/tools/test_inference.py. Concepts with zero
    # exemplars skip this gate (nothing to compare against yet).
    match_threshold: float = float(os.environ.get("MATCH_THRESHOLD", "0.55"))

    # ---- observability ------------------------------------------------------
    # Rolling per-stage timing + throughput summary, logged this often.
    # 0 disables. Counters are plain ints incremented without a lock, so they
    # are approximate — fine for logging, don't build logic on them.
    perf_log_interval_s: float = float(os.environ.get("PERF_LOG_INTERVAL_S", "10"))

    # ---- server -----------------------------------------------------------------
    host: str = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port: int = int(os.environ.get("BACKEND_PORT", "8000"))


settings = Settings()
