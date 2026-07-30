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
    sample_fps: float = float(os.environ.get("SAMPLE_FPS", "6"))

    # ---- display: video frames sent to the frontend for CenterFeed's canvas ---
    # Separate from sample_fps (the detection rate) on purpose — the video
    # should play smoothly even though SAM3/DINOv3 only run a few times a
    # second; JPEG quality is capped for WebSocket bandwidth, not the video's
    # actual quality.
    display_fps: float = float(os.environ.get("DISPLAY_FPS", "12"))
    display_jpeg_quality: int = int(os.environ.get("DISPLAY_JPEG_QUALITY", "70"))

    # ---- VLM: OpenAI-compatible endpoint (vllm or the llama.cpp standby) -----
    vlm_base_url: str = os.environ.get("VLM_BASE_URL", "http://localhost:8080/v1")
    vlm_model: str = os.environ.get("VLM_MODEL_NAME", "uav-vlm")
    vlm_timeout_s: float = float(os.environ.get("VLM_TIMEOUT_S", "30"))

    # ---- tracker (backend/app/tracker.py) -------------------------------------
    track_iou_threshold: float = float(os.environ.get("TRACK_IOU_THRESHOLD", "0.3"))
    track_max_missed_frames: int = int(os.environ.get("TRACK_MAX_MISSED_FRAMES", "10"))

    # ---- exemplar matching: DINOv3 cosine similarity gate ----------------------
    # SAM 3's text prompt casts a wide net; a concept with authored exemplars
    # (see app/store.py) is only confirmed if a detection's DINOv3 embedding is
    # at least this similar to one of them — same mechanism as
    # model_servers/perception/tools/test_inference.py. Concepts with zero
    # exemplars skip this gate (nothing to compare against yet).
    match_threshold: float = float(os.environ.get("MATCH_THRESHOLD", "0.55"))

    # ---- server -----------------------------------------------------------------
    host: str = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port: int = int(os.environ.get("BACKEND_PORT", "8000"))


settings = Settings()
