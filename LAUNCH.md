# Launching the UAV Stream Analysis System

Everything needed to bring the full stack up from cold, in order: model
servers → backend → frontend → a video stream to actually analyze. Every
command below is copy-pasteable and every option is explained — nothing is
assumed.

## Architecture, in one picture

```
uav_udp_stream/  --(UDP :5600, MPEG-TS: H.264 video + MISB KLV)-->  backend/
                                                                        |
                                                    imports in-process  |  calls over HTTP
                                                                        v                 v
                                                      model_servers/perception/   model_servers/ (vllm :8080)
                                                        (SAM 3 + DINOv3)              (Qwen3-VL)
                                                                        |
                                                                        v
                                                              frontend/ (:5173) via /ws + /api
```

The backend is the only thing that talks to the video stream and to
perception; it exposes everything else to the frontend over one WebSocket
(`/ws`) and a small REST API (`/api`). The VLM (`vllm`) is a separate
container the backend calls over HTTP — never run it and the `vlm` (Gemma 4
standby) container at the same time, they share port 8080.

---

## 0. Prerequisites (one-time)

- Docker Desktop with the WSL2 backend, GPU support enabled.
- NVIDIA Container Toolkit reachable from Docker. Verify:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  This should print your GPU (RTX 5000 Ada, 32GB in the `local` profile).
- Model weights already downloaded into `model_servers/models/` — see
  `model_servers/scripts/download_models.sh` (or `.bat` on native Windows).
  This guide assumes they're already there; if `model_servers/models/` is
  empty, run that script first (needs internet, and `hf auth login` for the
  gated SAM 3 checkpoint).
- `model_servers/.env` exists — the active profile. If it doesn't:
  ```bash
  cd model_servers
  cp profiles/local.env .env      # RTX 5000 Ada, 32GB — the default
  # cp profiles/full.env .env     # H100, 80GB — only if that's what you're on
  ```

---

## 1. Model servers (`model_servers/`)

All commands in this section run from `model_servers/`.

### 1a. Primary VLM — Qwen3-VL-8B-FP8 via vLLM

```bash
cd model_servers
docker compose --profile vllm up -d vllm
docker compose logs -f vllm       # watch it load, ~1-2 min
curl -sf localhost:8080/health
```

- `--profile vllm` — this service is gated behind a Compose profile so it
  doesn't start by accident; always required for this command.
- Why 8B, not the bigger 30B-A3B: the 30B-A3B checkpoint doesn't fit a 32GB
  card alongside SAM3+DINOv3 at any quantization down to official FP8
  (~31GB alone, zero headroom). The 30B-A3B BF16 checkpoint is still on disk
  (`models/qwen3-vl-30b-a3b-instruct/`) for whenever you move to the `full`
  H100 profile — see `profiles/full.env`.
- Health check must return before moving on — `curl` failing means the model
  is still loading (weight loading alone takes ~20-25s) or crashed; check
  `docker compose logs vllm` for the actual error before proceeding.

### 1b. Optional: Gemma 4 12B standby (llama.cpp)

Only if you actually want the alternative VLM — it is **not** needed for
normal operation and cannot run at the same time as `vllm` (port 8080):

```bash
docker compose stop vllm          # free port 8080 first
docker compose up -d vlm
```

Swap back with `docker compose stop vlm && docker compose --profile vllm up -d vllm`.

### 1c. Perception debug API (optional — NOT needed for the full pipeline)

`backend/` imports SAM 3 + DINOv3 **in-process** (see §2) — this container
exists only for poking at perception directly with curl, or for running
`perception/tools/*.py` (concept-pack authoring scripts). Skip this section
entirely for normal end-to-end operation.

```bash
docker compose --profile debug up -d perception-api
curl -sf localhost:8090/health     # {"status":"ok","backend":"sam3"}
```

---

## 2. Backend (repo root)

Run from the **repo root**, not `model_servers/` — the build context needs
to reach both `model_servers/perception/` and `uav_udp_stream/uavstream/`.

```bash
cd /path/to/stream_analyzer
docker compose up -d backend
docker compose logs -f backend    # loads SAM3+DINOv3, ~20-40s; watch for "perception ready"
curl -sf localhost:8000/health
```

What this container does on startup: loads SAM 3 + DINOv3 onto the GPU
in-process (no HTTP hop — see `model_servers/perception/README.md` §1.3),
seeds a default watch-list (`person`, `vehicle`, `tent`, `fire`), binds a UDP
listen socket on `:5600` for the video stream, and starts serving `/ws` and
`/api` on `:8000`.

**It joins `model_servers`'s Docker network** (`uav-model-servers_default`,
declared in the root `docker-compose.yml`) so it can reach `vllm` by service
name — this is why §1 must be running *first*; if `vllm` isn't up yet, the
container still starts fine, but Tier-2 VLM calls will fail until it is.

### Configuration (env vars, set in `docker-compose.yml`'s `backend.environment` block)

| Variable | Default (in compose) | What it controls |
|---|---|---|
| `STREAM_URL` | `udp://0.0.0.0:5600` | Bind address for the video+KLV listener. `0.0.0.0`, not `host.docker.internal` — this is a *listen* address, not a peer to connect to. |
| `PERCEPTION_BACKEND` | `sam3` | `sam3` or `sam2_gdino` (fallback, not implemented yet) |
| `SAM_WEIGHTS` | `/models/sam3` | Path inside the container (mounted from `model_servers/models/`) |
| `DINOV3_WEIGHTS` | `/models/dinov3-vitl16` | Same |
| `SAMPLE_FPS` | `5` | **Ceiling** on detection rate, not a pacing target. The detection thread is self-pacing (it always grabs the newest frame and loops), so this only exists to deliberately leave GPU headroom. Set high to let detection run flat out. |
| `DISPLAY_FPS` | `15` | How often a video frame is JPEG-encoded and pushed to the frontend. Runs on its own thread, fully independent of detection — video stays smooth even though SAM3 manages only a few frames a second. |
| `DISPLAY_JPEG_QUALITY` | `70` | JPEG quality for frames sent over `/ws` — trades bandwidth vs. clarity. |
| `DISPLAY_MAX_WIDTH` | `960` | Downscale before JPEG encoding. The console's video panel is much smaller than a 720/1080p frame, so this is nearly free bandwidth and encode time. `0` disables. |
| `VLM_MAX_CONCURRENCY` | `1` | Thread-pool size for Tier-2 description calls (they run off the detection thread). Kept at 1 because vLLM and SAM 3 share one GPU — see the contention note below. |
| `VLM_MAX_PENDING` | `3` | If this many VLM jobs are already queued, new events use the plain fallback label instead of piling up 30s requests. |
| `PERF_LOG_INTERVAL_S` | `10` | Rolling throughput/latency log (decode, convert, display, detect rates + p50/p95 stage times). `0` disables. **This is how you find out where time actually goes.** |
| `TRACK_MAX_MISSED_S` | `1.0` | How long a track coasts without a matching detection before being dropped. Wall-clock, since the detection interval is variable. |
| `VLM_BASE_URL` | `http://vllm:8080/v1` | Where Tier-2/3 calls go. Points at the `vllm` service by Docker network name. |
| `VLM_MODEL_NAME` | `uav-vlm` | Must match `--served-model-name` in `model_servers/docker-compose.yml`'s `vllm`/`vlm` service (both already use `uav-vlm`). |
| `VLM_TIMEOUT_S` | `30` | Per-call timeout before falling back to a plain `"{concept} detected."` label. |
| `TRACK_IOU_THRESHOLD` | `0.3` | Minimum box overlap to match a detection to an existing track between frames (`backend/app/tracker.py`). |
| `TRACK_MAX_MISSED_FRAMES` | `10` | How many consecutive frames a track can go undetected before being dropped. |
| `MATCH_THRESHOLD` | `0.55` | DINOv3 cosine-similarity gate for concepts with authored exemplars — see §5. |
| `CONTEXT_CROP_MARGIN` | `0.6` | How far the Tier-2/Tier-3 evidence crop expands past the tracked box, as a fraction of the box's own size — `0.6` ≈ 2.2x. Both the "what is this doing" VLM call and the Tier-3 chat evidence image use this crop. |
| `SCENE_INTERVAL_S` | `8.0` | How often (seconds) Tier-2 runs a whole-frame scene overview — see §5.1. One VLM call per interval tick, **not** one per detected object, so this is the flat, predictable Tier-2 cost knob regardless of how busy the scene is. Individual objects are queryable on demand instead (Objects tab → `GET /api/tracks/{id}/snapshot`), not automatically analyzed. |
| `EXEMPLAR_RECALL_BACKEND` | `both` | `sam3` \| `yoloe26` \| `both` — which recall source(s) run for concepts *with* authored exemplars (concepts without any always use SAM 3 text-prompt recall regardless). See "`EXEMPLAR_RECALL_BACKEND` cost" below for the measured overhead, and `app/config.py`'s docstring — `sam3` is the rollback switch if YOLOE-26 doesn't work out. |
| `YOLOE_WEIGHTS` | `/models/yoloe-26/yoloe-26s-seg.pt` | Path inside the container. Only loaded when `EXEMPLAR_RECALL_BACKEND` is `yoloe26` or `both`. |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` | Never phone home; weights are already local. |

To change any of these without editing `docker-compose.yml`, export them
before `docker compose up`, e.g. `SAMPLE_FPS=10 docker compose up -d backend`
— Compose interpolates `${VAR}` forms, but the current file hardcodes most
values directly, so editing the file is the more reliable way to change them
today.

---

## 3. Frontend

```bash
cd frontend
npm install        # first time only
npm run dev
```

Opens on **http://localhost:5173**. `vite.config.ts` proxies `/api` to
`localhost:8000` (the backend) — this works fine for REST calls. The
WebSocket (`/ws`) connects **directly** to `ws://localhost:8000/ws` in dev,
bypassing vite's proxy entirely — its WebSocket proxying couldn't sustain
the video frame throughput (~1.2MB/s) and dropped the connection repeatedly
(`socket hang up`); see `frontend/src/lib/wsFeed.ts`'s comment.

Other npm scripts: `npm run build` (typecheck + production bundle to
`dist/`), `npm run preview` (serve that bundle locally), `npm run typecheck`
(typecheck only, no build).

At this point, with §1 and §2 up, the frontend is live but idle — no video
source is connected yet, so `CenterFeed` shows its placeholder animation and
the watch-list is populated but nothing is being detected. That's expected;
proceed to §4.

---

## 4. Send it a video stream

Everything here runs from `uav_udp_stream/`. There are three starting
points depending on what footage you have.

### 4a. Synthetic test clip (no real footage needed)

```bash
cd uav_udp_stream
python make_video_stream.py --duration 30 --fps 30 --klv-rate 10
#   -> writes streams/uav_video.ts (default --out) with a rendered
#      crosshair+HUD video and synthetic orbiting-flight KLV telemetry
python stream_udp.py --file streams/uav_video.ts --target 127.0.0.1:5600 --loop
```

### 4b. A real video file, with synthetic KLV folded in

Use this when you have ordinary footage (no embedded telemetry) and want
realistic-looking flight metadata alongside it:

```bash
python make_video_stream.py --source "test_video/my_clip.mp4" --out "streams/my_clip_with_klv.ts"
python stream_udp.py --file "streams/my_clip_with_klv.ts" --target 127.0.0.1:5600 --loop
```

### 4c. A file that already has real embedded KLV (e.g. genuine UAV footage)

Check first — some sample UAV clips (e.g. the well-known "Day Flight.mpg")
already carry a real KLV data stream, no conversion needed:

```bash
ffprobe -hide_banner "test_video/your_file.mpg"
# look for a line like: Stream #0:1[...]: Data: klv (KLVA / 0x41564C4B)
# if present, skip make_video_stream.py entirely:
python stream_udp.py --file "test_video/your_file.mpg" --target 127.0.0.1:5600 --loop
```

### `make_video_stream.py` — full option reference

| Option | Meaning |
|---|---|
| `--out` | Output `.ts` path (default `streams/uav_video.ts`) |
| `--source` | Use a real video file instead of the rendered synthetic feed. It's looped and truncated to `--duration`, with the same crosshair/HUD overlay burned in on top — this only replaces the *video*, never the KLV (see below). |
| `--duration` | Clip length in seconds — applies to both modes (truncates the looped source, or sets the synthetic clip's length) |
| `--fps` | Frame rate for the synthetic generator; in `--source` mode it only sets the H.264 keyframe interval (`-g`), the source's own playback rate carries through unchanged |
| `--size` | Frame resolution — **synthetic mode only**; a `--source` file keeps its native resolution (the overlay scales to it automatically) |
| `--bitrate` | Video bitrate, e.g. `2M`/`4M`/`8M` — both modes |
| `--klv-rate` | KLV telemetry packets per second — both modes |
| `--no-hud` | Skip the burnt-in crosshair/HUD text overlay — both modes |
| `--lat`, `--lon`, `--radius`, `--altitude`, `--speed` | Synthetic orbit-flight parameters for the **KLV telemetry** (starting position, orbit radius, altitude in m, ground speed in m/s) — apply in both modes, since the telemetry is always fabricated by `uavstream.flight.OrbitPath` even when the video itself is real footage via `--source` |
| `--keep-video-only` | Also keep the intermediate video-only `.ts` (before KLV is folded in), for debugging — both modes |

### `stream_udp.py` — full option reference

| Option | Meaning |
|---|---|
| `--file` | The `.ts` to send (defaults to `config.json`'s `stream_file`, `streams/uav_video.ts`, if omitted) |
| `--target` | `host:port`, repeatable — send the same stream to multiple destinations at once. **Always pass this explicitly against the backend**: `config.json`'s baked-in default targets (`192.168.1.127:5600`, `127.0.0.1:5601`) are leftover from earlier NIC/pseudo-IP testing and don't point at the backend's listener — use `127.0.0.1:5600` to reach a backend running on the same machine, or the host's LAN IP if it's elsewhere. |
| `--port` | Shorthand for a single target's port |
| `--bitrate` | Only used as a pacing fallback if the file has no embedded PCR clock |
| `--packets` | TS packets per UDP datagram (default packs 7 into 1316 bytes to fit a standard MTU) — rarely needs changing |
| `--ttl` | Multicast TTL, if targeting a multicast address |
| `--iface` | Local IP to send multicast from |
| `--loop` | Repeat the file forever — use this for anything you want to keep watching in the UI rather than have stop after one pass |
| `--asap` | Ignore real-time pacing, blast the whole file as fast as possible — **don't use this against the backend**; it overran the UDP socket buffer and corrupted the stream (`Circular buffer overrun`) during testing. Real-time pacing (the default) is what actually works. |

### Watching/debugging the raw stream yourself (optional, separate from the backend)

```bash
python recv_udp.py --port 5601 --full          # print every decoded KLV tag
python recv_udp.py --port 5601 --json telemetry.jsonl   # log decoded telemetry to a file
```
Point this at a *different* port than the backend if running both at once —
UDP unicast delivery to one port goes to one listener; use `--target` on
`stream_udp.py` twice (once per port) to fan out to both.

---

## 5. Context authoring — teaching it what to look for

The default watch-list (`person`, `vehicle`, `tent`, `fire`) runs on SAM 3's
text prompt alone. Two ways to add or refine concepts:

**Live, from the running UI:** LeftRail's "Add context" / CenterFeed's "Draw
concept" button — creates a concept via `POST /api/concepts`, enabled
immediately, text-prompt-only (no exemplar images yet — real box-drawing in
the browser isn't built).

**From reference images, before or without opening the UI** — the same
DINOv3-embedding mechanism, run from the CLI (see
`model_servers/perception/README.md`'s "Context authoring" section):

```bash
cd model_servers
pip install opencv-python   # host-side GUI box-drawing tool, one-time — the
                             # container only has opencv-python-headless
python perception/tools/select_boxes.py --label tent --images tent1.jpg tent2.jpg
docker compose --profile debug up -d perception-api
docker compose exec perception-api python tools/embed_concept.py \
    --boxes concepts/tent.boxes.json --label tent --priority watch
```
This writes a concept pack to `model_servers/perception/concepts/` — not
yet picked up by the running `backend/` container automatically (it has its
own separate in-memory store, seeded fresh on startup); use the live
`POST /api/concepts/{id}/exemplars` endpoint instead if you want exemplars
in the actual running system.

---

## 6. Verify everything is actually working

```bash
curl -sf localhost:8080/health          # vLLM
curl -sf localhost:8000/health          # backend
curl -s  localhost:8000/api/concepts    # current watch-list
```
Then open **http://localhost:5173** — you should see real video playing,
telemetry updating in the top-left HUD, and (once something in frame matches a
watch-list concept) detection boxes with an event card appearing in the right
rail.

The status line shows **two** frame rates, and they are deliberately
different:

- **`N fps video`** — how fast the video itself updates. Expect 12-15.
- **`N fps detect`** — how often SAM 3 + DINOv3 actually run. Expect ~3-5.
  This is genuinely slower: SAM 3 manages only 5-6 FPS on an H200, a much
  stronger GPU than an RTX 5000 Ada. Boxes are interpolated between detections
  from each track's velocity so they still move smoothly, and fade slightly
  when they've been extrapolated past ~0.35s without a fresh detection.

For where the time is actually going, watch the backend's `perf:` log lines
(`PERF_LOG_INTERVAL_S`), which report decode/convert/display/detect rates and
p50/p95 stage latencies.

### Measured performance (RTX 5000 Ada, 720p source, `Day Flight.mpg`)

| Condition | Detection p50 | Detection rate | Video |
|---|---|---|---|
| 1 concept, vLLM idle | 169 ms | 5.4 /s | 13 /s |
| 4 concepts, vLLM **stopped** | 381-416 ms | 2.4 /s | 11-13 /s |
| 4 concepts, vLLM running + serving | ~3300 ms | 0.4 /s | 11-13 /s |

Three things worth internalising:

1. **Video is flat at 11-13 /s in every row.** That's the decoupling working —
   detection speed no longer affects playback at all. Verified directly by
   stopping the vLLM container mid-stream: video kept playing at full rate.
2. **GPU contention with vLLM is the single biggest factor — bigger than SAM 3
   itself.** Detection runs ~8x faster (3300 ms → ~400 ms) with vLLM stopped,
   because both models compete for the same card. There is no clean software
   fix for this short of a second GPU; `VLM_MAX_CONCURRENCY=1` limits the
   damage.
3. **Concept count is linear and cheap by comparison.** Each additional
   concept adds roughly 105 ms (one grounding-decoder pass) on top of a ~64 ms
   fixed image-encoder pass, which is shared across all concepts.

So the two levers that actually matter are **how much Tier-2 VLM traffic you
generate** and **how many concepts you leave enabled** — in that order.

### `EXEMPLAR_RECALL_BACKEND` cost (YOLOE-26 visual-prompt recall)

Measured with vLLM stopped (isolates the pure detection cost from VLM
contention above), one concept with one authored exemplar:

| `EXEMPLAR_RECALL_BACKEND` | Detection p50 | Detection p95 |
|---|---|---|
| `sam3` (YOLOE-26 not loaded) | 413 ms | 452 ms |
| `both` (SAM 3 + YOLOE-26 for that concept) | 1106-1133 ms | 1218-1665 ms |

YOLOE-26's exemplar-baked visual-prompt pass adds roughly **+700 ms** on top
of the SAM 3 + DINOv3 baseline for each concept that has exemplars — a
second full model forward pass, not free. It's only paid for concepts that
actually have authored exemplars (see `app/pipeline.py`'s `_process_frame`),
so a watch-list where most concepts are still text-prompt-only sees little
of this. `EXEMPLAR_RECALL_BACKEND=sam3` reverts to the pre-YOLOE cost
exactly (see `app/config.py`'s docstring — it's the rollback switch).

---

## 7. Shutting everything down

```bash
# stop whatever stream_udp.py / recv_udp.py you have running (Ctrl+C, or
# on Windows via Git Bash, taskkill with the real Windows PID — `pkill`
# isn't available, and MSYS's own `ps` PID numbering doesn't match
# Windows PIDs; use `ps -W` to find the real one)

cd /path/to/stream_analyzer
docker compose stop backend

cd model_servers
docker compose stop vllm perception-api vlm   # whichever are actually up
```
`stop` (not `down`) — containers and their config stay intact, so
`docker compose up -d ...` brings everything back without rebuilding.

---

## Troubleshooting — real issues hit while building this

- **vLLM crashes with `RuntimeError: UVA is not available`** — WSL2 disables
  pinned memory by default even on kernels that support it. Already fixed in
  `model_servers/docker-compose.yml` via `VLLM_WSL2_ENABLE_PIN_MEMORY: "1"`;
  if you see this again, check that env var is still set.
- **vLLM OOMs or the GPU shows near-100% memory used** — you're probably
  pointed at the 30B-A3B checkpoint instead of the 8B one. Check
  `VLLM_MODEL` in `model_servers/.env` matches `profiles/local.env`, and
  make sure `.env` was actually re-copied after any profile edit (`cp
  profiles/local.env .env` — editing the profile file alone does nothing
  until re-copied).
- **`av.error.OSError: ... bind failed`** on backend startup — another
  process (or a previous unclosed backend container) is already holding
  `:5600`. `docker compose stop backend` fully, confirm with `docker ps`,
  then start again.
- **Backend logs `Address already in use` on retry after a crash** — same
  root cause as above, at the Python level: if `ingest.py`'s stream reader
  is interrupted uncleanly the OS socket can linger. A full container
  restart clears it.
- **Browser console: `WebSocket ... failed: WebSocket is closed before the
  connection is established`** — harmless in dev. React 18 StrictMode
  double-invokes effects (mount → cleanup → mount), so the first connection
  attempt is intentionally torn down before it finishes; the second one is
  the real, working connection.
- **Frontend shows stale/zero data after many hot-reloads during
  development** — Vite HMR can leave a zustand store in a stale state when
  the store module itself is edited repeatedly. Fully restart `npm run dev`
  and hard-reload the browser (Ctrl+Shift+R) rather than trusting HMR.
- **`--asap` on `stream_udp.py`** — don't. It overruns the UDP receive
  buffer and corrupts the H.264 stream (`non-existing PPS referenced`, `no
  frame!` in backend logs). Use the default real-time pacing.
