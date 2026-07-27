# UAV Stream Analysis System — Local / Offline Deployment Plan

*Companion to `Architecture_and_Technology.md` and `DEMO.md`. This document pins
those designs to the actual hardware in hand, commits to concrete libraries with
justification, and lays out a buildable, fully-offline, **all-Docker**, single-box
plan.*

---

## 1. Context — why this document exists

The system is an **AI analyst for UAV video streams**: it watches the feed, learns
what matters from a few example boxes (no training), and returns a short, ranked
list of grounded moments the operator can trust and interrogate. The feed is
general EO/IR UAV video — **not thermal-specific**. That matters technically:
daylight EO video is *in-distribution* for the RGB-trained models, so text prompts
are reliable and example boxes are an accelerant rather than a crutch. Degraded,
night, or IR sub-feeds are the edge case the exemplar/embedding path covers, not
the central assumption.

Two deployment constraints drive the rest of this plan:

1. **Everything runs locally, offline, in Docker.** No native installs, no WSL
   venvs — all services are containers orchestrated by Docker Compose with GPU
   passthrough. "Offline" means *after a one-time download* of weights, images,
   packages, and map tiles; the running system is air-gappable.
2. **The GPU is 32 GB, not the doc's 80 GB H100.** The doc's own budget —
   SAM 3 (~5 GB) + DINOv2 (~3 GB) + **Qwen3-VL-30B FP8 (~32 GB)** ≈ 40 GB — does
   not fit: the 30B VLM alone fills the card. Per the "keep both" decision, the
   model layer carries **two configurations** side by side (§2).

### 1.1 Verified host hardware

| Resource | Value | Implication |
|---|---|---|
| GPU | RTX 5000 Ada, **32 GB**, compute **8.9** | FP8 + AWQ/GPTQ supported; fits the *local* profile, not the 30B |
| CPU | Xeon w7-3465X | ample for decode, overlay, orchestration |
| RAM | **256 GB** | room for CPU KV-offload, large frame buffers, Postgres cache |
| Virtualization | WSL2 + **Docker Desktop running** (GPU passthrough) | every service is a container |
| Runtime on host | Node 22, Python 3.13, ffmpeg 8.1 | present for dev; production paths are all containerized |
| Disk (D:) | ~323 GB free | enough for both model profiles + clips; weights are gitignored |

---

## 2. Two model configurations (kept in parallel)

The model layer is **config-driven** via a single `profile` setting. Pipeline
code, APIs, and UI are identical across profiles; only the served model IDs,
quantization, and VRAM caps change. The local box runs `local`; an H100 deployment
runs `full`, from the same repository and the same containers.

| Component | `local` profile (RTX 5000 Ada, 32 GB) | `full` profile (H100, 80 GB — the doc) |
|---|---|---|
| VLM | **Qwen2.5-VL-7B-Instruct** (AWQ 4-bit, ~6 GB) or Qwen3-VL-8B (FP8, ~9 GB) | **Qwen3-VL-30B-A3B** (FP8, ~32 GB) |
| Perception | SAM 3, or SAM 2.1 + GroundingDINO fallback (~5 GB) | SAM 3 (~5 GB) |
| Verifier | DINOv2 ViT-L/14 (~1.5 GB) | DINOv2 ViT-L/14 (~3 GB) |
| Approx. VRAM | **~14–18 GB** (headroom for image KV cache) | ~40 GB + buffers |
| Serving | vLLM, one GPU, all resident | vLLM, one GPU, all resident |

Profiles live in `model_servers/profiles/{local,full}.yaml` and are selected by an
env var (`MODEL_PROFILE=local`). The orchestrator only ever talks to the servers
over HTTP, so swapping a profile is a `docker compose` restart with zero app changes.

### 2.1 Perception: SAM 3 primary, SAM 2.1 fallback (both wired)

SAM 3's Promptable Concept Segmentation is the doc's design. Because SAM 3 weight
availability offline is not yet confirmed, the perception container exposes **one
internal interface** (`detect_track(frame, concepts, exemplars) -> detections`)
with **two backends**:

- **`sam3`** — native concept prompts + tracking (preferred).
- **`sam2_gdino`** — SAM 2.1 for segment+track, **GroundingDINO** (or YOLO-World)
  for open-vocab detection from text, plus **DINOv2** exemplar matching for the
  box-drawn concepts. Fully available offline today; reproduces the doc behavior
  with two models behind the same API.

The backend is a config switch; the rest of the system never knows which is live.

---

## 3. Technology stack, with justification

Every layer below ships as a **Docker container**; a top-level `docker-compose.yml`
brings up the whole system (models + DB + backend + frontend) with one command.

### 3.1 Model servers (`model_servers/`) — dedicated hosting folder

Its own Compose file so the heavy GPU services can be brought up/down independently
of the app. GPU passthrough via the NVIDIA Container Toolkit. Rationale for
containerizing: reproducible CUDA/toolkit versions, isolation from the host, and it
air-gaps as `docker save` images.

| Library / image | Role | Why this one |
|---|---|---|
| **vLLM** (OpenAI-compatible server) | serve the VLM | FP8 + AWQ on Ada, paged-attention KV cache, native tool-calling, OpenAI API so the orchestrator calls it like any tool. SGLang is a drop-in alternative. |
| **transformers / accelerate / torch** | load SAM, DINOv2, GroundingDINO | standard, offline-cacheable weights; `HF_HUB_OFFLINE=1` at runtime |
| **SAM 3** *(or)* **sam2 + GroundingDINO / YOLO-World (ultralytics)** | perception backend | matches §2.1; open-vocab detect + segment + track |
| **DINOv2** | exemplar appearance embeddings | verifier + fallback concept matcher; robust on degraded/IR frames |
| **FastAPI + uvicorn** (SAM/DINO wrappers) | HTTP endpoints for the non-LLM models | no standard server ships for these; a thin async wrapper is the norm |
| **pgvector/pgvector:pg16** (Postgres image) | relational + vector store | pgvector keeps embeddings beside relational rows; one query for similarity (doc §5) |

### 3.2 Orchestrator / backend (`backend/`) — container

**FastAPI** (async, same language as the ML ecosystem) hosting the 3-tier
scheduler, the geometric event engine, DB access, and a **WebSocket** to the UI.

| Library | Role | Why |
|---|---|---|
| **fastapi + uvicorn + pydantic** | API, schemas, WS | async, typed, first-class WebSocket |
| **PyAV** (`av`) | decode the UDP MPEG-TS to frames | binds libav directly; no shell-out; reads the same TS `uav_udp_stream` emits |
| **`uavstream`** (existing package) | pull KLV lat/long/heading off the TS | **reuses the code already built** — the sensor layer feeds the analyzer |
| **opencv-python + supervision** | draw masks/boxes/trails, encode JPEG | doc's "marking is a free overlay"; supervision gives clean annotators |
| **httpx** | call model servers | async client to vLLM/SAM/DINO |
| **SQLAlchemy + psycopg** (or **sqlite + sqlite-vec**) | persistence | Postgres+pgvector for the full store; SQLite path for the lightest demo (doc §5) |
| **numpy** | geometry, motion (speed/heading), event rules | Tier-1 motion analytics without the VLM |

The event engine is **geometric and cheap** (doc §4): first-appearance,
low-confidence adjudication, subject-approaching-cluster — all from SAM track IDs
+ KLV motion, no VLM in the hot loop.

### 3.3 Frontend (`frontend/`) — professional console, container

**React + Vite + TypeScript + Tailwind + shadcn/ui**, built to static assets and
served by an **nginx** container. shadcn is chosen over MUI for a dense, bespoke,
dark ISR console: components are **vendored into the repo** (perfect for offline),
fully restyleable, and Radix-backed for accessibility.

| Library | Role | Why |
|---|---|---|
| **react + vite + typescript** | app shell, fast HMR | standard, offline dev server; static build for prod |
| **tailwindcss + shadcn/ui + lucide-react** | the three-zone console, cards, panels, icons | professional look, no runtime CDN, themeable dark UI |
| **zustand** + **@tanstack/react-query** | client state + server cache | light, predictable; RQ manages event-feed/polling |
| Live feed via **`<canvas>` + WebSocket** (server-drawn JPEG frames) | glanceable center video with overlays | simplest reliable low-latency path fully offline; **aiortc/WebRTC** is the later upgrade |
| **maplibre-gl + pmtiles** (vendored `.pmtiles`) | KLV lat/long track panel | MapLibre is offline-capable; **PMTiles is a single local file — no tile server, no internet** — and ties the telemetry work into the console |

The layout follows `DEMO.md` §3 exactly: left watch-list (draw-to-teach), small
center live view, hero event feed on the right, timeline strip at the bottom.

### 3.4 Offline strategy (Docker-centric)

One-time, online, run once via `scripts/`:
1. `huggingface-cli download` all profile weights into `data/models/` (both
   profiles), mounted read-only into the model containers; `HF_HUB_OFFLINE=1` at runtime.
2. Build every app image and `docker compose pull` the base/DB images, then
   `docker save` the full set into an air-gapped bundle.
3. `npm ci` during the frontend image build (multi-stage); no runtime npm.
4. Fetch a region **`.pmtiles`** and fonts/glyphs into `frontend/public/`.

After that, unplugging the network changes nothing. `scripts/verify_offline.sh`
brings the whole stack up with `--network none`-style isolation as an acceptance test.

---

## 4. Repository layout (monorepo)

The project root becomes `stream_analyzer/`. The existing streaming project stays
intact as the ingest/sensor layer; a **dedicated `model_servers/`** folder hosts
all model serving. Every deployable is a container.

```
stream_analyzer/
├─ docs/                      # architecture, demo, this plan
├─ uav_udp_stream/            # EXISTING: TS+KLV sensor / simulator (ingest source)
├─ model_servers/            # ← DEDICATED model hosting (GPU containers)
│  ├─ docker-compose.yml      #   vllm + sam + dino + postgres, GPU passthrough
│  ├─ profiles/               #   local.yaml, full.yaml  (the "both configs")
│  ├─ vllm/                   #   VLM service config / entrypoint
│  ├─ sam/                    #   Dockerfile + FastAPI wrapper: sam3 | sam2_gdino
│  ├─ dino/                   #   Dockerfile + FastAPI wrapper: DINOv2 embeddings
│  └─ README.md               #   bring-up, health checks, VRAM notes
├─ backend/                   # Dockerfile + FastAPI orchestrator (tiers, events, DB, WS)
├─ frontend/                  # Dockerfile (build → nginx) React + shadcn console
├─ data/                      # models/, clips/, frames/, thumbs/  (gitignored, mounted)
├─ scripts/                   # download_models, offline_bundle, verify_offline
└─ docker-compose.yml         # top-level: models + db + backend + frontend, one command
```

---

## 5. Data flow (single box, all containers)

```
UAV clip ──▶ uav_udp_stream (UDP TS + KLV) ──▶ backend ingest (PyAV decode + uavstream KLV)
                                                    │
                 Tier 1 ── SAM container: detect/segment/track (sampled frames) ─┐
                                                    │                            │
         overlays (OpenCV/supervision) ◀───────────┘                            │
                 │                                                             events
          WebSocket (JPEG frames + detections) ──▶ frontend canvas              │
                                                    │                  Tier 2 ── vLLM one-liner
                 watch-list draw-box ──▶ context_exemplars ──▶ SAM/DINO         │
                                                    │                  Tier 3 ── vLLM Q&A (on click)
                      Postgres+pgvector  ◀── frames/detections/tracks/events/vlm_outputs
                                                    │
                        map panel ◀── KLV lat/long track ; timeline export ◀── events
```

Reuses the DB schema from `Architecture_and_Technology.md` §5 verbatim.

---

## 6. Phased build

| Phase | Deliverable | Proves |
|---|---|---|
| 0 | Monorepo scaffold; Compose files; `scripts/download_models`; offline bundle; profiles wired | air-gap readiness |
| 1 | `model_servers/` up: vLLM + SAM + DINO + Postgres containers, GPU passthrough, health checks | models serve locally on 32 GB |
| 2 | Backend container: consume `uav_udp_stream` UDP, decode frames+KLV, Tier-1 SAM overlays over WS | live glanceable feed with tracking |
| 3 | Frontend container: three-zone console — canvas feed, watch-list draw-to-teach, event feed | the demo's core UX |
| 4 | Geometric event engine + Tier-2 VLM one-liners + DB persistence | events populate with grounded text |
| 5 | Detail view + scoped Q&A (Tier-3) + memory queries + timeline export | trust + interrogation + deliverable |
| 6 | Map panel (KLV track) + polish + `verify_offline` pass | full offline acceptance |

Per `DEMO.md` §6: context library, triage-feedback learning, and multi-stream are
visually present but lightly stubbed until the core proves out.

---

## 7. Open risks / honest limits

- **Text prompts are reliable on EO/daylight video** (in-distribution); example
  boxes + DINOv2 embeddings carry the **degraded / night / IR** edge cases. This is
  the general-UAV posture, softer than the thermal-only assumption in the source docs.
- The 7–8B local VLM writes shorter, plainer descriptions than the 30B — fine for
  the doc's deliberately terse grounded lines; the `full` profile is there when an
  H100 is available.
- "Chat with the live feed" is chat with an index updated every few seconds, not
  frame-perfect (doc §6). Surface tiers in the UI so the escalation is honest.
- SAM 3 offline availability is unconfirmed → the `sam2_gdino` fallback keeps the
  project unblocked either way.
- GPU passthrough to Docker requires the NVIDIA Container Toolkit configured in the
  WSL2 engine; validated as a Phase-1 gate.

---

## 8. Verification (end-to-end, offline)

1. Loop a UAV clip through `uav_udp_stream` over UDP to the backend container.
2. Confirm Tier-1 overlays + tracks render on the canvas in near real-time.
3. Draw boxes for the chosen concepts (e.g. `tent/person/vehicle`); confirm they go
   live and events populate the feed with Tier-2 one-liners.
4. Open an event → scoped Q&A returns a grounded answer (Tier-3).
5. Ask a memory question → DB query jumps to the earlier frame.
6. Export the incident timeline.
7. **Disable networking and repeat 1–6** — the acceptance test for "fully offline."

> Note: the source docs (`Architecture_and_Technology.md`, `DEMO.md`) are written
> with thermal-specific language. This plan generalizes to any UAV EO/IR feed; say
> the word if you want those two docs de-thermalized to match.
```
