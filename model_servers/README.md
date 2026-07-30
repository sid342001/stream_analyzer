# `model_servers/` — Local model hosting

Everything model-related for the UAV Stream Analysis System lives here: the
VLM server(s), the perception models (SAM 3 + DINOv3), the weights, the download
and hosting instructions, and the profile configs. Everything runs **locally, in
Docker, offline** (after a one-time download).

---

## 1. What is hosted, and how

Two very different kinds of model, served two different ways — on purpose.

| Model | Kind | Served via | Path | Why |
|---|---|---|---|---|
| **Qwen3-VL-30B-A3B** | multimodal LLM | **vLLM** `vllm/vllm-openai` (container) | primary reasoning path, Tier 2–3 | HF-format MoE; OpenAI-compatible; see the Gemma 4 comparison below for why this stays primary |
| **Gemma 4 12B** (encoder-free, GGUF+mmproj) | multimodal LLM | **llama.cpp** `llama-server` (container), optional | standby/alt VLM, hot-swap only | brought up later if/when needed — see §1.3 |
| **SAM 3** | vision transformer (PyTorch) | container today (`perception-api`, :8090); in-process once `backend/` exists | perception, **Tier 1, every sampled frame** | real-time hot path; no HTTP overhead once in-process |
| **DINOv3** | vision transformer (PyTorch) | same container as SAM 3 | verifier, embeds SAM 3 crops | tight pipeline with SAM 3; share the frame tensor |

> **Current state vs. eventual state:** `backend/` (the FastAPI orchestrator) doesn't
> exist yet. Until it does, SAM 3 + DINOv3 are hosted as their own `perception-api`
> container so they're actually runnable today. Once `backend/` is built, it should
> import `perception/` in-process and this container becomes debug-only again — see
> §1.3.

### 1.1 Qwen3-VL via vLLM (primary)

Qwen3-VL-30B-A3B is hosted HF-format through vLLM rather than GGUF through
llama.cpp. vLLM keeps the full MoE resident on GPU (no CPU-offload path, unlike
llama.cpp's `-ngl`), which is fine at Q-equivalent FP8/AWQ sizes on a 32 GB card,
and `vllm/vllm-openai` exposes the same OpenAI-compatible `/v1/chat/completions`
with image inputs, so the orchestrator's call shape doesn't change. `llama.cpp`
is kept in the stack (the `vlm` service) specifically as the engine for the
Gemma 4 12B standby below — GGUF+mmproj is llama.cpp-native, not vLLM's format.

### 1.2 Gemma 4 12B — optional standby, hosted later

An encoder-free multimodal model (Google, Apache 2.0) considered as an
alternative to Qwen3-VL. Decision: **keep Qwen3-VL primary**, host Gemma 4 12B
as an optional hot-swappable standby via the `vlm` (llama.cpp) service once you
download it — reasons:

- **Task fit.** Qwen3-VL is explicitly trained/benchmarked on grounding tasks
  (RefCOCO/+/g, ODinW-13, CountBench) — the closer match to "describe/answer
  about what SAM 3 already located," which is all the VLM does in this
  pipeline (Tier 2 one-liners off SAM 3 crops, Tier 3 grounded Q&A). Gemma 4's
  benchmark edge is in domains (STEM reasoning, native audio) this pipeline
  doesn't touch.
- **Maturity.** Gemma 4 12B's encoder-free design is a brand-new code path in
  llama.cpp (added ~June 2026) and already had a real crash bug from the exact
  mechanism that makes it novel (zero attention-head-dim math with no vision
  encoder — `ggml-org/llama.cpp` issue #24085, fixed in PR #24088). Qwen3-VL's
  encoder+mmproj path is the long-established pattern the GGUF vision ecosystem
  is built around.
- **Why keep it at all:** smaller footprint than the 30B-class primary, one
  unified text/image/audio architecture, and audio becomes relevant if
  acoustic UAV sensors are ever added. Worth having as a fast, light fallback
  or a second opinion — not as the default.

Bring it up only once hosted: `docker compose up -d vlm` (don't run it and
`vllm` together — they share port 8080).

### 1.3 Why SAM 3 + DINOv3 stay in-memory (not an API) — the eventual design

- They are **not** LLMs — llama.cpp/vLLM don't serve them; they run under PyTorch.
- SAM 3 runs on **every sampled frame** (Tier 1). DINOv3 embeds the **crops SAM 3
  just produced**. Putting an HTTP boundary between them (or between them and the
  pipeline) means, per frame: JPEG-encode → POST → decode → to-tensor → infer →
  RLE-serialize masks → POST back → parse. That overhead is pure loss on the hot path.
- **Decision: load both in one process, co-resident on the GPU, imported in-process
  by the backend.** The decoded frame tensor is handed to SAM 3 directly; SAM 3's
  crops go straight to DINOv3. Lowest latency, simplest data path.
- The code lives here as an importable package (`perception/`); the **backend
  container installs and calls it in-process**. Weights live in `models/`.
- A thin FastAPI wrapper (`perception/service.py`) is included **for debugging and
  non-realtime clients only** — the live path never uses it.

This is Docker-compatible: "in-process" means inside the backend container's Python
process. The perception package + weights are hosted here; the backend imports them.

---

## 2. VRAM budget on the 32 GB card

Qwen3-VL (vLLM) + SAM 3 + DINOv3 share one GPU today, as three separate
containers. Realistic footprints:

| Resident set | Approx. VRAM | Fits 32 GB? |
|---|---|---|
| Qwen3-VL-30B-A3B (vLLM, FP8/AWQ) + SAM 3 + DINOv3 (ViT-L) | ~22–28 GB | ✅ comfortable |
| …plus Gemma 4 12B standby also resident | ~35–42 GB | ❌ too tight — hot-swap instead, don't run both |
| 30B-class VLM FP8 + SAM 3 + DINOv3 (`full` profile) | ~40 GB | ❌ needs the H100 (`full` profile) |

**Guidance:** run **one VLM at a time** as the reasoning brain, with SAM 3 +
DINOv3 always resident for Tier 1. Bring Gemma 4 12B up only when you actually
want it: stop `vllm`, start `vlm` (llama.cpp) — don't run both at once, they
share port 8080 and the combined footprint doesn't fit.

Profiles capture this: `profiles/local.env` (this box) and `profiles/full.env`
(H100). Compose reads one via `env_file`.

---

## 3. Prerequisites (one-time, needs internet)

```bash
# 1) NVIDIA Container Toolkit so Docker can see the GPU (inside WSL2 Ubuntu)
#    verify with:
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# 2) Hugging Face CLI for downloads (host or a throwaway container)
pip install -U "huggingface_hub[cli]"

# 3) (gated models) log in once; SAM 3 and some VLM repos may require accepting terms
huggingface-cli login
```

---

## 4. Download the models

All weights land in `./models/` (gitignored) and are mounted **read-only** into
the containers. Run `scripts/download_models.sh` or the commands below.

> ⚠️ **Confirm exact quant tags on Hugging Face before downloading** — the repo
> IDs below are confirmed real as of this writing, but specific quant files move.

```bash
export HF_HOME=$PWD/models/.hf          # keep the HF cache local for air-gapping
cd model_servers

# --- Primary VLM: Qwen3-VL, HF format for vLLM (not GGUF) -------------------------
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct \
    --local-dir models/qwen3-vl-30b-a3b-instruct

# --- SAM 3 (PyTorch weights) — gated, request access on the HF repo first --------
huggingface-cli login   # after your access request is accepted
huggingface-cli download facebook/sam3 --local-dir models/sam3

# --- DINOv3 (PyTorch) -------------------------------------------------------------
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir models/dinov3

# --- Optional, only when you're ready to host the standby: Gemma 4 12B GGUF ------
huggingface-cli download unsloth/gemma-4-12B-it-GGUF \
    gemma-4-12b-it-Q4_K_M.gguf  mmproj-gemma-4-12b-it-f16.gguf \
    --local-dir models/gemma4-12b
```

Fallback if SAM 3 access isn't granted yet: download **SAM 2.1** +
**GroundingDINO** instead and set `PERCEPTION_BACKEND=sam2_gdino` (see
`perception/README.md`).

---

## 5. Host the primary VLM — Qwen3-VL via vLLM

```bash
cd model_servers
cp profiles/local.env .env         # or: ln -s profiles/local.env .env
docker compose --profile vllm up -d vllm
docker compose logs -f vllm        # watch it load
```

What Compose runs under the hood (GPU, HF format, OpenAI API on :8080):

```bash
vllm serve /models/qwen3-vl-30b-a3b-instruct \
  --port 8080 --served-model-name uav-vlm \
  --gpu-memory-utilization 0.85 --max-model-len 32768
```

**Health check + a real vision call:**

```bash
curl -s localhost:8080/health

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "uav-vlm",
  "messages": [{"role":"user","content":[
    {"type":"text","text":"One terse line: what is in this crop?"},
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'"$(base64 -w0 sample.jpg)"'"}}
  ]}],
  "max_tokens": 64
}'
```

---

## 6. Host the perception models — SAM 3 + DINOv3

There's no `backend/` orchestrator yet to import `perception/` in-process (the
eventual design, §1.3), so host them as their own container for now:

```bash
docker compose --profile debug up -d perception-api   # exposes :8090
docker compose logs -f perception-api                 # watch weights load
```

```bash
curl -s localhost:8090/health

curl -s localhost:8090/detect_track \
  -F "image=@sample.jpg" \
  -F 'concepts=[{"label":"person","text_prompt":"person"}]'
```

Point it at the weights via `.env` (already set by the profile):
`SAM_WEIGHTS=/models/sam3`, `DINOV3_WEIGHTS=/models/dinov3`,
`PERCEPTION_BACKEND=sam3|sam2_gdino`.

When `backend/` gets built, switch to importing `perception/` directly instead
(see `perception/interface.py` and §1.3) and retire this container back to
debug-only use.

---

## 7. Optional: Gemma 4 12B standby via llama.cpp

Bring it up only once you've downloaded it (§4) and actually want it — see §1.2
for why this stays a standby, not the default:

```bash
docker compose up -d vlm           # stop `vllm` first — they share port 8080
docker compose logs -f vlm
```

Compose runs `llama-server -m <gguf> --mmproj <mmproj> --host 0.0.0.0 --port 8080
-ngl 99 -c 8192 --parallel 2 --flash-attn --alias uav-vlm` — same OpenAI API
shape, same `/v1/chat/completions` calls, so nothing downstream needs to change
when you swap between it and `vllm`.

---

## 8. Offline / air-gap

1. Download everything in §3–§4 once (online).
2. `docker compose pull` base images, then `docker save` them + your built images
   into a tarball for the air-gapped box.
3. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (already in the profiles) so
   nothing phones home.
4. `scripts/verify_offline.sh` brings the stack up with networking disabled as the
   acceptance test.

Weights and the HF cache live under `./models/` and are mounted read-only — nothing
is fetched at runtime.

---

## 9. Layout

```
model_servers/
├─ README.md                # this file
├─ docker-compose.yml       # vllm (primary) + perception-api + vlm (Gemma 4 standby)
├─ .env                     # copy of the active profile (gitignored)
├─ profiles/
│  ├─ local.env             # RTX 5000 Ada 32 GB: Qwen3-VL (vLLM) + SAM 3 + DINOv3
│  └─ full.env              # H100 80 GB (the architecture doc)
├─ llamacpp/                # entrypoint / tuning notes for the Gemma 4 standby
├─ vllm/                    # notes for the primary vLLM service
├─ perception/              # SAM 3 + DINOv3 — Dockerfile + FastAPI service today;
│  │                        # importable in-process once backend/ exists (§1.3)
│  ├─ README.md
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ interface.py          # Perception.detect_track / embed (the shared API)
│  └─ service.py            # FastAPI wrapper, :8090
├─ scripts/
│  ├─ download_models.sh
│  └─ verify_offline.sh
└─ models/                  # weights + HF cache (gitignored, mounted read-only)
```
