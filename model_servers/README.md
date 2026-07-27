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
| Large VLM — your **~27B Q4 GGUF + mmproj** | multimodal LLM | **llama.cpp** `llama-server` (container) | slow/reasoning path, Tier 2–3 | GGUF+mmproj is llama.cpp-native; OpenAI-compatible; hot-swappable |
| **Qwen3-VL** (GGUF) | multimodal LLM | **llama.cpp** `llama-server` (container) | alt/second VLM | same server, same API — just another `-m` |
| **SAM 3** | vision transformer (PyTorch) | **in-memory, in the backend process** | perception, **Tier 1, every sampled frame** | real-time hot path; no HTTP overhead |
| **DINOv3** | vision transformer (PyTorch) | **in-memory, same process as SAM 3** | verifier, embeds SAM 3 crops | tight pipeline with SAM 3; share the frame tensor |

### 1.1 Why llama.cpp for the VLMs (not vLLM)

- **Format match.** You want **Q4 GGUF + mmproj**. That is llama.cpp's native
  format. vLLM consumes HF-format weights with AWQ/GPTQ/FP8 — it does **not** load
  mmproj GGUF. Choosing llama.cpp means your stated artifacts run as-is.
- **VRAM efficiency + CPU offload.** llama.cpp fits large models on a 32 GB card
  and can spill layers to CPU RAM (you have 256 GB) via `-ngl`, so a 27B Q4 runs
  even when VRAM is tight. vLLM keeps everything resident on the GPU.
- **Single-box, few-user, real-time.** One operator (or a few). llama.cpp's
  single-stream latency is excellent and the server is one lightweight container.
- **Hot-swap.** Swapping the 27B for Qwen3-VL is changing one `-m` flag / profile,
  not re-provisioning a serving engine.
- **OpenAI-compatible + vision.** `llama-server` exposes `/v1/chat/completions`
  with image inputs, so the orchestrator calls it exactly like any OpenAI tool.

> **When vLLM would win:** many concurrent sessions with heavy batched throughput,
> or FP8 tensor-parallel across datacenter GPUs. Not this deployment. A vLLM
> alternative is stubbed in `vllm/` if you ever need it — see §7.

### 1.2 Why SAM 3 + DINOv3 stay in-memory (not an API)

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

llama.cpp VLM + in-memory perception share one GPU. Realistic footprints:

| Resident set | Approx. VRAM | Fits 32 GB? |
|---|---|---|
| 27B Q4_K_M VLM (+mmproj ~1 GB) + SAM 3 + DINOv3 (ViT-L) | ~22–26 GB | ✅ comfortable |
| …plus a second VLM (Qwen3-VL-8B Q4, ~6 GB) resident | ~28–33 GB | ⚠️ too tight — hot-swap instead |
| 30B-class VLM FP8 + SAM 3 + DINOv3 (`full` profile) | ~40 GB | ❌ needs the H100 (`full` profile) |

**Guidance:** run **one large VLM at a time** as the reasoning brain, with SAM 3 +
DINOv3 always resident for Tier 1. If you want both the 27B and Qwen3-VL available,
**hot-swap** them in `llama-server` (load on request / restart the `vlm` service
with the other profile) rather than keeping both resident. Use `-ngl` to offload
VLM layers to CPU RAM if you need extra headroom.

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

> ⚠️ **Confirm exact repo IDs and quant tags on Hugging Face before downloading.**
> Model naming moves fast; the IDs below are the *pattern*, not gospel. In
> particular, verify the precise name of your "~27B Q4 mmproj" VLM — set it in
> `profiles/local.env`.

```bash
export HF_HOME=$PWD/models/.hf          # keep the HF cache local for air-gapping
cd model_servers

# --- Large VLM: your ~27B Q4 GGUF + mmproj (edit repo/files to match yours) -------
huggingface-cli download <ORG>/<YOUR-27B-VLM-GGUF> \
    <model>.Q4_K_M.gguf  mmproj-<model>-f16.gguf \
    --local-dir models/vlm-27b

# --- Qwen3-VL (GGUF for llama.cpp; pick a size that fits, e.g. 8B) ----------------
huggingface-cli download <ORG>/Qwen3-VL-8B-GGUF \
    Qwen3-VL-8B.Q4_K_M.gguf  mmproj-Qwen3-VL-8B-f16.gguf \
    --local-dir models/qwen3vl

# --- SAM 3 (PyTorch weights, loaded in-process by perception) --------------------
huggingface-cli download facebook/sam3 --local-dir models/sam3            # confirm repo

# --- DINOv3 (PyTorch, in-process) ------------------------------------------------
huggingface-cli download facebook/dinov3-vitl16-pretrain --local-dir models/dinov3   # confirm repo
```

Fallback if SAM 3 weights aren't available to you: download **SAM 2.1** +
**GroundingDINO** instead and set `PERCEPTION_BACKEND=sam2_gdino` (see
`perception/README.md`).

---

## 5. Host the VLM (llama.cpp)

The VLM server is defined in `docker-compose.yml` and parameterized by the active
profile. Bring it up:

```bash
cd model_servers
cp profiles/local.env .env         # or: ln -s profiles/local.env .env
docker compose up -d vlm
docker compose logs -f vlm         # watch it load
```

What Compose runs under the hood (GPU, GGUF + mmproj, OpenAI API on :8080):

```bash
llama-server \
  -m       /models/vlm-27b/<model>.Q4_K_M.gguf \
  --mmproj /models/vlm-27b/mmproj-<model>-f16.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 8192 --parallel 2 --flash-attn \
  --alias uav-vlm
```

- `-ngl 99` = offload all layers to GPU; **lower it** (e.g. `-ngl 60`) to spill to
  CPU RAM and free VRAM for perception.
- `-c 8192` = context; images consume context, keep prompts terse (matches the
  doc's "terse grounded lines").
- `--parallel 2` = a couple of concurrent slots; raise only if you have headroom.

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

**Hot-swap to Qwen3-VL:** point the profile at `models/qwen3vl/…` and
`docker compose up -d --force-recreate vlm`. Same endpoint, same API.

---

## 6. Host the perception models (SAM 3 + DINOv3, in-memory)

These are **not** started as their own live service. The **backend container**
installs the `perception/` package and loads the weights once at startup:

```python
# in the backend, at startup (illustrative — see perception/interface.py)
from perception import Perception
percept = Perception(profile_env())      # loads SAM 3 + DINOv3 onto the GPU, once
# per sampled frame, in-process, no HTTP:
dets = percept.detect_track(frame, concepts, exemplars)
crops_emb = percept.embed(frame, dets)   # DINOv3 on SAM 3's crops
```

Point the backend at the weights via env (set in the top-level compose):
`SAM_WEIGHTS=/models/sam3`, `DINOV3_WEIGHTS=/models/dinov3`,
`PERCEPTION_BACKEND=sam3|sam2_gdino`.

Optional debug API (never on the live path):
```bash
docker compose --profile debug up -d perception-api   # exposes :8090 for poking
```

---

## 7. Optional: vLLM alternative

If you later need high-concurrency serving, `vllm/` holds a Compose service for an
HF-format Qwen3-VL (AWQ/FP8). It exposes the **same OpenAI API on :8080**, so the
orchestrator is unchanged. Do not run it *and* llama.cpp on the same port/GPU at
once — pick one VLM engine. It cannot load your GGUF/mmproj files.

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
├─ docker-compose.yml       # vlm (llama.cpp) + optional perception-api / vllm
├─ .env                     # copy of the active profile (gitignored)
├─ profiles/
│  ├─ local.env             # RTX 5000 Ada 32 GB: 27B Q4 + SAM 3 + DINOv3
│  └─ full.env              # H100 80 GB: 30B FP8 (the architecture doc)
├─ llamacpp/                # entrypoint / tuning notes for llama-server
├─ vllm/                    # optional vLLM alternative (§7)
├─ perception/              # SAM 3 + DINOv3 in-memory package (imported by backend)
│  ├─ README.md
│  └─ interface.py          # Perception.detect_track / embed  (the in-process API)
├─ scripts/
│  ├─ download_models.sh
│  └─ verify_offline.sh
└─ models/                  # weights + HF cache (gitignored, mounted read-only)
```
