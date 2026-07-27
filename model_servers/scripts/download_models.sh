#!/usr/bin/env bash
# Download all model weights into ./models (run once, online).
# Edit the repo IDs / filenames to match the exact models you want — the values
# here are the PATTERN. Confirm names on Hugging Face before running.
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="$PWD/models/.hf"        # keep the cache local for air-gapping
mkdir -p models

dl() { echo ">> $*"; huggingface-cli download "$@"; }

# ---- Large VLM: your ~27B Q4 GGUF + mmproj (EDIT to match yours) --------------
: "${VLM_REPO:=<ORG>/<YOUR-27B-VLM-GGUF>}"
: "${VLM_FILE:=model.Q4_K_M.gguf}"
: "${VLM_MMPROJ_FILE:=mmproj-model-f16.gguf}"
dl "$VLM_REPO" "$VLM_FILE" "$VLM_MMPROJ_FILE" --local-dir models/vlm-27b

# ---- Qwen3-VL (GGUF for llama.cpp; pick a size that fits) ---------------------
: "${QWEN_REPO:=<ORG>/Qwen3-VL-8B-GGUF}"
dl "$QWEN_REPO" "Qwen3-VL-8B.Q4_K_M.gguf" "mmproj-Qwen3-VL-8B-f16.gguf" \
   --local-dir models/qwen3vl

# ---- SAM 3 (PyTorch, in-process perception) — confirm repo -------------------
dl "facebook/sam3" --local-dir models/sam3 || \
  echo "!! SAM 3 unavailable — use SAM 2.1 + GroundingDINO fallback (see perception/README.md)"

# ---- DINOv3 (PyTorch, in-process) — confirm repo -----------------------------
dl "facebook/dinov3-vitl16-pretrain" --local-dir models/dinov3

echo "done. weights in ./models (gitignored)."
