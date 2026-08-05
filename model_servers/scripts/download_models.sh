#!/usr/bin/env bash
# Download all model weights into ./models (run once, online).
# Intended for later air-gapped deployment.
#
# Uses `hf download` (huggingface_hub v1.0+) — the old `huggingface-cli
# download` command and its --resume-download / --local-dir-use-symlinks
# flags are gone as of v1.0. Resume is on by default now, and `--local-dir`
# writes real files (not symlinks) by default too, so neither flag is needed.

set -euo pipefail

cd "$(dirname "$0")/.."

# Keep Hugging Face cache inside the project
export HF_HOME="$PWD/models/.hf"

mkdir -p models

dl() {
    echo ">> $*"
    hf download "$@"
}

###############################################################################
# Qwen3-VL-8B-FP8 (vLLM) — primary VLM for the `local` (32GB) profile
###############################################################################
# 8B, not 30B-A3B: the 30B-A3B model doesn't fit a 32GB card alongside
# SAM3+DINOv3 at any quantization down to official FP8 (~31GB alone, zero
# headroom). 8B-FP8 is ~9-10GB — comfortable room for perception + KV cache.

: "${QWEN_REPO:=Qwen/Qwen3-VL-8B-Instruct-FP8}"

dl \
    "$QWEN_REPO" \
    --local-dir models/qwen3-vl-8b-instruct-fp8

###############################################################################
# Qwen3-VL-30B-A3B-Instruct (BF16) — VLM for the `full` (H100 80GB) profile
###############################################################################
# Only needed once you're actually deploying on the H100 — 58GB, no reason to
# pull it onto the 32GB local box otherwise. Full precision fits comfortably
# in 80GB alongside SAM3+DINOv3, no re-quantization needed.

: "${QWEN_FULL_REPO:=Qwen/Qwen3-VL-30B-A3B-Instruct}"

dl \
    "$QWEN_FULL_REPO" \
    --local-dir models/qwen3-vl-30b-a3b-instruct

###############################################################################
# Gemma 4 12B GGUF (llama.cpp standby)
###############################################################################

: "${GEMMA_REPO:=unsloth/gemma-4-12B-it-GGUF}"
: "${GEMMA_FILE:=gemma-4-12b-it-UD-Q5_K_XL.gguf}"
: "${GEMMA_MMPROJ_FILE:=mmproj-F16.gguf}"

# Confirm both filenames on the repo before running — Unsloth's "UD" dynamic
# quants and mmproj tags change as new ones are cut.
dl \
    "$GEMMA_REPO" \
    "$GEMMA_FILE" \
    "$GEMMA_MMPROJ_FILE" \
    --local-dir models/gemma4-12b

###############################################################################
# SAM 3 — gated, request access on the HF repo first, then `hf auth login`
###############################################################################

: "${SAM_REPO:=facebook/sam3}"

dl \
    "$SAM_REPO" \
    --local-dir models/sam3 \
|| echo "!! SAM 3 unavailable (access not granted?) — fall back to SAM 2.1 + GroundingDINO, see perception/README.md"

###############################################################################
# DINOv3 ViT-L (general purpose embeddings)
###############################################################################

: "${DINO_REPO:=facebook/dinov3-vitl16-pretrain-lvd1689m}"

dl \
    "$DINO_REPO" \
    --local-dir models/dinov3-vitl16

###############################################################################
# YOLOE-26 (Ultralytics) — exemplar/visual-prompt recall alongside SAM 3
###############################################################################
# Not on Hugging Face — a plain GitHub release asset. URL confirmed by
# actually running `ultralytics==8.4.115`'s own downloader against a real
# install and observing what it fetched (see
# model_servers/perception/interface.py's module docstring for why this
# project verifies rather than assumes): it pulled from
# github.com/ultralytics/assets' v8.4.0 release — hence the hardcoded
# release tag below; bump ASSETS_RELEASE if a newer `ultralytics` moves it.
# "s" size chosen as a starting point (small/fast — this only needs to run
# for concepts that have exemplars, on a GPU already shared by SAM 3 +
# DINOv3 + the VLM); override YOLOE_VARIANT for a larger/more accurate one.

: "${YOLOE_VARIANT:=yoloe-26s-seg.pt}"
: "${ASSETS_RELEASE:=v8.4.0}"

mkdir -p models/yoloe-26
curl -fL "https://github.com/ultralytics/assets/releases/download/${ASSETS_RELEASE}/${YOLOE_VARIANT}" \
    -o "models/yoloe-26/${YOLOE_VARIANT}" \
&& echo ">> saved models/yoloe-26/${YOLOE_VARIANT}" \
|| echo "!! YOLOE-26 download failed — check YOLOE_VARIANT/ASSETS_RELEASE against https://github.com/ultralytics/assets/releases"

###############################################################################

echo
echo "=============================================="
echo "Model download complete."
echo
echo "Directory layout:"
echo
echo "models/"
echo "├── .hf/"
echo "├── qwen3-vl-8b-instruct-fp8/   # local (32GB) profile — primary"
echo "├── qwen3-vl-30b-a3b-instruct/  # full (H100) profile only"
echo "├── gemma4-12b/                 # GGUF + mmproj together"
echo "├── sam3/"
echo "└── dinov3-vitl16/"
echo
echo "SAM 3 inference library (if needed):"
echo "pip install git+https://github.com/facebookresearch/sam3.git"
echo "=============================================="
