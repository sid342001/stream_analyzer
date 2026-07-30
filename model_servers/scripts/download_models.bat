@echo off
setlocal EnableDelayedExpansion

:: Download all model weights into ./models (run once, online).
:: Intended for later air-gapped deployment.
::
:: Uses `hf download` (huggingface_hub v1.0+) — the old `huggingface-cli
:: download` command and its --resume-download / --local-dir-use-symlinks
:: flags are gone as of v1.0. Resume is on by default now, and `--local-dir`
:: writes real files (not symlinks) by default too, so neither flag is needed.

:: Move to the model_servers directory (parent of the scripts directory)
cd /d "%~dp0.."

:: Keep Hugging Face cache inside the project
set "HF_HOME=%CD%\models\.hf"

if not exist "models" mkdir models

:: ###############################################################################
:: Qwen3-VL (vLLM / Transformers) - primary VLM
:: ###############################################################################

set "QWEN_REPO=Qwen/Qwen3-VL-30B-A3B-Instruct"

echo ^>^> hf download %QWEN_REPO% --local-dir models\qwen3-vl-30b-a3b-instruct
call hf download "%QWEN_REPO%" --local-dir models\qwen3-vl-30b-a3b-instruct

:: ###############################################################################
:: Gemma 4 12B GGUF (llama.cpp standby)
:: ###############################################################################

set "GEMMA_REPO=unsloth/gemma-4-12B-it-GGUF"
set "GEMMA_FILE=gemma-4-12b-it-UD-Q5_K_XL.gguf"
set "GEMMA_MMPROJ_FILE=mmproj-F16.gguf"

:: Confirm both filenames on the repo before running — Unsloth's "UD" dynamic
:: quants and mmproj tags change as new ones are cut.
echo ^>^> hf download %GEMMA_REPO% %GEMMA_FILE% %GEMMA_MMPROJ_FILE% --local-dir models\gemma4-12b
call hf download "%GEMMA_REPO%" "%GEMMA_FILE%" "%GEMMA_MMPROJ_FILE%" --local-dir models\gemma4-12b

:: ###############################################################################
:: SAM 3 - gated, request access on the HF repo first, then `hf auth login`
:: ###############################################################################

set "SAM_REPO=facebook/sam3"

echo ^>^> hf download %SAM_REPO% --local-dir models\sam3
call hf download "%SAM_REPO%" --local-dir models\sam3
if %errorlevel% neq 0 (
    echo !! SAM 3 unavailable ^(access not granted?^) - fall back to SAM 2.1 + GroundingDINO, see perception\README.md
)

:: ###############################################################################
:: DINOv3 ViT-L (general purpose embeddings)
:: ###############################################################################

set "DINO_REPO=facebook/dinov3-vitl16-pretrain-lvd1689m"

echo ^>^> hf download %DINO_REPO% --local-dir models\dinov3-vitl16
call hf download "%DINO_REPO%" --local-dir models\dinov3-vitl16

:: ###############################################################################

echo.
echo ==============================================
echo Model download complete.
echo.
echo Directory layout:
echo.
echo models\
echo +-- .hf\
echo +-- qwen3-vl-30b-a3b-instruct\
echo +-- gemma4-12b\            # GGUF + mmproj together
echo +-- sam3\
echo +-- dinov3-vitl16\
echo.
echo SAM 3 inference library ^(if needed^):
echo pip install git+https://github.com/facebookresearch/sam3.git
echo ==============================================

endlocal
