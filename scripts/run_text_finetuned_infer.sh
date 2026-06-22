#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/motion_gen/overfit_2samples_text.yaml}"
CKPT="${CKPT:?Set CKPT to the text-conditioned checkpoint path, for example outputs/.../last.ckpt}"
EXP_NAME="${EXP_NAME:-text_overfit_infer}"
DENOISING_STEPS="${DENOISING_STEPS:-5}"
PROMPT_OVERRIDE="${PROMPT_OVERRIDE:-}"

cd "${ROOT_DIR}"
mkdir -p logs deliverables/text_overfit_infer

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -u main.py \
  --config "${CONFIG}" \
  --override \
  exp_name="${EXP_NAME}" \
  debug=True \
  test=True \
  is_test=True \
  validation.denoising_steps="${DENOISING_STEPS}" \
  model.module_name=model.motion_generation.text_conditioned_audio2face \
  prompt_override="${PROMPT_OVERRIDE}" \
  resume_ckpt="${CKPT}" \
  2>&1 | tee "logs/${EXP_NAME}.log"

latest_dir="$(ls -td outputs/*_${EXP_NAME} 2>/dev/null | head -n 1 || true)"
if [ -z "${latest_dir}" ]; then
  echo "[FAIL] no output directory found for ${EXP_NAME}" >&2
  exit 1
fi

find "${latest_dir}" -type f -name "*.mp4" -print -exec cp -f {} deliverables/text_overfit_infer/ \;
echo "[OK] copied text-conditioned videos to deliverables/text_overfit_infer"
