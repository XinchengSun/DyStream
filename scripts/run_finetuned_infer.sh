#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/motion_gen/overfit_2samples.yaml}"
CKPT="${CKPT:?Set CKPT to the finetuned checkpoint path, for example outputs/.../last.ckpt}"
EXP_NAME="${EXP_NAME:-overfit_infer}"
DENOISING_STEPS="${DENOISING_STEPS:-5}"

cd "${ROOT_DIR}"

if [ ! -d tools ]; then
  echo "[finetuned-infer] missing tools/. Run: bash scripts/download_assets.sh" >&2
  exit 1
fi

if [ ! -f "${CKPT}" ]; then
  echo "[finetuned-infer] missing checkpoint: ${CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -u main.py \
  --config "${CONFIG}" \
  --override \
  exp_name="${EXP_NAME}" \
  debug=True \
  test=True \
  is_test=True \
  validation.denoising_steps="${DENOISING_STEPS}" \
  model.module_name=model.motion_generation.realtime_audio2face_cuda_graph \
  resume_ckpt="${CKPT}"
