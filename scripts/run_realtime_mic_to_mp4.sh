#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
python -u realtime_mic_to_mp4.py \
  --feature_lag_frames "${FEATURE_LAG_FRAMES:-3}" \
  --hop_ms "${HOP_MS:-200}" \
  --denoising_steps "${DENOISING_STEPS:-1}" \
  --motion_gpu "${MOTION_GPU:-0}" \
  --render_gpu "${RENDER_GPU:-1}" \
  --port "${PORT:-6008}"
