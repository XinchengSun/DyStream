#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/motion_gen/overfit_2samples.yaml}"
CKPT="${CKPT:-checkpoints/last.ckpt}"
EXP_NAME="${EXP_NAME:-official_overfit_inputs}"
DENOISING_STEPS="${DENOISING_STEPS:-5}"
TRAIN_BS="${TRAIN_BS:-1}"
VAL_BS="${VAL_BS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "${ROOT_DIR}"

if [ ! -f "${CKPT}" ]; then
  echo "[official-infer] missing checkpoint: ${CKPT}" >&2
  exit 1
fi

if [ ! -f data_json/overfit_test.json ]; then
  echo "[official-infer] missing data_json/overfit_test.json." >&2
  echo "[official-infer] Run first: DEVICE=cuda BATCH_SIZE=8 bash scripts/prepare_two_overfit_samples.sh" >&2
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
  data.train_bs="${TRAIN_BS}" \
  data.val_bs="${VAL_BS}" \
  data.num_workers="${NUM_WORKERS}" \
  model.module_name=model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder \
  resume_ckpt="${CKPT}"

latest_dir="$(ls -td outputs/*_${EXP_NAME} 2>/dev/null | head -n 1 || true)"
if [ -z "${latest_dir}" ]; then
  echo "[official-infer] inference finished, but no output dir matching outputs/*_${EXP_NAME} was found" >&2
  exit 1
fi

mkdir -p deliverables/official_infer
find "${latest_dir}" -type f -name "*.mp4" -print -exec cp -f {} deliverables/official_infer/ \;

echo "[official-infer] copied videos to deliverables/official_infer"
