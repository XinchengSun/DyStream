#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/motion_gen/overfit_2samples_selected.yaml}"
CKPT="${CKPT:-checkpoints/last.ckpt}"
EXP_NAME="${EXP_NAME:-overfit_2samples_selected}"
MAX_STEPS="${MAX_STEPS:-8000}"
LR="${LR:-5e-5}"
TRAIN_BS="${TRAIN_BS:-16}"
VAL_BS="${VAL_BS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "${ROOT_DIR}"
mkdir -p logs

if [ ! -d tools ]; then
  echo "[selected-overfit] missing tools/. Run: bash scripts/download_assets.sh" >&2
  exit 1
fi

if [ ! -f "${CKPT}" ]; then
  echo "[selected-overfit] missing checkpoint: ${CKPT}" >&2
  exit 1
fi

if [ ! -f data_overfit_selected/sample_001/motion.npz ] || [ ! -f data_overfit_selected/sample_002/motion.npz ]; then
  echo "[selected-overfit] missing selected motion latents." >&2
  echo "[selected-overfit] Run: DEVICE=cuda BATCH_SIZE=16 bash scripts/prepare_selected_overfit_samples.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python scripts/preflight_overfit_train.py \
  --fix \
  --ckpt "${CKPT}" \
  --train-json data_json/overfit_train_selected.json \
  --test-json data_json/overfit_test_selected.json \
  --expected-test-items 2

python -u train.py \
  --config "${CONFIG}" \
  --override \
  exp_name="${EXP_NAME}" \
  resume_ckpt="${CKPT}" \
  resume_mode=weights_only \
  trainer.max_steps="${MAX_STEPS}" \
  solver.max_train_steps="${MAX_STEPS}" \
  solver.learning_rate="${LR}" \
  data.train_bs="${TRAIN_BS}" \
  data.val_bs="${VAL_BS}" \
  data.num_workers="${NUM_WORKERS}" \
  2>&1 | tee "logs/${EXP_NAME}.log"
