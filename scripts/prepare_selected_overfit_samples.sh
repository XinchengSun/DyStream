#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_STRIDE_FRAMES="${TRAIN_STRIDE_FRAMES:-16}"

cd "${ROOT_DIR}"

for sample in sample_001 sample_002; do
  sample_dir="data_overfit_selected/${sample}"
  if [ ! -f "${sample_dir}/gt.mp4" ]; then
    echo "[prepare-selected] missing ${sample_dir}/gt.mp4" >&2
    exit 1
  fi

  if [ ! -f "${sample_dir}/audio.wav" ]; then
    ffmpeg -y -i "${sample_dir}/gt.mp4" -ac 1 -ar 16000 "${sample_dir}/audio.wav"
  fi

  if [ ! -f "${sample_dir}/ref.png" ]; then
    ffmpeg -y -i "${sample_dir}/gt.mp4" -frames:v 1 "${sample_dir}/ref.png"
  fi

  if [ ! -f "${sample_dir}/ref_resize.png" ]; then
    cp "${sample_dir}/ref.png" "${sample_dir}/ref_resize.png"
  fi

  if [ ! -f "${sample_dir}/motion.npz" ]; then
    python scripts/extract_video_motion_latents.py \
      --video "${sample_dir}/gt.mp4" \
      --output "${sample_dir}/motion.npz" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}"
  fi
done

python scripts/prepare_overfit_metadata.py \
  --manifest data_json/overfit_items_selected.json \
  --train-out data_json/overfit_train_selected.json \
  --test-out data_json/overfit_test_selected.json \
  --window-frames 96 \
  --train-stride-frames "${TRAIN_STRIDE_FRAMES}" \
  --fix-motion-key

echo "[prepare-selected] selected overfit data ready"
