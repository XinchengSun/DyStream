#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"

cd "${ROOT_DIR}"

for sample in sample_001 sample_002; do
  sample_dir="data_overfit/${sample}"
  if [ ! -f "${sample_dir}/gt.mp4" ]; then
    echo "[prepare] missing ${sample_dir}/gt.mp4" >&2
    exit 1
  fi

  if [ ! -f "${sample_dir}/audio.wav" ]; then
    ffmpeg -y -i "${sample_dir}/gt.mp4" -ac 1 -ar 16000 "${sample_dir}/audio.wav"
  fi

  if [ ! -f "${sample_dir}/ref.png" ]; then
    ffmpeg -y -i "${sample_dir}/gt.mp4" -frames:v 1 "${sample_dir}/ref.png"
  fi

  if [ ! -f "${sample_dir}/motion.npz" ]; then
    python scripts/extract_video_motion_latents.py \
      --video "${sample_dir}/gt.mp4" \
      --output "${sample_dir}/motion.npz" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}"
  fi
done

cat > data_json/overfit_items.json <<'JSON'
[
  {
    "video_id": "overfit_sample_001",
    "gt_video_path": "data_overfit/sample_001/gt.mp4",
    "image_path": "data_overfit/sample_001/ref.png",
    "audio_path": "data_overfit/sample_001/audio.wav",
    "motion_path": "data_overfit/sample_001/motion.npz",
    "caption": "adult male speaker, frontal cropped talking-head video, indoor interview lighting, neutral expression, English speech"
  },
  {
    "video_id": "overfit_sample_002",
    "gt_video_path": "data_overfit/sample_002/gt.mp4",
    "image_path": "data_overfit/sample_002/ref.png",
    "audio_path": "data_overfit/sample_002/audio.wav",
    "motion_path": "data_overfit/sample_002/motion.npz",
    "caption": "adult female speaker, frontal cropped talking-head video, indoor interview lighting, neutral expression, English speech"
  }
]
JSON

python scripts/prepare_overfit_metadata.py \
  --manifest data_json/overfit_items.json \
  --window-frames 96 \
  --fix-motion-key

echo "[prepare] overfit data ready"
