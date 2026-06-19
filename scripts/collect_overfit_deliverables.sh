#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p deliverables/gt deliverables/official_infer deliverables/audio_overfit_infer deliverables/text_overfit_infer

cp -f data_overfit/sample_001/gt.mp4 deliverables/gt/sample_001.mp4
cp -f data_overfit/sample_002/gt.mp4 deliverables/gt/sample_002.mp4

echo "[deliverables] GT:"
find deliverables/gt -type f -name "*.mp4" -print
echo "[deliverables] official:"
find deliverables/official_infer -type f -name "*.mp4" -print || true
echo "[deliverables] audio overfit:"
find deliverables/audio_overfit_infer -type f -name "*.mp4" -print || true
echo "[deliverables] text overfit:"
find deliverables/text_overfit_infer -type f -name "*.mp4" -print || true
