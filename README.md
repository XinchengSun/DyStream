# DyStream Two-Sample Overfit and Text-Condition Extension

This repository is a compact code package for the DyStream assessment work. It is not a full copy of the original DyStream repository. It contains the files that were added or modified for:

1. Two-sample overfit training with `train.py`.
2. Official-checkpoint, audio-overfit, and text-overfit inference.
3. Text-condition injection where `caption` is used as the prompt during training and inference.

The full file-by-file explanation is in:

```text
docs/overfit_text_code_map_20260618.md
```

## What Is Included

Important files:

```text
train.py
main.py
video_to_latent.py
datasets/single_dyadic_prev_audio.py
model/motion_generation/text_conditioned_audio2face.py
configs/motion_gen/overfit_2samples_selected.yaml
configs/motion_gen/overfit_2samples_selected_text.yaml
scripts/extract_video_motion_latents.py
scripts/prepare_overfit_metadata.py
scripts/prepare_selected_overfit_samples.sh
scripts/run_selected_overfit.sh
scripts/run_official_overfit_infer.sh
scripts/run_finetuned_infer.sh
scripts/run_text_finetuned_infer.sh
data_json/overfit_items_selected.json
data_json/overfit_train_selected.json
data_json/overfit_test_selected.json
```

## What Is Not Included

Large runtime assets are intentionally not included:

```text
checkpoints/
tools/
data_overfit_selected/
generated videos
raw source videos
```

The original DyStream repository, official checkpoint, and wrapping encoder/decoder assets are still required to run this code.

## Core Idea

The overfit experiment uses two manually selected talking-head clips. The selected train metadata is expanded into many sliding windows, instead of using only two metadata rows. This is important because DyStream trains on a fixed-length window; if the train JSON only has two rows, training only sees the beginning of each video.

The text-condition extension adds a text-conditioned model variant. Captions are read from metadata, passed through the dataset and training/inference code, encoded with a text encoder, and injected into the generation hidden states.

## Minimal Reproduction

Run inside a prepared DyStream checkout:

```bash
cd /ICML/ZJU/DyStream-main
```

Prepare selected metadata and motion latents:

```bash
TRAIN_STRIDE_FRAMES=1 DEVICE=cuda BATCH_SIZE=64 bash scripts/prepare_selected_overfit_samples.sh
```

If you already have a cropped `gt.mp4` and only need to rebuild the DyStream motion latent file, use either entry point below. They are equivalent:

```bash
python video_to_latent.py \
  --video data_overfit_selected/sample_001/gt.mp4 \
  --output data_overfit_selected/sample_001/motion.npz \
  --device cuda \
  --batch-size 64

python scripts/extract_video_motion_latents.py \
  --video data_overfit_selected/sample_001/gt.mp4 \
  --output data_overfit_selected/sample_001/motion.npz \
  --device cuda \
  --batch-size 64
```

This requires the original DyStream `tools/visualization_0416` assets and their wrapping/motion encoder checkpoints to be present in the checkout.

Pure audio overfit:

```bash
GPU_ID=0 \
TRAIN_BS=64 \
VAL_BS=2 \
MAX_STEPS=8000 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
EXP_NAME=overfit_2samples_selected_stride1_bs64 \
bash scripts/run_selected_overfit.sh
```

Text-conditioned overfit:

```bash
GPU_ID=0 \
TRAIN_BS=64 \
VAL_BS=2 \
MAX_STEPS=8000 \
CONFIG=configs/motion_gen/overfit_2samples_selected_text.yaml \
EXP_NAME=overfit_2samples_selected_text_stride1_bs64 \
bash scripts/run_selected_overfit.sh
```

Official checkpoint inference:

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
EXP_NAME=official_selected_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_official_overfit_infer.sh
```

Audio-overfit inference:

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
CKPT=checkpoints/package_audio_step1500.ckpt \
EXP_NAME=audio_selected_step1500_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_finetuned_infer.sh
```

Text-overfit inference:

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected_text.yaml \
CKPT=checkpoints/package_text_step1500.ckpt \
EXP_NAME=text_selected_step1500_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_text_finetuned_infer.sh
```

## Deliverable Structure

The local generated comparison package used this structure:

```text
deliverables/selected_comparison_step1500/
  01_gt/
  02_official_infer/
  03_audio_overfit_infer/
  04_text_overfit_infer/
```

Each folder contains two MP4 files, one for each selected sample.

## Notes

- `resume_mode=weights_only` loads official model weights without restoring the original trainer step or optimizer state.
- The text-conditioned implementation satisfies the requirement that `caption` is used as prompt during training. It is a first-pass conditioning implementation, not a mature strong-control model.
- If a checkpoint filename contains `=`, create a clean symlink before passing it through config overrides, for example `checkpoints/package_audio_step1500.ckpt`.

## Realtime Microphone MP4 Runtime

This repository also includes the non-weight files for the lagged realtime microphone-to-MP4 command collected from `/root/autodl-tmp/DyStream_cudagraph_streamtest`.

Run it with:

```bash
bash scripts/run_realtime_mic_to_mp4.sh
```

The helper defaults to the requested command:

```bash
OMP_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1 python -u realtime_mic_to_mp4.py \
  --feature_lag_frames 3 \
  --hop_ms 200 \
  --denoising_steps 1 \
  --motion_gpu 0 \
  --render_gpu 1 \
  --port 6008
```

Model weights are intentionally excluded from git. See `docs/realtime_mic_to_mp4.md` for the required local asset paths.
