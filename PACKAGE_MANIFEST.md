# DyStream Package Manifest

This repository is being used to organize the DyStream two-sample overfit and text-condition extension work.

## Uploaded to this GitHub repository

The following files have been uploaded successfully:

```text
README.md
docs/overfit_text_code_map_20260618.md
PACKAGE_MANIFEST.md
```

`README.md` explains the task goal, included code paths, reproduction commands, deliverable layout, and known caveats.

`docs/overfit_text_code_map_20260618.md` is the Chinese code map describing which files are used for overfit training, text-condition training, inference, and packaging.

## Complete local package

A complete local code package has been generated at:

```text
D:\桌面\DyStream-main\deliverables\overfit_text_code_package_20260618
D:\桌面\DyStream-main\deliverables\overfit_text_code_package_20260618.zip
```

The package intentionally contains our task-related code, configs, scripts, metadata, and documentation. It does not include large checkpoints, HuggingFace tools, raw source videos, or generated result videos.

## Complete package file tree

```text
overfit_text_code_package_20260618/
  README.md
  PACKAGE_MANIFEST.md
  video_to_latent.py
  docs/
    overfit_text_code_map_20260618.md
    current_task_status_20260529.md
  train.py
  main.py
  datasets/
    single_dyadic_prev_audio.py
  model/
    motion_generation/
      motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder_text.py
  configs/
    motion_gen/
      overfit_2samples_selected.yaml
      overfit_2samples_selected_text.yaml
      overfit_2samples.yaml
      overfit_2samples_text.yaml
  scripts/
    extract_video_motion_latents.py
    prepare_overfit_metadata.py
    prepare_selected_overfit_samples.sh
    preflight_overfit_train.py
    run_selected_overfit.sh
    run_official_overfit_infer.sh
    run_finetuned_infer.sh
    run_text_finetuned_infer.sh
    collect_overfit_deliverables.sh
    update_overfit_captions.py
    prepare_two_overfit_samples.sh
  data_json/
    overfit_items_selected.json
    overfit_train_selected.json
    overfit_test_selected.json
    overfit_items.template.json
```

## Recommended final upload command

When GitHub HTTPS access is stable on the local machine, upload the complete package with:

```bash
git clone https://github.com/XinchengSun/Xincheng-Sun.git
cd Xincheng-Sun
cp -r /path/to/overfit_text_code_package_20260618/* .
git add .
git commit -m "Add DyStream overfit and text-condition package"
git push origin main
```

On Windows PowerShell from the current project machine, the equivalent source path is:

```powershell
D:\桌面\DyStream-main\deliverables\overfit_text_code_package_20260618
```

## Notes

The code package is scoped to the assessment work:

1. Two-sample overfit training through `train.py`.
2. Official-checkpoint inference for baseline comparison.
3. Fine-tuned overfit inference.
4. Text-condition injection where caption is used as prompt during training and inference.
5. Video-to-motion-latent extraction for rebuilding `motion.npz` from cropped `gt.mp4`.

Large runtime dependencies must still be prepared from the original DyStream project:

```text
checkpoints/last.ckpt
tools/wrapping_encoder_decoder and related checkpoints
prepared two-sample data under data_overfit/
```

The motion latent extraction entry points are:

```text
video_to_latent.py
scripts/extract_video_motion_latents.py
```

## Realtime microphone MP4 runtime update

The following non-weight files were added for the lagged realtime microphone-to-MP4 command:

```text
app.py
mic_realtime_to_mp4_lag1.py
dual_gpu_mic_realtime_ui_v8_playbuffer.py
utils.py
requirements.txt
configs/motion_gen/sample.yaml
img_files/3.png
img_files/11.png
model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py
model/motion_generation/motion_gen_utils_dev.py
tools/visualization_0416/configs/
tools/visualization_0416/utils/
scripts/run_mic_realtime_to_mp4_lag1.sh
docs/mic_realtime_mp4_runtime.md
```

The checked-in run helper captures:

```bash
OMP_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1 python -u mic_realtime_to_mp4_lag1.py \
  --feature_lag_frames 3 \
  --hop_ms 200 \
  --denoising_steps 1 \
  --motion_gpu 0 \
  --render_gpu 1 \
  --port 6008
```

The following runtime assets remain external and are ignored by git:

```text
checkpoints/last.ckpt
tools/hf_models/wav2vec2-base-960h/
tools/visualization_0416/pretrained_model/epoch=0-step=312000.ckpt
tools/visualization_0416/utils/face_landmarker.task
```
