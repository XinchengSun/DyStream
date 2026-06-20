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
