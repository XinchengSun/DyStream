# Mic Realtime MP4 Runtime

This package contains the non-weight files needed by the lagged microphone-to-MP4 runtime:

```bash
cd /root/autodl-tmp/DyStream_cudagraph_streamtest && \
export OMP_NUM_THREADS=1 && \
export TRANSFORMERS_OFFLINE=1 && \
export HF_HUB_OFFLINE=1 && \
CUDA_VISIBLE_DEVICES=0,1 python -u realtime_mic_to_mp4.py \
  --feature_lag_frames 3 \
  --hop_ms 200 \
  --denoising_steps 1 \
  --motion_gpu 0 \
  --render_gpu 1 \
  --port 6008
```

In this repository, the equivalent checked-in helper is:

```bash
bash scripts/run_realtime_mic_to_mp4.sh
```

## Included Source Scope

The runtime source bundle includes:

```text
realtime_mic_to_mp4.py
realtime_mic_pipeline.py
app.py
utils.py
configs/motion_gen/sample.yaml
model/motion_generation/realtime_audio2face_cuda_graph.py
model/motion_generation/motion_gen_utils_dev.py
tools/visualization_0416/configs/
tools/visualization_0416/utils/face_detector.py
tools/visualization_0416/utils/model_0506/
img_files/3.png
img_files/11.png
requirements.txt
```

Generated outputs, Python caches, backup files, and model weights were excluded.

## External Assets

Place these assets before running:

```text
checkpoints/last.ckpt
tools/hf_models/wav2vec2-base-960h/
tools/visualization_0416/pretrained_model/epoch=0-step=312000.ckpt
tools/visualization_0416/utils/face_landmarker.task
```

`DYSTREAM_WAV2VEC2_PATH` can override the default Wav2Vec2 directory:

```bash
export DYSTREAM_WAV2VEC2_PATH=/path/to/wav2vec2-base-960h
```

`DYSTREAM_FACE_LANDMARKER_PATH` can override the default MediaPipe face landmarker path:

```bash
export DYSTREAM_FACE_LANDMARKER_PATH=/path/to/face_landmarker.task
```

The MP4 benchmark writes runtime outputs under:

```text
stream_outputs/mic_benchmark/
```
