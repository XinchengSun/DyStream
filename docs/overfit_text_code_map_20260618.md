# DyStream 过拟合训练与文本条件改造文件说明

本仓库整理 DyStream 考核中两部分工作：

1. 两条数据过拟合训练。
2. 文本控制条件注入，训练时 caption 作为 prompt。

本仓库是代码整理包，不是完整 DyStream 工程。完整运行仍需要原 DyStream 代码、官方权重、tools wrapping encoder/decoder 和数据素材。

## 1. 过拟合实验链路

目标：

```text
官方权重 checkpoints/last.ckpt
-> 使用 train.py weights-only 加载模型参数
-> selected 两条数据展开训练窗口
-> 训练纯音频 overfit checkpoint
-> 推理得到过拟合后音视频
```

核心文件：

```text
train.py
configs/motion_gen/overfit_2samples_selected.yaml
scripts/prepare_selected_overfit_samples.sh
scripts/prepare_overfit_metadata.py
scripts/preflight_overfit_train.py
scripts/run_selected_overfit.sh
data_json/overfit_items_selected.json
data_json/overfit_train_selected.json
data_json/overfit_test_selected.json
```

关键改动：

- `train.py` 增加 `resume_mode=weights_only`。
- 只加载官方模型权重，不恢复官方 trainer step、optimizer、scheduler。
- `prepare_overfit_metadata.py` 增加 `--train-stride-frames`。
- 把两条视频展开成多个训练窗口，避免只训练开头 96 帧。

## 2. 文本条件注入链路

目标：

```text
metadata caption
-> dataset 读取 caption
-> train/main 传 caption
-> text model 编码 prompt
-> 注入 motion generation
-> 训练文本条件 overfit checkpoint
```

核心文件：

```text
datasets/single_dyadic_prev_audio.py
train.py
main.py
model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder_text.py
configs/motion_gen/overfit_2samples_selected_text.yaml
scripts/run_text_finetuned_infer.sh
```

关键改动：

- dataset 从 metadata item 读取 `caption`。
- `train.py` 训练时把 `caption` 传给模型 forward。
- `main.py` 推理时把 `caption` 传给模型 inference。
- 新增 `_text.py` 模型文件，增加 text encoder 和 text condition 注入。
- config 中打开 `use_text_condition: true`。

## 3. Selected 数据说明

selected 两条数据来自用户指定片段：

```text
sample_001: cDXtKSnAEQs.avi, 10:37-10:57, left half crop
sample_002: fKkpDbcnlFk.avi, 00:00-00:16, right half crop
```

每条样本期望目录：

```text
data_overfit_selected/sample_001/
  gt.mp4
  audio.wav
  ref.png
  ref_resize.png
  motion.npz

data_overfit_selected/sample_002/
  gt.mp4
  audio.wav
  ref.png
  ref_resize.png
  motion.npz
```

## 4. 最小复现命令

准备数据 metadata 和 motion latent：

```bash
cd /ICML/ZJU/DyStream-main
TRAIN_STRIDE_FRAMES=1 DEVICE=cuda BATCH_SIZE=64 bash scripts/prepare_selected_overfit_samples.sh
```

纯音频 overfit：

```bash
GPU_ID=0 \
TRAIN_BS=64 \
VAL_BS=2 \
MAX_STEPS=8000 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
EXP_NAME=overfit_2samples_selected_stride1_bs64 \
bash scripts/run_selected_overfit.sh
```

文本条件 overfit：

```bash
GPU_ID=0 \
TRAIN_BS=64 \
VAL_BS=2 \
MAX_STEPS=8000 \
CONFIG=configs/motion_gen/overfit_2samples_selected_text.yaml \
EXP_NAME=overfit_2samples_selected_text_stride1_bs64 \
bash scripts/run_selected_overfit.sh
```

官方权重推理：

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
EXP_NAME=official_selected_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_official_overfit_infer.sh
```

纯音频 overfit 推理：

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
CKPT=checkpoints/package_audio_step1500.ckpt \
EXP_NAME=audio_selected_step1500_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_finetuned_infer.sh
```

文本条件 overfit 推理：

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected_text.yaml \
CKPT=checkpoints/package_text_step1500.ckpt \
EXP_NAME=text_selected_step1500_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_text_finetuned_infer.sh
```

## 5. 交付目录

本地交付目录：

```text
D:\桌面\DyStream-main\deliverables\selected_comparison_step1500
```

结构：

```text
01_gt/
02_official_infer/
03_audio_overfit_infer/
04_text_overfit_infer/
```

每个目录两条 mp4，并且都包含 audio + video stream。

## 6. 注意事项

- 文本条件实现是第一版，满足 caption 作为 prompt 参与训练和推理，但不是强控制模型。
- validation 自动转视频会额外吃显存，两个训练同时跑时可能 OOM；checkpoint 保存后单独推理更稳。
- 如果 checkpoint 文件名带 `=`，建议建立不带等号的软链接，例如 `checkpoints/package_audio_step1500.ckpt`。
