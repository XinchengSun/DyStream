# DyStream 过拟合训练与文本条件改造文件说明

更新时间：2026-06-18

本地项目目录：

```text
D:\桌面\DyStream-main
```

服务器项目目录：

```text
/ICML/ZJU/DyStream-main
```

这份文档说明当前项目里与“两条数据过拟合训练”和“文本条件注入”相关的主要文件分别负责什么，方便后续复现、检查和交付。

## 1. 核心改造目标

本次主要做了两条链路：

1. 纯音频两条数据过拟合：

```text
官方权重 checkpoints/last.ckpt
-> weights-only 加载模型参数
-> selected 两条数据窗口化训练
-> 输出 audio overfit 推理视频
```

2. 文本条件两条数据过拟合：

```text
caption/prompt 写入 metadata
-> dataset 读取 caption
-> train/main 把 caption 传给模型
-> 文本条件模型编码 caption 并注入音频 hidden
-> 输出 text overfit 推理视频
```

## 2. 数据与 Metadata 文件

### `data_overfit_selected/sample_001/`

第一条 selected overfit 数据目录。

来源：

```text
D:\桌面\demo\cDXtKSnAEQs.avi
10:37 - 10:57
左半部分裁剪
```

主要文件：

```text
gt.mp4          原始裁剪后的 GT 视频
audio.wav       从同一时间段抽出的 16k mono 音频
ref.png         参考图
ref_resize.png  参考图，供 DyStream 推理/warpping 使用
motion.npz      用 DyStream motion/wrapping encoder 提取的 motion latent
preview.jpg     数据检查预览图
```

### `data_overfit_selected/sample_002/`

第二条 selected overfit 数据目录。

来源：

```text
D:\桌面\demo\fKkpDbcnlFk.avi
00:00 - 00:16
右半部分裁剪
```

主要文件同 `sample_001`。

### `data_json/overfit_items_selected.json`

selected 两条原始样本的 manifest。

作用：

- 记录每条样本的 `gt.mp4`、`audio.wav`、`ref.png`、`motion.npz`
- 记录 `caption`
- 是生成 train/test metadata 的输入

### `data_json/overfit_train_selected.json`

selected 训练 metadata。

作用：

- 不是简单两条数据，而是把两条视频展开成多个训练窗口
- 解决“只训练每条视频最前面 96 帧”的问题
- 训练时 dataset 会按照每个 item 的 `start_idx/end_idx` 取窗口

关键原因：

DyStream 的训练窗口长度由配置控制，例如：

```text
cbh_window_length = 96
```

如果 train json 只有两条 item，就基本只训练每条视频开头一段。  
所以这里需要把完整视频展开成很多窗口，才能真正 overfit 整条视频。

### `data_json/overfit_test_selected.json`

selected 测试 metadata。

作用：

- 保持两条完整测试样本
- 用于官方权重推理、纯音频 overfit 推理、文本 overfit 推理
- 交付视频的四组对比都基于这个 test json

## 3. 数据准备脚本

### `scripts/prepare_selected_overfit_samples.sh`

selected 数据准备入口脚本。

作用：

- 检查 `data_overfit_selected/sample_001` 和 `sample_002`
- 确认 `gt.mp4/audio.wav/ref.png/ref_resize.png` 存在
- 调用 motion latent 提取脚本生成 `motion.npz`
- 调用 `prepare_overfit_metadata.py` 生成 train/test json

常用命令：

```bash
cd /ICML/ZJU/DyStream-main
TRAIN_STRIDE_FRAMES=1 DEVICE=cuda BATCH_SIZE=64 bash scripts/prepare_selected_overfit_samples.sh
```

### `scripts/prepare_overfit_metadata.py`

metadata 生成脚本。

这份脚本是关键改造点之一。

作用：

- 从 `overfit_items_selected.json` 生成 train/test metadata
- 支持 caption 写入
- 支持 `--train-stride-frames`
- 能把一条长视频展开成多个训练窗口

关键参数：

```bash
--train-stride-frames 1
```

含义：

- 训练窗口尽量密集滑动
- 让两条数据被充分训练
- 提高 overfit 可能性

### `scripts/preflight_overfit_train.py`

训练前检查脚本。

作用：

- 检查官方 checkpoint 是否是真实 PyTorch checkpoint
- 检查 train/test json 是否存在
- 检查 motion latent 是否可读
- 检查 `train.py` 是否支持 `resume_mode=weights_only`

这个脚本可以提前发现路径错误、Git LFS pointer 权重、metadata 格式错误等问题。

## 4. 纯音频过拟合训练文件

### `configs/motion_gen/overfit_2samples_selected.yaml`

selected 纯音频 overfit 配置。

作用：

- 指向 selected train/test metadata
- 使用原始纯音频 DyStream motion generation 模型
- 从官方权重开始训练
- 用于生成 audio-only overfit checkpoint

关键配置含义：

```yaml
resume_ckpt: checkpoints/last.ckpt
resume_mode: weights_only
```

含义：

- 加载官方模型参数
- 不恢复官方 trainer step
- 不恢复官方 optimizer/scheduler 状态
- overfit 训练从新的 step 开始

### `scripts/run_selected_overfit.sh`

selected 训练启动脚本。

作用：

- 同时支持纯音频配置和文本配置
- 通过 `CONFIG` 环境变量切换
- 通过 `CKPT` 指定起始权重
- 调用 `train.py`

纯音频训练命令：

```bash
cd /ICML/ZJU/DyStream-main
GPU_ID=0 \
TRAIN_BS=64 \
VAL_BS=2 \
MAX_STEPS=8000 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
EXP_NAME=overfit_2samples_selected_stride1_bs64 \
bash scripts/run_selected_overfit.sh
```

### `train.py`

训练主入口。

这里做了重要修改：

```text
resume_mode=weights_only
```

作用：

- 官方权重只作为模型初始化
- 不把官方 checkpoint 的 global step、optimizer state、scheduler state 恢复进来
- 避免“看起来 resume 了，但其实 trainer 状态来自官方训练”的问题

关键逻辑位置：

```text
train.py
resume_mode = cfg.get("resume_mode", "trainer")
```

## 5. 文本条件注入文件

### `configs/motion_gen/overfit_2samples_selected_text.yaml`

selected 文本条件 overfit 配置。

作用：

- 指向 selected train/test metadata
- 使用文本条件模型
- 打开文本条件注入
- 从官方权重 weights-only 初始化

关键配置：

```yaml
use_text_condition: true
module_name: model.motion_generation.text_conditioned_audio2face
```

文本训练命令：

```bash
cd /ICML/ZJU/DyStream-main
GPU_ID=0 \
TRAIN_BS=64 \
VAL_BS=2 \
MAX_STEPS=8000 \
CONFIG=configs/motion_gen/overfit_2samples_selected_text.yaml \
EXP_NAME=overfit_2samples_selected_text_stride1_bs64 \
bash scripts/run_selected_overfit.sh
```

### `model/motion_generation/text_conditioned_audio2face.py`

文本条件模型文件。

这是文本条件注入的核心文件。

作用：

- 保留原 DyStream audio-driven motion generation 主体
- 增加 text encoder
- 把 caption/prompt 编码成 text embedding
- 将 text embedding 投影到模型 hidden 维度
- 注入到 audio hidden / generation hidden 中
- 训练和推理时都可以接收 `caption`

主要逻辑：

```text
encode_text_condition(...)
apply_text_condition(...)
forward(..., caption=None)
inference(..., caption=None)
```

当前实现定位：

- 满足“caption 作为 prompt 参与训练和推理”
- 属于第一版文本条件注入
- 不是强控制模型，不保证精确控制表情、头动、风格

### `datasets/single_dyadic_prev_audio.py`

DyStream dataset 文件。

这里增加/确认了 batch 中携带 caption：

```python
caption=data_item.get("caption", "")
```

作用：

- 从 metadata item 读取 `caption`
- 训练时 batch 里可以拿到 prompt
- 给文本条件模型使用

### `train.py`

除了 weights-only resume，这里也接入了文本条件。

作用：

- 从 batch 中读取 `caption`
- 如果配置里 `use_text_condition: true`
- 就把 `caption` 传给模型 forward

关键逻辑：

```text
caption = batch.get("caption")
if use_text_condition:
    model_kwargs["caption"] = caption
```

### `main.py`

推理主入口。

作用：

- 推理时读取 test metadata 中的 `caption`
- 如果模型配置打开 `use_text_condition`
- 就把 caption 传给模型 inference

同时支持：

```text
prompt_override
```

也就是说可以在推理时强行覆盖 metadata 里的 caption。

### `scripts/update_overfit_captions.py`

旧版 overfit 数据的 caption 更新脚本。

作用：

- 给旧的 `overfit_train.json/overfit_test.json` 补 caption
- 主要用于旧版两条数据
- selected 数据主要走 `overfit_items_selected.json`

## 6. 推理与交付脚本

### `scripts/run_official_overfit_infer.sh`

官方权重推理脚本。

作用：

- 使用 `checkpoints/last.ckpt`
- 对 overfit test metadata 做推理
- 输出官方 baseline 视频

selected 数据推理时需要指定：

```bash
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml
EXP_NAME=official_selected_infer_package
```

### `scripts/run_finetuned_infer.sh`

纯音频过拟合 checkpoint 推理脚本。

作用：

- 指定 finetuned checkpoint
- 对 test metadata 推理
- 输出 audio overfit 视频

示例：

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
CKPT=checkpoints/package_audio_step1500.ckpt \
EXP_NAME=audio_selected_step1500_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_finetuned_infer.sh
```

### `scripts/run_text_finetuned_infer.sh`

文本条件过拟合 checkpoint 推理脚本。

作用：

- 指定文本条件 checkpoint
- 使用文本条件模型
- 推理时传 caption/prompt
- 输出 text overfit 视频

示例：

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected_text.yaml \
CKPT=checkpoints/package_text_step1500.ckpt \
EXP_NAME=text_selected_step1500_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_text_finetuned_infer.sh
```

### `scripts/collect_overfit_deliverables.sh`

旧版交付收集脚本。

作用：

- 把 GT、官方推理、纯音频 overfit、文本 overfit 的视频复制到 `deliverables/`
- 主要是早期 overfit 数据的收集脚本

selected 最终整理结果现在在：

```text
deliverables/selected_comparison_step1500/
```

## 7. 最终对比交付目录

本地目录：

```text
D:\桌面\DyStream-main\deliverables\selected_comparison_step1500
```

服务器对应目录：

```text
/ICML/ZJU/DyStream-main/deliverables/selected_comparison_step1500
```

目录结构：

```text
selected_comparison_step1500/
  01_gt/
    sample_001_gt.mp4
    sample_002_gt.mp4
  02_official_infer/
    sample_001_official.mp4
    sample_002_official.mp4
  03_audio_overfit_infer/
    sample_001_audio_overfit_step1500.mp4
    sample_002_audio_overfit_step1500.mp4
  04_text_overfit_infer/
    sample_001_text_overfit_step1500.mp4
    sample_002_text_overfit_step1500.mp4
```

含义：

- `01_gt`：裁剪后的原始 GT 视频
- `02_official_infer`：官方权重推理结果
- `03_audio_overfit_infer`：纯音频 overfit checkpoint 推理结果
- `04_text_overfit_infer`：文本条件 overfit checkpoint 推理结果

## 8. Online UI 相关文件

Online UI 是另一条链路，和 overfit/text 训练不是同一件事，但项目里也有相关改造。

### `online_streaming_demo.py`

online-style 分段推理后端 demo。

作用：

- 维护 DyStream online state
- 保存历史音频、past motion、anchor motion
- 支持 chunk-by-chunk 生成 segment
- 支持静音 gate / idle motion
- 支持调用 latent-to-video

### `online_streaming_ui.py`

Gradio 交互 UI。

作用：

- 上传/选择参考图
- recorded/uploaded audio 推理
- live microphone 分段输入
- 后台 worker 生成视频 segment
- 前端轮询最新 segment

当前定位：

- 是 online-style segmented demo
- 不是严格每几十毫秒输出一帧的真实时流式视频系统

### `scripts/run_online_ui.sh`

启动 online UI 的脚本。

命令：

```bash
cd /ICML/ZJU/DyStream-main
PORT=7860 bash scripts/run_online_ui.sh
```

### `scripts/check_online_ui_assets.py`

online UI 启动前检查脚本。

作用：

- 检查 checkpoint
- 检查 tools
- 检查 ffmpeg/ffprobe
- 避免 UI 启动后才报依赖错误

## 9. 最小复现流程

### 9.1 准备 selected 数据

```bash
cd /ICML/ZJU/DyStream-main
TRAIN_STRIDE_FRAMES=1 DEVICE=cuda BATCH_SIZE=64 bash scripts/prepare_selected_overfit_samples.sh
```

### 9.2 纯音频 overfit

```bash
GPU_ID=0 \
TRAIN_BS=64 \
VAL_BS=2 \
MAX_STEPS=8000 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
EXP_NAME=overfit_2samples_selected_stride1_bs64 \
bash scripts/run_selected_overfit.sh
```

### 9.3 文本条件 overfit

```bash
GPU_ID=0 \
TRAIN_BS=64 \
VAL_BS=2 \
MAX_STEPS=8000 \
CONFIG=configs/motion_gen/overfit_2samples_selected_text.yaml \
EXP_NAME=overfit_2samples_selected_text_stride1_bs64 \
bash scripts/run_selected_overfit.sh
```

### 9.4 官方权重 selected 推理

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
EXP_NAME=official_selected_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_official_overfit_infer.sh
```

### 9.5 纯音频 overfit 推理

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected.yaml \
CKPT=checkpoints/package_audio_step1500.ckpt \
EXP_NAME=audio_selected_step1500_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_finetuned_infer.sh
```

### 9.6 文本条件 overfit 推理

```bash
GPU_ID=0 \
CONFIG=configs/motion_gen/overfit_2samples_selected_text.yaml \
CKPT=checkpoints/package_text_step1500.ckpt \
EXP_NAME=text_selected_step1500_infer_package \
DENOISING_STEPS=10 \
bash scripts/run_text_finetuned_infer.sh
```

## 10. 注意事项

1. 如果 checkpoint 路径里有 `=`，有些 override parser 会解析失败。

例如：

```text
step_step=1500.ckpt
```

可以用不带等号的软链接：

```bash
ln -sf ../outputs/.../step_step=1500.ckpt checkpoints/package_audio_step1500.ckpt
```

2. 自动 validation 转视频会额外吃显存。

两个训练进程同时跑时，验证阶段可能因为 latent-to-video 和训练同时占用 GPU 而 OOM。  
这不一定表示训练 checkpoint 坏了。更稳妥的方式是训练保存 checkpoint 后，停掉训练，单独跑推理。

3. 文本条件版本是第一版。

它满足：

```text
caption 作为 prompt 参与训练和推理
```

但不保证：

```text
强控制表情
强控制头部动作
强控制视频风格
```

4. selected 数据比旧版两条 overfit 数据更适合当前任务。

原因：

- 人脸更清楚
- 音频、视频、motion latent 对齐更严格
- train metadata 展开了完整视频窗口

