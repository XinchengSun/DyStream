# DyStream 考核任务当前状态总结

更新时间：2026-05-29  
项目目录：`/ICML/ZJU/DyStream-main`  
本地目录：`D:\桌面\DyStream-main`

> 2026-06-01 online UI 继续开发记录见：
>
> ```text
> D:\桌面\DyStream-main\docs\online_ui_dev_log_20260601.md
> /ICML/ZJU/DyStream-main/docs/online_ui_dev_log_20260601.md
> ```
>
> 当前 CPU 阶段已完成：`latent_to_video.py` 拆出 `WrappingDecoderSession`、`online_streaming_demo.py` 接入常驻 decoder、增加 `DYSTREAM_ONLINE_DECODER=mock`、增加 CPU smoke、`online_streaming_ui.py` 增加 mock session 并验证 start/recorded callback 不丢 reference preview。真实画质和实时口型仍需 GPU + ffmpeg 环境继续验证。

## 1. 原始任务要求

### 1.1 读论文和代码

需要理解：

- DyStream 论文：`https://arxiv.org/abs/2512.24408`
- SoulX-FlashHead 论文：`https://arxiv.org/pdf/2602.07449`
- DyStream 代码：`https://github.com/RobinWitch/DyStream`
- SoulX-FlashHead 代码：`https://github.com/Soul-AILab/SoulX-FlashHead`

要求是在 DyStream 基础上开发，只学习 SoulX-FlashHead 里的 online S2V 方案。

### 1.2 跑通离线推理

需要在 DyStream 上跑通官方权重推理，确认官方 baseline 可以输出视频。

### 1.3 两条数据过拟合实验

要求：

- 自行组织 2 条数据。
- 使用 `train.py` 训练。
- 训推一致。
- resume 原始官方权重。
- 做过拟合。

需要交付：

- 原始 GT 音视频。
- 官方权重推理结果。
- 过拟合训练后推理出来的音视频。

### 1.4 Online 流式推理 UI

要求：

- 外界实时不断给出音频，不知道什么时候结束。
- 后端需要实时把音频转成视频。
- 需要交付一个交互 UI。
- 用户可以按住说话。
- 屏幕上可以实时转录出视频。

### 1.5 文本控制条件注入

要求：

- 当前项目是纯音频驱动视频生成，无法控制视觉画面。
- 需要注入文本视频控制条件。
- 训练时 caption 作为 prompt。
- 完成文本条件版本的过拟合实验。

## 2. 当前已经完成的内容

### 2.1 数据准备

已经从原始双人视频中截取并整理了 2 条 overfit 数据。

当前数据目录：

```text
/ICML/ZJU/DyStream-main/data_overfit/sample_001/
/ICML/ZJU/DyStream-main/data_overfit/sample_002/
```

每条数据包含：

```text
gt.mp4
audio.wav
ref.png
ref_resize.png
motion.npz
```

metadata：

```text
/ICML/ZJU/DyStream-main/data_json/overfit_train.json
/ICML/ZJU/DyStream-main/data_json/overfit_test.json
```

### 2.2 官方权重离线推理

已经跑通官方权重推理。

使用的官方权重：

```text
/ICML/ZJU/DyStream-main/checkpoints/last.ckpt
```

注意：之前误下过 Git LFS pointer 文件，后面已经重新下载真实权重并替换。真实权重开头是 zip/pytorch checkpoint，不是 `version https://git-lfs...`。

官方推理结果目录：

```text
/ICML/ZJU/DyStream-main/deliverables/official_infer/
```

### 2.3 纯音频两条数据过拟合

已经完成纯音频 overfit 训练。

训练是从官方权重 `checkpoints/last.ckpt` 加载模型参数开始的，不恢复官方 trainer step。

推理结果目录：

```text
/ICML/ZJU/DyStream-main/deliverables/audio_overfit_infer/
```

### 2.4 文本条件注入

已经做了第一版文本条件注入。

实现方向：

- 数据集读取 `caption`。
- batch 中携带 caption。
- 新增文本条件模型文件。
- 使用文本 encoder 提取 prompt embedding。
- 将文本条件注入到 motion generation。
- 新增文本条件配置：

```text
/ICML/ZJU/DyStream-main/configs/motion_gen/overfit_2samples_text.yaml
```

文本条件模型文件：

```text
/ICML/ZJU/DyStream-main/model/motion_generation/text_conditioned_audio2face.py
```

文本 overfit 是从官方权重开始，不是从纯音频 overfit 权重继续训练。

文本条件推理结果目录：

```text
/ICML/ZJU/DyStream-main/deliverables/text_overfit_infer/
```

### 2.5 当前交付视频目录

当前应该有四组核心视频：

```text
/ICML/ZJU/DyStream-main/deliverables/gt/
/ICML/ZJU/DyStream-main/deliverables/official_infer/
/ICML/ZJU/DyStream-main/deliverables/audio_overfit_infer/
/ICML/ZJU/DyStream-main/deliverables/text_overfit_infer/
```

打包文件曾生成在：

```text
/ICML/ZJU/DyStream-main/deliverables/packages/
```

## 3. Online UI 当前真实状态

### 3.1 已经做过的 online 相关代码

新增/修改过的主要文件：

```text
/ICML/ZJU/DyStream-main/online_streaming_demo.py
/ICML/ZJU/DyStream-main/online_streaming_ui.py
/ICML/ZJU/DyStream-main/scripts/run_online_ui.sh
/ICML/ZJU/DyStream-main/scripts/check_online_ui_assets.py
```

UI 启动命令：

```bash
cd /ICML/ZJU/DyStream-main
PORT=7860 bash scripts/run_online_ui.sh
```

本地通过 SSH tunnel 打开：

```text
http://localhost:7860
```

### 3.2 已经解决过的问题

之前遇到过这些问题，并已做过修复：

- Gradio 找不到 `ffmpeg`。
- Gradio 找不到 `ffprobe`。
- Gradio `Video` 组件 postprocess 报错。
- Gradio analytics 触发坏 pandas 报错。
- 上传参考图后，motion anchor 仍来自 sample，导致参考图和 motion latent 不匹配。
- 静音时嘴巴仍然乱动。
- 长音频只显示一个约 13 秒 segment，没有自动拼完整 final.mp4。

对应处理：

- 将 `imageio_ffmpeg` 的 ffmpeg 链接到项目 `.bin/ffmpeg`。
- 绕过 Gradio 的 `video_is_playable` 检查。
- 禁用 Gradio analytics。
- 上传参考图时，先用 DyStream motion encoder 编码参考图 motion latent，作为 anchor/past motion。
- 加入 `Silence gate`，低音量时不走扩散生成。
- 加入 `Idle motion`，静音时不是完全冻结，而是叠加很弱的 idle motion。
- 长音频分段生成后自动 concat 成 `final.mp4`。

### 3.3 当前 UI 的严重问题

当前 online UI 还没有达到原始要求中的“按住说话，屏幕实时转录出视频”的标准。

当前真实情况：

- `Recorded/uploaded audio` tab 可以录完或上传一段音频，然后生成视频。
- `Live microphone` tab 做了初步接入，但体验不合格。
- 用户上传图片后，页面右侧主要显示的是静态图片。
- live 模式下不是稳定地实时显示连续人像视频。
- 即使后端生成了 segment，前端体验仍然不像真正实时视频流。
- 当前不是严格实时，而是在线分段生成。

用户反馈的当前问题：

```text
前端把图片上传后，就是个图片显示，没有任何面部动作，纯图片。
```

这个反馈是成立的。当前 UI 没有达到用户期望。

### 3.4 为什么当前 live 还不是真实时

DyStream 当前配置里：

```text
cbh_window_length = 96
inpainting_length = 94
pose_fps = 25
```

也就是说，模型推理需要较长上下文窗口。当前 live 设计至少需要攒接近一个 chunk 才能生成 segment。

现在 live 更接近：

```text
麦克风输入 -> 累积一段音频 -> 生成一个视频 segment -> 前端更新
```

而不是：

```text
麦克风输入 -> 每几十毫秒实时输出连续视频帧
```

因此，当前 online UI 只能算 online-style 分段推理 demo，不能算严格实时交付。

## 4. 当前代码做过的 online 设计尝试

### 4.1 SoulX-FlashHead 参考点

参考过 SoulX-FlashHead 的 online S2V 逻辑，主要思路包括：

- 音频按 chunk 输入。
- 使用历史音频缓存。
- chunk-by-chunk 推理。
- 结果通过队列或 yield 分段返回前端。
- 多个 chunk 合成一个视频片段。

### 4.2 DyStream 当前 online 适配思路

当前实现思路：

- 用 `DyStreamOnlineState` 保存模型状态。
- 保存：

```text
state.past_motion
state.anchor_motion
state.audio_history
state.idle_motion
```

- 每个 chunk 生成时：

```text
历史音频 prefix + 当前音频 segment -> DyStream inference -> motion latent -> latent_to_video -> mp4
```

- 静音段：

```text
RMS < Silence gate -> 不调用 DyStream inference -> 使用 idle motion
```

- 上传参考图：

```text
ref image -> motion encoder -> ref motion latent -> anchor_motion/past_motion
```

## 5. 现在还没有完成的内容

### 5.1 真正的 online 实时 UI 没完成

原始要求：

```text
用户可以按住说话，屏幕上可以实时转录出视频
```

当前没有达标。

缺失点：

- 没有稳定的实时麦克风到视频连续播放链路。
- 前端没有自然连续播放 segment。
- 后端 decoder 没有常驻缓存。
- 每个 segment 仍然调用 `latent_to_video.py` 子进程，延迟较高。
- 还没有做 producer-consumer 异步队列。
- 还没有把生成的视频帧流式推到前端。

### 5.2 面部自然度仍然一般

当前生成自然度不稳定，尤其：

- 嘴形可能不自然。
- 上传不匹配参考图时容易扭曲。
- 静音 idle motion 只能弱模拟，不是真正学到的自然 idle。
- 官方权重 online 效果弱于离线完整推理。

### 5.3 文本控制只是第一版

文本条件注入已经实现第一版，但只是满足：

```text
caption 作为 prompt 参与训练和推理
```

还不是强控制模型。

例如下面这些强控制还没有保证：

- 指定表情。
- 指定头部动作。
- 指定风格。
- 指定嘴型表现。

## 6. 下一步应该怎么做

### 6.1 优先修 online UI

下一步应该集中修 online UI，而不是继续堆零碎功能。

目标：

```text
用户进入页面 -> 上传/选择参考图 -> 按住说话 -> 右侧持续出现人像视频片段 -> 停止后得到完整 MP4
```

具体应该做：

1. 保留模型常驻，不要每次 segment 都重新初始化 decoder。
2. 不再每个 segment 调 `latent_to_video.py` 子进程。
3. 把 wrapping decoder 改成 Python 内部常驻对象。
4. 建立音频输入队列：

```text
mic stream -> audio queue -> model worker -> video segment queue -> frontend
```

5. 前端连续播放 segment，不只是显示单个文件。
6. Start session 后，右侧应该显示一个初始 talking-head canvas/video placeholder，而不是只显示静态图片。
7. 生成第一个 segment 前，需要明确显示：

```text
模型已加载，正在缓存音频
```

8. 每生成一个 segment，需要自动 append 到播放列表。
9. Finalize 后输出完整 final.mp4。

### 6.2 如果继续走当前方案，最低限度要补

如果不重构 decoder，至少要补：

- Start session 后先生成一个 1 秒 idle mp4，而不是只显示图片。
- 麦克风 stream 每次触发时，在日志和 UI 显示已收到音频长度。
- segment 生成后，前端必须立即显示 mp4。
- 如果音频不足一个 chunk，要在 UI 写清楚还在 buffer。
- 如果长时间没生成，要明确显示原因。

### 6.3 更合理的验收方式

应该准备三个 online demo：

1. `recorded/uploaded`：录完生成完整 MP4，用于稳定展示质量。
2. `live segmented`：边说边每隔几秒出一个视频 segment。
3. `offline baseline`：同一段音频用离线推理生成，对比 online 质量损失。

## 7. 当前结论

目前已经完成：

- 论文/代码基础理解。
- 官方权重离线推理。
- 两条数据准备。
- 纯音频 overfit。
- 文本条件注入第一版。
- 文本条件 overfit。
- deliverables 四组视频。
- online 分段推理的部分后端逻辑。

目前没有完成到位：

- 真正符合要求的 online 实时交互 UI。
- 用户按住说话时屏幕稳定实时出现人像视频。
- 高自然度 live talking head。

当前最严重的问题：

```text
Live UI 用户上传图片后主要显示静态图片，不能稳定实时显示面部动作视频。
```

这部分需要继续重做/重构，不能算完全交付。

## 8. 服务器关键路径和目录

### 8.1 服务器连接信息

最后一次使用的 GPU 服务器连接信息：

```bash
ssh -p 42235 root@xj-member.bitahub.com
```

说明：

- 这是当时使用的 A100 80GB 实例。
- 如果实例已经关机或端口变了，需要以新开的服务器 SSH 信息为准。
- 之前通过本地端口转发访问 Gradio：

```bash
ssh -p 42235 -N -L 7860:127.0.0.1:7860 root@xj-member.bitahub.com
```

本地浏览器访问：

```text
http://localhost:7860
```

### 8.2 服务器主目录

核心工作目录：

```text
/ICML/ZJU/DyStream-main
```

SoulX-FlashHead 参考代码目录：

```text
/ICML/ZJU/SoulX-FlashHead
```

DyStream assets 曾经的下载/备份目录：

```text
/ICML/ZJU/DyStream-assets
/ICML/ZJU/DyStream-real-assets
```

### 8.3 权重路径

DyStream 官方主权重：

```text
/ICML/ZJU/DyStream-main/checkpoints/last.ckpt
```

wrapping/decoder 相关权重：

```text
/ICML/ZJU/DyStream-main/tools/pretrained_model/epoch=0-step=312000.ckpt
```

MediaPipe face landmarker：

```text
/ICML/ZJU/DyStream-main/tools/visualization_0416/utils/face_landmarker.task
```

注意：

- `checkpoints/last.ckpt` 必须是真实 7GB 左右权重。
- 如果文件开头是 `version https://git-lfs.github.com/spec/v1`，那就是错误的 Git LFS pointer。

### 8.4 数据路径

两条 overfit 数据：

```text
/ICML/ZJU/DyStream-main/data_overfit/sample_001/
/ICML/ZJU/DyStream-main/data_overfit/sample_002/
```

关键文件：

```text
gt.mp4
audio.wav
ref.png
ref_resize.png
motion.npz
```

训练/测试 metadata：

```text
/ICML/ZJU/DyStream-main/data_json/overfit_train.json
/ICML/ZJU/DyStream-main/data_json/overfit_test.json
```

### 8.5 交付结果路径

四组核心交付视频：

```text
/ICML/ZJU/DyStream-main/deliverables/gt/
/ICML/ZJU/DyStream-main/deliverables/official_infer/
/ICML/ZJU/DyStream-main/deliverables/audio_overfit_infer/
/ICML/ZJU/DyStream-main/deliverables/text_overfit_infer/
```

打包目录：

```text
/ICML/ZJU/DyStream-main/deliverables/packages/
```

online smoke / online UI 输出通常在：

```text
/ICML/ZJU/DyStream-main/outputs/online_stream_*/
/ICML/ZJU/DyStream-main/outputs/online_ui_*/
```

### 8.6 日志路径

主要日志目录：

```text
/ICML/ZJU/DyStream-main/logs/
```

Online UI 日志：

```text
/ICML/ZJU/DyStream-main/logs/online_ui_7860.log
/ICML/ZJU/DyStream-main/logs/online_ui_7860.pid
```

训练/推理日志常见文件：

```text
/ICML/ZJU/DyStream-main/logs/03_official_infer.log
/ICML/ZJU/DyStream-main/logs/07_text_overfit_3000.log
/ICML/ZJU/DyStream-main/logs/text_overfit_2samples_3000.log
```

实际文件名可能因多次运行而不同，需要 `ls -lh logs` 查看。

## 9. 关键代码文件和它们负责什么

### 9.1 训练入口

```text
/ICML/ZJU/DyStream-main/train.py
```

用途：

- 纯音频 overfit 训练。
- 文本条件 overfit 训练。

### 9.2 离线推理入口

```text
/ICML/ZJU/DyStream-main/main.py
```

用途：

- 官方权重离线推理。
- overfit 权重推理。

### 9.3 数据集

```text
/ICML/ZJU/DyStream-main/datasets/single_dyadic_prev_audio.py
```

用途：

- 读取音频、motion latent、caption。
- 文本条件注入时这里要保证 batch 中有 caption。

### 9.4 原始音频驱动模型

```text
/ICML/ZJU/DyStream-main/model/motion_generation/realtime_audio2face_cuda_graph.py
```

用途：

- DyStream 原始纯音频 motion generation。
- online demo 现在主要调用这个模型的 `inference`。

### 9.5 文本条件模型

```text
/ICML/ZJU/DyStream-main/model/motion_generation/text_conditioned_audio2face.py
```

用途：

- 第一版文本条件注入。
- caption/prompt embedding 注入 motion generation。

### 9.6 online 后端 demo

```text
/ICML/ZJU/DyStream-main/online_streaming_demo.py
```

用途：

- 加载 DyStream 模型。
- 加载/编码参考图 motion latent。
- 分段处理音频。
- 调用 DyStream inference 生成 motion latent。
- 调用 wrapping/decoder 生成 mp4。
- 拼接 final.mp4。

当前已经加过：

- `Silence gate`
- `Idle motion`
- 上传参考图 motion encoder
- 音频历史 prefix
- 分段生成
- final.mp4 concat

### 9.7 online Gradio UI

```text
/ICML/ZJU/DyStream-main/online_streaming_ui.py
```

用途：

- Gradio 前端。
- 包含 `Recorded/uploaded audio` tab。
- 包含尝试做的 `Live microphone` tab。

当前主要问题就在这个文件及其调用链。

### 9.8 wrapping/decoder

```text
/ICML/ZJU/DyStream-main/tools/visualization_0416/latent_to_video.py
```

用途：

- 把 motion latent + ref image + audio 转成最终 mp4。

当前 online 最大性能瓶颈之一：

```text
每个 segment 都通过 subprocess 调一次 latent_to_video.py
```

这会导致 live 延迟很高，也不适合真正流式。

## 10. 当前任务卡在哪里

### 10.1 核心卡点

当前任务主要卡在：

```text
Online 流式推理 UI 没有达到“按住说话，屏幕实时转录出视频”的验收标准。
```

更具体地说：

- 现在可以做录完/上传音频后的分段生成。
- 现在可以生成 mp4。
- 现在可以处理静音 gate 和 idle motion。
- 但是 live tab 体验不合格。
- 用户上传图片后，页面看起来主要是静态图片。
- 不是一个稳定的实时 talking-head 视频流。

### 10.2 为什么会卡

当前 online 链路是：

```text
麦克风/音频 -> 累积 chunk -> DyStream inference -> motion latent -> subprocess 调 latent_to_video.py -> mp4 -> Gradio Video 显示
```

问题：

1. `latent_to_video.py` 每段都重新初始化 decoder，很慢。
2. Gradio `Video` 更适合展示文件，不适合连续低延迟视频流。
3. 当前没有前端播放队列，不会自然 append segment。
4. 当前没有真正的 producer-consumer 异步队列。
5. 当前没有 decoder 常驻对象。
6. DyStream 模型需要较长上下文窗口，低延迟 chunk 很难直接稳定。

### 10.3 当前不是哪个部分卡

已经确认不是这些问题：

- 不是官方权重不能加载。
- 不是 overfit 数据不存在。
- 不是 `ffmpeg/ffprobe` 展示问题。
- 不是完全不能生成 mp4。
- 不是静音逻辑没加。

真正问题是：

```text
online UI 架构还停留在“分段生成文件”，没有做成“实时连续视频流”。
```

## 11. 如果继续改，应该改哪里

### 11.1 最应该优先改的文件

优先改：

```text
/ICML/ZJU/DyStream-main/online_streaming_ui.py
/ICML/ZJU/DyStream-main/online_streaming_demo.py
```

这两个文件负责当前 online UI 和 online 推理封装。

### 11.2 第二优先级

需要拆掉 subprocess decoder 时，改：

```text
/ICML/ZJU/DyStream-main/tools/visualization_0416/latent_to_video.py
```

目标：

- 不要作为命令行子进程反复启动。
- 把里面模型初始化和单段 decode 封装成 Python class/function。
- online session start 时初始化一次 decoder。
- 每个 segment 只调用内存中的 decode 函数。

### 11.3 如果要继续优化模型推理

可能涉及：

```text
/ICML/ZJU/DyStream-main/model/motion_generation/realtime_audio2face_cuda_graph.py
```

但不建议一上来改模型主体。优先应该改 UI/decoder pipeline。

### 11.4 如果要继续优化文本条件

涉及：

```text
/ICML/ZJU/DyStream-main/datasets/single_dyadic_prev_audio.py
/ICML/ZJU/DyStream-main/model/motion_generation/text_conditioned_audio2face.py
/ICML/ZJU/DyStream-main/configs/motion_gen/overfit_2samples_text.yaml
```

当前文本条件不是最紧急卡点。

## 12. 继续开发 online UI 的建议路线

### 12.1 最低限度修法

如果时间很紧，最低限度应该做：

1. Start session 后，马上生成一个短 idle mp4，而不是只显示图片。
2. Live 页面显示 buffer 状态：

```text
received audio: X seconds
waiting for first chunk: Y / required Z samples
generating segment: N
```

3. 每生成一个 segment 后，页面必须自动更新视频。
4. 前端显示 segment 列表。
5. Finalize 后生成完整 final.mp4。

这个方案仍然不是真正实时，但至少用户不会觉得“没有人像动起来”。

### 12.2 正确重构方向

真正应该做：

```text
Gradio microphone stream
    -> audio queue
    -> model worker
    -> decoder worker
    -> video segment queue
    -> frontend append/play
```

后端需要至少两个常驻对象：

```text
DyStreamOnlineSession
WrappingDecoderSession
```

`WrappingDecoderSession` 应该从 `latent_to_video.py` 里拆出来，避免每个 segment 重新初始化模型。

### 12.3 验证标准

online UI 改完后，至少要验证：

1. Start session 后 5 秒内页面有初始人像或 idle 视频。
2. 录音时日志持续显示收到音频。
3. 达到一个 chunk 后，生成第一个 segment。
4. 右侧播放器能自动显示 segment。
5. 继续说话能生成第二、第三个 segment。
6. Finalize 后有完整 `final.mp4`。
7. 静音时嘴不乱动。
8. 长音频不会只显示第一个 13 秒。

## 13. 常用命令

### 13.1 启动 online UI

```bash
cd /ICML/ZJU/DyStream-main
PORT=7860 bash scripts/run_online_ui.sh
```

后台启动：

```bash
cd /ICML/ZJU/DyStream-main
PORT=7860 nohup bash scripts/run_online_ui.sh > logs/online_ui_7860.log 2>&1 &
echo $! > logs/online_ui_7860.pid
```

### 13.2 查看 online UI 日志

```bash
cd /ICML/ZJU/DyStream-main
tail -f logs/online_ui_7860.log
```

### 13.3 重启 online UI

```bash
cd /ICML/ZJU/DyStream-main
pkill -f "python.*online_streaming_ui.py" 2>/dev/null || true
sleep 2
PORT=7860 nohup bash scripts/run_online_ui.sh > logs/online_ui_7860.log 2>&1 &
echo $! > logs/online_ui_7860.pid
```

### 13.4 检查端口

```bash
ss -ltnp | grep 7860 || true
```

### 13.5 检查 GPU

```bash
nvidia-smi
```

### 13.6 查看最近生成的视频

```bash
cd /ICML/ZJU/DyStream-main
find outputs -type f -name "*.mp4" -printf "%TY-%Tm-%Td %TH:%TM:%TS %p %s bytes\n" | sort | tail -n 30
```

### 13.7 检查关键资产

```bash
cd /ICML/ZJU/DyStream-main
python scripts/check_online_ui_assets.py
```

### 13.8 语法检查

```bash
cd /ICML/ZJU/DyStream-main
python -m py_compile online_streaming_demo.py online_streaming_ui.py
```

## 14. 交接结论

如果后续只需要补考核，最应该继续投入的是：

```text
online_streaming_ui.py
online_streaming_demo.py
tools/visualization_0416/latent_to_video.py
```

当前最重要的未完成事项：

```text
把 online UI 从“生成 mp4 文件后展示”重构为“麦克风流式输入、后端异步生成、前端连续播放 segment”的交互系统。
```

当前已经完成的训练/离线推理/过拟合/文本条件部分可以作为已有交付保留，不应再轻易动。
