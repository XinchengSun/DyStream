import os
import time
import queue
import argparse
import multiprocessing as mp

import numpy as np


def put_drop_old(q, item):
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def get_paths(app, sample):
    if sample == 1:
        image_path = os.path.join(app.PROJECT_ROOT, "img_files", "3.png")
    else:
        image_path = os.path.join(app.PROJECT_ROOT, "img_files", "11.png")
    return image_path


def motion_worker(args, anchor_q, audio_q, motion_q):
    """
    Original-app-aligned streaming motion worker.

    Core difference from previous versions:
    - Keep prefix silence + cumulative mic audio.
    - Use global start_idx exactly like app.py/model.inference_cuda_graph.
    - Use speaker-only mode: audio_other is always zeros_like(audio).
    - Generate only frames whose original window is available.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.motion_gpu)

    import time
    import queue
    import numpy as np
    import torch
    from contextlib import nullcontext
    import app

    device = app.DEVICE

    print(f"[MOTION_GPU{args.motion_gpu}] loading DyStream model...", flush=True)
    app.load_dystream_model()

    model = app._dystream_model
    noise_scheduler = app._noise_scheduler

    audio_sr = int(app.OmegaConf.select(app._dystream_cfg.config, "model.audio_sr", default=16000))
    pose_fps = int(app.OmegaConf.select(app._dystream_cfg.config, "model.pose_fps", default=25))

    samples_per_frame = int(audio_sr / pose_fps)
    hop_samples = int(audio_sr * args.hop_ms / 1000)

    window = int(model.cfg.cbh_window_length)
    context_frames = int(model.inpainting_length)
    prefix_samples = context_frames * samples_per_frame

    print(
        f"[MOTION_GPU{args.motion_gpu}] audio_sr={audio_sr}, pose_fps={pose_fps}, "
        f"window={window}, context={context_frames}, prefix_sec={prefix_samples / audio_sr:.3f}, "
        f"feature_lag_frames={args.feature_lag_frames}",
        flush=True,
    )

    # Render worker sends anchor motion latent first.
    anchor_np = anchor_q.get()
    anchor_motion = torch.from_numpy(anchor_np).float().to(device)
    if anchor_motion.dim() == 1:
        anchor_motion = anchor_motion.unsqueeze(0).unsqueeze(0)
    elif anchor_motion.dim() == 2:
        anchor_motion = anchor_motion.unsqueeze(0)

    past_motion = anchor_motion.repeat(1, context_frames, 1).detach()

    # Exactly like model.inference_cuda_graph initialization.
    past_audio = torch.zeros([1, 80], device=device)
    past_audio_other = torch.zeros([1, 80], device=device)

    if app._dystream_ema is not None:
        app._dystream_ema.to(device)
        ctx = app._dystream_ema.average_parameters(model.parameters())
    else:
        ctx = nullcontext()

    pending = np.zeros(0, dtype=np.float32)
    real_audio = np.zeros(0, dtype=np.float32)

    generated_idx = 0  # original global start_idx already generated
    step = 0

    with ctx, torch.inference_mode():
        if not getattr(model, "cuda_graph_enabled", False):
            model.setup_cuda_graphs(num_inference_steps=args.denoising_steps)

        print(f"[MOTION_GPU{args.motion_gpu}] ready. Waiting microphone audio...", flush=True)

        while True:
            try:
                chunk = audio_q.get(timeout=0.1)
                if chunk is None:
                    break

                chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                pending = np.concatenate([pending, chunk], axis=0)
            except queue.Empty:
                pass

            # Consume full 200ms hops from pending mic audio.
            while pending.shape[0] >= hop_samples:
                hop = pending[:hop_samples]
                pending = pending[hop_samples:]

                hop_rms = float(np.sqrt(np.mean(hop.astype(np.float32) ** 2) + 1e-12))

                # Accumulate real mic audio after prefix silence.
                real_audio = np.concatenate([real_audio, hop.astype(np.float32)], axis=0)

                # Original app does:
                # audio_self = zeros(prefix) + full_speaker_audio
                audio_full = np.concatenate(
                    [np.zeros(prefix_samples, dtype=np.float32), real_audio],
                    axis=0,
                )

                # Align to integer pose frames.
                total_len = audio_full.shape[0] // samples_per_frame
                usable_samples = total_len * samples_per_frame
                audio_full = audio_full[:usable_samples]

                # Original model can produce start_idx in [0, total_len - window].
                available_outputs = max(0, total_len - window + 1)

                # Add a small feature lag so Wav2Vec features near the right boundary are less unstable.
                target_outputs = max(0, available_outputs - int(args.feature_lag_frames))

                if target_outputs <= generated_idx:
                    print(
                        f"[MOTION_LIVE] step={step:05d} WAIT rms={hop_rms:.6f} "
                        f"generated={generated_idx} target={target_outputs} pending={pending.shape[0]}",
                        flush=True,
                    )
                    step += 1
                    continue

                audio_tensor = torch.from_numpy(audio_full).float().unsqueeze(0).to(device)
                audio_other_tensor = torch.zeros_like(audio_tensor).to(device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Compute features over cumulative audio, not a 4s rolling crop.
                audio_feat = model.get_audio2face_fea(audio_tensor, None, total_len)
                audio_other_feat = model.get_audio2face_fea_other(audio_other_tensor, None, total_len)

                out_frames = []
                produced_start = generated_idx

                while generated_idx < target_outputs:
                    start_idx = generated_idx
                    end_idx = start_idx + window

                    audio_slice_len = window * samples_per_frame
                    audio_slice_start = start_idx * samples_per_frame

                    audio_slice = audio_tensor[:, audio_slice_start:audio_slice_start + audio_slice_len]
                    audio_slice_other = audio_other_tensor[:, audio_slice_start:audio_slice_start + audio_slice_len]

                    out1 = model.one_clip_only_inference_cuda_graph(
                        per_compute_audio_feature=audio_feat[:, start_idx:end_idx],
                        per_compute_audio_other_feature=audio_other_feat[:, start_idx:end_idx],
                        past_audio_self=past_audio,
                        audio_self=audio_slice,
                        past_audio_other=past_audio_other,
                        audio_other=audio_slice_other,
                        past_motion=past_motion,
                        gen_frames=1,
                        anchor_latent=anchor_motion,
                        noise_scheduler=noise_scheduler,
                        num_inference_steps=args.denoising_steps,
                    )

                    past_motion = torch.cat([past_motion, out1.detach()], dim=1)[:, -context_frames:].detach()

                    # Match original model.inference_cuda_graph update.
                    past_audio = audio_slice[:, :-samples_per_frame].detach()
                    past_audio_other = audio_slice_other[:, :-samples_per_frame].detach()

                    out_frames.append(out1.detach())
                    generated_idx += 1

                out = torch.cat(out_frames, dim=1)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                item = {
                    "step": step,
                    "motion_np": out.detach().cpu().numpy(),
                    "motion_done_time": t1,
                    "motion_total": t1 - t0,
                    "silent": False,
                    "hop_rms": hop_rms,
                    "produced": out.shape[1],
                    "produced_start": produced_start,
                    "produced_end": generated_idx,
                }

                motion_q.put(item)

                print(
                    f"[MOTION_LIVE] step={step:05d} ORIG rms={hop_rms:.6f} "
                    f"total={t1 - t0:.4f}s produced={out.shape[1]} "
                    f"idx={produced_start}->{generated_idx} pending={pending.shape[0]}",
                    flush=True,
                )

                step += 1


def render_chunk_fp32_loop(torch, np, motion_chunk, render_src_motion, face_feat, flow_estimator, face_generator, device):
    motion_latents = motion_chunk.squeeze(0).to(device).float()
    src_motion = render_src_motion.squeeze(0).to(device).float()

    frames = []

    with torch.inference_mode():
        for i in range(motion_latents.shape[0]):
            tgt = flow_estimator(src_motion, motion_latents[i:i + 1])
            recon = face_generator(tgt, face_feat)

            video_u8 = ((recon.float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            frame_np = video_u8.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
            frames.append(frame_np)

    return np.concatenate(frames, axis=0)

def render_worker(args, anchor_q, motion_q, frame_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.render_gpu)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    from PIL import Image
    import app

    device = app.DEVICE

    print(f"[RENDER_GPU{args.render_gpu}] loading visualization...", flush=True)
    app.load_visualization_model()

    image_path = get_paths(app, args.sample)

    print(f"[RENDER_GPU{args.render_gpu}] processing reference image...", flush=True)
    image_pil = Image.open(image_path).convert("RGB")
    resized_pil, masked_pil, motion_latent_cpu = app.process_image(image_pil)

    anchor_np = motion_latent_cpu.numpy()

    transform = app._vis_ctx["transform"]
    face_encoder = app._vis_ctx["face_encoder"]
    flow_estimator = app._vis_ctx["flow_estimator"]
    face_generator = app._vis_ctx["face_generator"]

    ref_img_tensor = transform(resized_pil.convert("RGB")).unsqueeze(0).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fe0 = time.perf_counter()
    face_feat = face_encoder(ref_img_tensor).detach()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fe1 = time.perf_counter()

    print(f"[RENDER_GPU{args.render_gpu}] cached face_feat={fe1 - fe0:.4f}s", flush=True)

    warm_anchor = torch.from_numpy(anchor_np).float()
    if warm_anchor.dim() == 1:
        warm_anchor = warm_anchor.unsqueeze(0).unsqueeze(0)
    elif warm_anchor.dim() == 2:
        warm_anchor = warm_anchor.unsqueeze(0)

    warm_anchor = warm_anchor.to(device)
    warm_frames = max(1, int(round(25 * args.hop_ms / 1000)))
    warm_chunk = warm_anchor.repeat(1, warm_frames, 1)

    print(f"[RENDER_GPU{args.render_gpu}] prewarming LIA...", flush=True)
    for k in range(3):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        a = time.perf_counter()
        _ = render_chunk_fp32_loop(
            torch,
            np,
            warm_chunk,
            warm_anchor[:, 0:1, :],
            face_feat,
            flow_estimator,
            face_generator,
            device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        b = time.perf_counter()
        print(f"[RENDER_WARMUP] {k} render={b - a:.4f}s", flush=True)

    # 预热完成后再放行 motion。
    anchor_q.put(anchor_np)
    print(f"[RENDER_GPU{args.render_gpu}] ready. Waiting motion chunks...", flush=True)

    render_src_motion = None

    while True:
        try:
            item = motion_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break

        motion_np = item["motion_np"]
        motion_chunk = torch.from_numpy(motion_np).float()

        if render_src_motion is None:
            render_src_motion = motion_chunk[:, 0:1, :].detach()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        r0 = time.perf_counter()

        video_np = render_chunk_fp32_loop(
            torch,
            np,
            motion_chunk,
            render_src_motion,
            face_feat,
            flow_estimator,
            face_generator,
            device,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        r1 = time.perf_counter()

        # 把 5 帧拆开塞给前端。队列满就丢旧帧，保证显示最新。
        for f in video_np:
            put_drop_old(frame_q, f)

        print(
            f"[RENDER_LIVE] step={item['step']:05d} render={r1 - r0:.4f}s "
            f"motion_total={item['motion_total']:.4f}s "
            f"frames={item.get('produced', -1)} "
            f"idx={item.get('produced_start', -1)}->{item.get('produced_end', -1)} "
            f"rms={item.get('hop_rms', -1):.6f}",
            flush=True
        )

    print(f"[RENDER_GPU{args.render_gpu}] stopped.", flush=True)


AUDIO_Q = None
FRAME_Q = None


def normalize_audio_for_queue(audio, state):
    """
    Gradio mic input: usually (sr, np.ndarray).
    兼容两种情况：
    1. streaming chunk：每次只给新片段
    2. cumulative audio：每次给从开始到现在的全量音频
    """
    if audio is None:
        return None, state, "no audio"

    sr, arr = audio
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    # int16 / int32 转 float32 [-1, 1]
    if np.issubdtype(arr.dtype, np.integer):
        maxv = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / maxv
    else:
        arr = arr.astype(np.float32)

    # 防止过大
    arr = np.clip(arr, -1.0, 1.0)

    last_len = int(state.get("last_len", 0))
    last_sr = int(state.get("last_sr", sr))

    # 如果像累计音频，就只取新增部分；否则认为它本身就是 chunk。
    if sr == last_sr and arr.shape[0] > last_len and (arr.shape[0] - last_len) < int(sr * 2.0):
        chunk = arr[last_len:]
        new_last_len = arr.shape[0]
    else:
        chunk = arr
        new_last_len = 0

    state["last_len"] = new_last_len
    state["last_sr"] = int(sr)

    if chunk.size == 0:
        return None, state, "empty chunk"

    # 重采样到 16k。用 numpy 插值，避免主进程依赖 librosa。
    target_sr = 16000
    if sr != target_sr:
        old_x = np.linspace(0.0, 1.0, num=chunk.shape[0], endpoint=False)
        new_len = max(1, int(round(chunk.shape[0] * target_sr / sr)))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        chunk = np.interp(new_x, old_x, chunk).astype(np.float32)

    return chunk.astype(np.float32), state, f"mic chunk={chunk.shape[0]} samples @16k"


def ui_main(args, audio_q, frame_q):
    import gradio as gr

    global AUDIO_Q, FRAME_Q
    AUDIO_Q = audio_q
    FRAME_Q = frame_q

    last_frame = {"img": np.zeros((512, 512, 3), dtype=np.uint8)}
    playback_state = {
        "started": False,
        "shown": 0,
    }

    def push_mic(audio, state):
        if state is None:
            state = {"last_len": 0, "last_sr": 0}

        chunk, state, msg = normalize_audio_for_queue(audio, state)

        if audio is not None:
            raw_sr, raw_arr = audio
            raw_arr = np.asarray(raw_arr)
            raw_len = raw_arr.shape[0]
            raw_dtype = str(raw_arr.dtype)
        else:
            raw_sr, raw_len, raw_dtype = -1, -1, "none"

        if chunk is not None:
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-12))
            peak = float(np.max(np.abs(chunk)) + 1e-12)
            print(
                f"[MIC_DEBUG] raw_sr={raw_sr} raw_len={raw_len} raw_dtype={raw_dtype} | "
                f"{msg} rms={rms:.6f} peak={peak:.6f} "
                f"state_last_len={state.get('last_len', -1)}",
                flush=True,
            )
            put_drop_old(AUDIO_Q, chunk)
        else:
            print(
                f"[MIC_DEBUG] raw_sr={raw_sr} raw_len={raw_len} raw_dtype={raw_dtype} | {msg}",
                flush=True,
            )

        return msg, state

    def pull_frame():
        # FIFO 播放，不清空队列。
        # 先攒够 ui_start_buffer_frames，再开始按 Timer 播放，避免 render burst 导致口型节奏飘。
        qsize = FRAME_Q.qsize()

        if not playback_state["started"]:
            if qsize >= int(args.ui_start_buffer_frames):
                playback_state["started"] = True
                print(
                    f"[UI_PLAY] start playback buffer={qsize} frames",
                    flush=True,
                )
            else:
                return last_frame["img"]

        try:
            img = FRAME_Q.get_nowait()
            last_frame["img"] = img
            playback_state["shown"] += 1
        except queue.Empty:
            pass

        return last_frame["img"]

    with gr.Blocks(title="DyStream Dual-GPU Mic Realtime") as demo:
        gr.Markdown(
            """
# DyStream 双卡实时麦克风 Demo

浏览器麦克风输入 → GPU0 生成 motion → GPU1 渲染头像 → 页面实时显示最新帧。

当前版本：
- 200ms hop
- 每次 5 帧
- motion stride=1
- renderer FP32 原始逐帧 LIA
- 不保存 mp4
"""
        )

        with gr.Row():
            try:
                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="Microphone realtime input",
                )
            except TypeError:
                mic = gr.Audio(
                    source="microphone",
                    type="numpy",
                    streaming=True,
                    label="Microphone realtime input",
                )

            out_img = gr.Image(
                label="Realtime avatar frame",
                type="numpy",
                height=512,
                width=512,
            )

        status = gr.Textbox(label="Mic status", value="waiting microphone...")
        st = gr.State({"last_len": 0, "last_sr": 0})

        mic.stream(
            fn=push_mic,
            inputs=[mic, st],
            outputs=[status, st],
        )

        timer = gr.Timer(0.04)
        timer.tick(
            fn=pull_frame,
            inputs=None,
            outputs=out_img,
        )

    demo.queue(max_size=16)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        show_error=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1, choices=[1, 2])
    parser.add_argument("--hop_ms", type=int, default=200)
    parser.add_argument("--denoising_steps", type=int, default=1)
    parser.add_argument("--motion_gpu", type=int, default=0)
    parser.add_argument("--render_gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--silence_threshold", type=float, default=0.005)
    parser.add_argument("--feature_lag_frames", type=int, default=25)
    parser.add_argument("--ui_start_buffer_frames", type=int, default=25)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")

    anchor_q = ctx.Queue(maxsize=1)
    audio_q = ctx.Queue(maxsize=16)
    motion_q = ctx.Queue(maxsize=4)
    frame_q = ctx.Queue(maxsize=30)

    p_render = ctx.Process(target=render_worker, args=(args, anchor_q, motion_q, frame_q))
    p_motion = ctx.Process(target=motion_worker, args=(args, anchor_q, audio_q, motion_q))

    p_render.start()
    p_motion.start()

    try:
        ui_main(args, audio_q, frame_q)
    finally:
        try:
            put_drop_old(audio_q, None)
            put_drop_old(motion_q, None)
        except Exception:
            pass

        p_motion.terminate()
        p_render.terminate()

        p_motion.join(timeout=3)
        p_render.join(timeout=3)


if __name__ == "__main__":
    main()
