import os
import time
import queue
import argparse
import multiprocessing as mp

import numpy as np
import gradio as gr
import soundfile as sf
import librosa

import app
import dual_gpu_mic_realtime_ui_v8_playbuffer as base


def audio_to_16k_float(audio):
    """
    Gradio streaming audio normally returns (sr, np.ndarray).
    Convert to mono float32 16k.
    """
    if audio is None:
        return None, "no audio"

    sr, arr = audio
    arr = np.asarray(arr)

    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    if arr.dtype == np.int16:
        x = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        x = arr.astype(np.float32) / 2147483648.0
    else:
        x = arr.astype(np.float32)
        if np.max(np.abs(x)) > 2.0:
            x = x / 32768.0

    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)

    x = np.asarray(x, dtype=np.float32)
    return x, f"raw_sr={sr}, raw_len={len(arr)}, chunk_16k={len(x)}"


def drain_frame_queue(frame_q, frame_list):
    got = 0
    while True:
        try:
            item = frame_q.get_nowait()
        except queue.Empty:
            break

        if isinstance(item, dict):
            frame = item.get("frame")
        else:
            frame = item

        if frame is not None:
            frame_list.append(np.asarray(frame, dtype=np.uint8).copy())
            got += 1

    return got


def put_silence(audio_q, seconds, chunk_samples=8000):
    total = int(seconds * 16000)
    sent = 0
    while sent < total:
        n = min(chunk_samples, total - sent)
        audio_q.put(np.zeros(n, dtype=np.float32))
        sent += n


def main():
    print("[LAG1_LOWLATENCY] feature_lag_frames=1, hop_ms=200")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--hop_ms", type=int, default=200)
    parser.add_argument("--denoising_steps", type=int, default=1)
    parser.add_argument("--motion_gpu", type=int, default=0)
    parser.add_argument("--render_gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--feature_lag_frames", type=int, default=1)
    parser.add_argument("--ui_start_buffer_frames", type=int, default=0)
    parser.add_argument("--flush_silence_sec", type=float, default=1.5)
    parser.add_argument("--save_timeout_sec", type=float, default=12.0)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")

    anchor_q = ctx.Queue(maxsize=8)
    audio_q = ctx.Queue(maxsize=4096)
    motion_q = ctx.Queue(maxsize=4096)
    frame_q = ctx.Queue(maxsize=50000)

    render_p = ctx.Process(
        target=base.render_worker,
        args=(args, anchor_q, motion_q, frame_q),
        daemon=True,
    )
    motion_p = ctx.Process(
        target=base.motion_worker,
        args=(args, anchor_q, audio_q, motion_q),
        daemon=True,
    )

    render_p.start()
    motion_p.start()

    record = {
        "audio_chunks": [],
        "frames": [],
        "first_voice_wall": None,
        "first_audio_wall": None,
        "last_audio_wall": None,
        "save_done": None,
    }

    def push_mic(audio):
        chunk, msg = audio_to_16k_float(audio)

        if chunk is None or len(chunk) == 0:
            return "waiting mic..."

        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-12))
        now = time.time()

        if record["first_audio_wall"] is None:
            record["first_audio_wall"] = now

        # 用较小阈值记录“开始说话”的时间。静音底噪一般 0.0002~0.001。
        if record["first_voice_wall"] is None and rms > 0.01:
            record["first_voice_wall"] = now

        record["last_audio_wall"] = now

        record["audio_chunks"].append(chunk.copy())
        audio_q.put(chunk.copy())

        # 不显示视频，但顺手 drain，避免 frame_q 太大。
        drain_frame_queue(frame_q, record["frames"])

        return (
            f"{msg} | rms={rms:.6f} | "
            f"audio_sec={sum(len(c) for c in record['audio_chunks']) / 16000:.2f}s | "
            f"frames={len(record['frames'])}"
        )

    def save_mp4():
        if len(record["audio_chunks"]) == 0:
            return "no audio recorded yet", None

        save_click_wall = time.time()

        # 说完后主动补静音，让 feature_lag 后面的帧能 flush 出来。
        put_silence(audio_q, args.flush_silence_sec, chunk_samples=8000)

        audio = np.concatenate(record["audio_chunks"], axis=0).astype(np.float32)
        expected_frames = int(len(audio) / 16000 * 25)

        print(
            f"[BENCH_SAVE] expected_frames={expected_frames}, "
            f"current_frames={len(record['frames'])}, flush_silence={args.flush_silence_sec}s",
            flush=True,
        )

        deadline = time.time() + args.save_timeout_sec
        last_count = -1
        stable_rounds = 0

        while time.time() < deadline:
            drain_frame_queue(frame_q, record["frames"])

            if len(record["frames"]) >= expected_frames:
                break

            if len(record["frames"]) == last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0
                last_count = len(record["frames"])

            time.sleep(0.05)

        render_done_wall = time.time()

        if len(record["frames"]) == 0:
            return "no frames rendered", None

        frames = np.stack(record["frames"], axis=0).astype(np.uint8)

        # 只保存与原始录音时长对应的帧，不把 flush 静音也保存进去。
        if expected_frames > 0 and frames.shape[0] > expected_frames:
            frames = frames[:expected_frames]

        audio_len = int(frames.shape[0] / 25 * 16000)
        audio_out = audio[:audio_len]
        if len(audio_out) < audio_len:
            audio_out = np.pad(audio_out, (0, audio_len - len(audio_out)))

        out_dir = os.path.join(os.getcwd(), "stream_outputs", "mic_benchmark")
        os.makedirs(out_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(out_dir, f"mic_bench_{ts}.wav")
        mp4_path = os.path.join(out_dir, f"mic_bench_{ts}.mp4")

        sf.write(wav_path, audio_out, 16000)

        mux_start = time.time()
        app.save_video_with_audio(frames, wav_path, mp4_path, fps=25)
        mux_done = time.time()

        first_ref = record["first_voice_wall"] or record["first_audio_wall"] or save_click_wall
        audio_duration = len(audio) / 16000
        total_from_voice = mux_done - first_ref
        after_last_audio = mux_done - (record["last_audio_wall"] or save_click_wall)
        after_save_click = mux_done - save_click_wall

        msg = (
            f"saved={mp4_path}\n"
            f"audio_duration={audio_duration:.2f}s\n"
            f"video_frames={frames.shape[0]}, video_duration={frames.shape[0] / 25:.2f}s\n"
            f"render_wait_after_save={render_done_wall - save_click_wall:.2f}s\n"
            f"mux_time={mux_done - mux_start:.2f}s\n"
            f"total_from_first_voice={total_from_voice:.2f}s\n"
            f"after_last_audio_callback={after_last_audio:.2f}s\n"
            f"after_save_click={after_save_click:.2f}s"
        )

        print("[BENCH_RESULT]\n" + msg, flush=True)
        return msg, mp4_path

    def clear_record():
        # 只建议刚启动后点一次。不要连续多轮复用同一个 worker。
        record["audio_chunks"].clear()
        record["frames"].clear()
        record["first_voice_wall"] = None
        record["first_audio_wall"] = None
        record["last_audio_wall"] = None

        # drain stale frames
        drain_frame_queue(frame_q, record["frames"])
        record["frames"].clear()

        return "cleared. Now speak once, then click Save MP4."

    with gr.Blocks(title="DyStream Mic Realtime to MP4 Benchmark") as demo:
        gr.Markdown(
            """
# DyStream 实时麦克风输入 → 后台流水线 → MP4 Benchmark

用途：不显示实时视频，只验证“你说话时后台是否已经在推理，结束后多久能拿到 MP4”。

建议测试：
1. 点击 Clear
2. 说 8 秒左右
3. 说完立刻点 Save MP4
4. 看保存耗时和 MP4 效果
"""
        )

        mic = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=True,
            label="Microphone",
        )

        status = gr.Textbox(label="Mic status")
        clear_btn = gr.Button("Clear")
        save_btn = gr.Button("Save MP4")
        save_status = gr.Textbox(label="Benchmark result", lines=8)
        save_file = gr.File(label="Saved MP4")

        mic.stream(
            fn=push_mic,
            inputs=mic,
            outputs=status,
            show_progress=False,
        )

        clear_btn.click(
            fn=clear_record,
            inputs=None,
            outputs=status,
        )

        save_btn.click(
            fn=save_mp4,
            inputs=None,
            outputs=[save_status, save_file],
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
