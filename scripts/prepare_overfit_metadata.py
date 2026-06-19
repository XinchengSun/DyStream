#!/usr/bin/env python3
"""Prepare DyStream metadata JSON files for a tiny overfit run.

The input manifest is a JSON list. Each item should contain:
  video_id, image_path, audio_path, motion_path

Optional fields:
  gt_video_path, caption, dataset_type, audio_other_path, motion_other_path

The training dataset currently expects motion npz files to contain a
`random_data` key with shape [T, 512]. If your latent file only has
`motion_latent`, pass --fix-motion-key to create a compatible copy.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_motion_frames(path: Path) -> tuple[int, str]:
    data = np.load(path, allow_pickle=True)
    if "random_data" in data:
        arr = data["random_data"]
        return int(arr.shape[0]), "random_data"
    if "motion_latent" in data:
        arr = data["motion_latent"]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return int(arr.shape[0]), "motion_latent"
    raise KeyError(f"{path} has neither random_data nor motion_latent")


def ensure_random_data_npz(src: Path, fixed_dir: Path, root: Path) -> Path:
    data = np.load(src, allow_pickle=True)
    if "random_data" in data:
        return src
    if "motion_latent" not in data:
        raise KeyError(f"{src} has neither random_data nor motion_latent")

    arr = data["motion_latent"]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    fixed_dir.mkdir(parents=True, exist_ok=True)
    dst = fixed_dir / src.name
    np.savez(dst, random_data=arr, motion_latent=arr)
    print(f"[metadata] wrote compatible latent: {relpath(dst, root)}")
    return dst


def verify_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def build_item(raw: dict, mode: str, root: Path, window_frames: int, fix_motion_key: bool, fixed_dir: Path) -> dict:
    required = ["video_id", "image_path", "audio_path", "motion_path"]
    missing = [k for k in required if not raw.get(k)]
    if missing:
        raise ValueError(f"manifest item is missing required fields {missing}: {raw}")

    image_path = (root / raw["image_path"]).resolve() if not Path(raw["image_path"]).is_absolute() else Path(raw["image_path"])
    audio_path = (root / raw["audio_path"]).resolve() if not Path(raw["audio_path"]).is_absolute() else Path(raw["audio_path"])
    motion_path = (root / raw["motion_path"]).resolve() if not Path(raw["motion_path"]).is_absolute() else Path(raw["motion_path"])

    verify_file(image_path, "image_path")
    verify_file(audio_path, "audio_path")
    verify_file(motion_path, "motion_path")

    if fix_motion_key:
        motion_path = ensure_random_data_npz(motion_path, fixed_dir, root)

    motion_frames, motion_key = load_motion_frames(motion_path)
    if motion_frames < window_frames:
        raise ValueError(
            f"{motion_path} has only {motion_frames} frames in {motion_key}, "
            f"but cbh_window_length/window_frames is {window_frames}."
        )

    audio_other_path = raw.get("audio_other_path")
    motion_other_path = raw.get("motion_other_path")

    if audio_other_path:
        other_audio = (root / audio_other_path).resolve() if not Path(audio_other_path).is_absolute() else Path(audio_other_path)
        verify_file(other_audio, "audio_other_path")
        audio_other_path = relpath(other_audio, root)

    if motion_other_path:
        other_motion = (root / motion_other_path).resolve() if not Path(motion_other_path).is_absolute() else Path(motion_other_path)
        verify_file(other_motion, "motion_other_path")
        motion_other_path = relpath(other_motion, root)

    gt_video_path = raw.get("gt_video_path")
    if gt_video_path:
        gt = (root / gt_video_path).resolve() if not Path(gt_video_path).is_absolute() else Path(gt_video_path)
        verify_file(gt, "gt_video_path")
        gt_video_path = relpath(gt, root)

    return {
        "origin_video_path": gt_video_path,
        "resampled_video_path": relpath(image_path, root),
        "audio_path": relpath(audio_path, root),
        "audio_self_path": relpath(audio_path, root),
        "audio_other_path": audio_other_path,
        "motion_self_path": relpath(motion_path, root),
        "motion_other_path": motion_other_path,
        "mode": mode,
        "dataset_type": raw.get("dataset_type", "speaker_only"),
        "video_id": raw["video_id"],
        "caption": raw.get("caption", ""),
        "start_idx": int(raw.get("start_idx", 0)),
        "end_idx": int(raw.get("end_idx", window_frames)),
        "frames": motion_frames,
    }


def expand_train_windows(item: dict, window_frames: int, stride_frames: int) -> list[dict]:
    if stride_frames <= 0:
        return [item]

    max_start = max(0, int(item["frames"]) - int(window_frames))
    starts = list(range(0, max_start + 1, int(stride_frames)))
    if starts[-1] != max_start:
        starts.append(max_start)

    windows = []
    for start in starts:
        window = dict(item)
        window["video_id"] = f"{item['video_id']}_w{start:04d}"
        window["start_idx"] = start
        window["end_idx"] = start + int(window_frames)
        windows.append(window)
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_json/overfit_items.json")
    parser.add_argument("--train-out", default="data_json/overfit_train.json")
    parser.add_argument("--test-out", default="data_json/overfit_test.json")
    parser.add_argument("--window-frames", type=int, default=96)
    parser.add_argument("--train-stride-frames", type=int, default=0)
    parser.add_argument("--fix-motion-key", action="store_true")
    parser.add_argument("--fixed-motion-dir", default="data_json/overfit_fixed_latents")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / args.manifest
    train_out = root / args.train_out
    test_out = root / args.test_out
    fixed_dir = root / args.fixed_motion_dir

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Create it from data_json/overfit_items.template.json first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or len(manifest) == 0:
        raise ValueError("manifest must be a non-empty JSON list")

    train_items = []
    for item in manifest:
        base_item = build_item(item, "train", root, args.window_frames, args.fix_motion_key, fixed_dir)
        train_items.extend(expand_train_windows(base_item, args.window_frames, args.train_stride_frames))
    test_items = [
        build_item(item, "test_wild", root, args.window_frames, args.fix_motion_key, fixed_dir)
        for item in manifest
    ]

    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text(json.dumps(train_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test_out.write_text(json.dumps(test_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[metadata] wrote {relpath(train_out, root)} ({len(train_items)} items)")
    print(f"[metadata] wrote {relpath(test_out, root)} ({len(test_items)} items)")


if __name__ == "__main__":
    main()
