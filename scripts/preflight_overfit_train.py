#!/usr/bin/env python3
"""Preflight checks before running the two-sample DyStream overfit training."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"missing {label}: {path}")
    ok(f"{label}: {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")


def check_ckpt(root: Path, ckpt: Path) -> None:
    require_file(ckpt, "checkpoint")
    head = ckpt.read_bytes()[:120]
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        fail(f"checkpoint is a Git LFS pointer, not real weights: {ckpt}")
    if not (head.startswith(b"PK") or b"archive/data.pkl" in head):
        print(f"[WARN] checkpoint header is not the usual zip header: {head[:40]!r}")
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        fail("checkpoint loaded, but has no state_dict")
    ok(f"torch.load checkpoint works; epoch={checkpoint.get('epoch')} global_step={checkpoint.get('global_step')}")
    ok(f"state_dict keys={len(checkpoint['state_dict'])}; ema_state={'ema_state' in checkpoint}")


def check_json(root: Path, path: Path, label: str, expected_count: int | None = None, min_count: int = 1) -> list[dict]:
    require_file(path, label)
    data = json.loads(path.read_text())
    if expected_count is not None and len(data) != expected_count:
        fail(f"{label} should contain exactly {expected_count} items, got {len(data)}")
    if len(data) < min_count:
        fail(f"{label} should contain at least {min_count} item(s), got {len(data)}")
    ok(f"{label} has {len(data)} item(s)")
    return data


def check_item(root: Path, item: dict, fix: bool) -> None:
    video_id = item.get("video_id", "<missing>")
    for key in ["origin_video_path", "resampled_video_path", "audio_path", "motion_self_path"]:
        if key not in item:
            fail(f"{video_id}: missing key {key}")
        require_file(root / item[key], f"{video_id}.{key}")

    motion_path = root / item["motion_self_path"]
    motion = np.load(motion_path)
    key = "random_data" if "random_data" in motion else "motion_latent"
    arr = motion[key]
    if arr.ndim != 2 or arr.shape[1] != 512:
        fail(f"{video_id}: bad motion shape {arr.shape}; expected [T, 512]")
    if arr.shape[0] < 96:
        fail(f"{video_id}: motion has only {arr.shape[0]} frames; expected at least 96")
    ok(f"{video_id}: motion {key} shape={arr.shape}")

    ref = root / item["resampled_video_path"]
    ref_resize = ref.with_name(ref.stem + "_resize" + ref.suffix)
    if not ref_resize.exists():
        if fix:
            shutil.copy2(ref, ref_resize)
            ok(f"{video_id}: created {ref_resize.relative_to(root)}")
        else:
            fail(f"{video_id}: missing {ref_resize}; run with --fix")
    else:
        ok(f"{video_id}: ref resize exists: {ref_resize.relative_to(root)}")


def check_train_py(root: Path) -> None:
    train_py = root / "train.py"
    require_file(train_py, "train.py")
    text = train_py.read_text()
    if "resume_mode" not in text or "weights_only=False" not in text or "fit_ckpt_path" not in text:
        fail("train.py does not contain the weights-only resume guard")
    ok("train.py supports resume_mode=weights_only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--ckpt", default="checkpoints/last.ckpt")
    parser.add_argument("--train-json", default="data_json/overfit_train.json")
    parser.add_argument("--test-json", default="data_json/overfit_test.json")
    parser.add_argument("--expected-train-items", type=int, default=0)
    parser.add_argument("--expected-test-items", type=int, default=2)
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"[INFO] root={root}")
    print(f"[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        fail("CUDA is not available")

    check_ckpt(root, (root / args.ckpt).resolve())
    check_train_py(root)
    train_expected = args.expected_train_items if args.expected_train_items > 0 else None
    test_expected = args.expected_test_items if args.expected_test_items > 0 else None
    train_items = check_json(root, root / args.train_json, "train metadata", expected_count=train_expected, min_count=2)
    test_items = check_json(root, root / args.test_json, "test metadata", expected_count=test_expected, min_count=1)
    for item in train_items + test_items:
        check_item(root, item, args.fix)

    ok("overfit training preflight passed")


if __name__ == "__main__":
    main()
