#!/usr/bin/env python3
"""Extract DyStream motion latents for every frame of a video.

This creates an npz file with `random_data` and `motion_latent`, both shaped
[T, 512], matching datasets/single_dyadic_prev_audio.py.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image


def setup_imports(project_root: Path, version: str) -> Path:
    vis_dir = project_root / "tools" / "visualization_0416"
    vis_model_dir = vis_dir / "utils" / f"model_{version}"

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(vis_dir) not in sys.path:
        sys.path.append(str(vis_dir))

    import model as model_pkg

    vis_model_package = vis_model_dir / "model"
    if str(vis_model_package) not in model_pkg.__path__:
        model_pkg.__path__.append(str(vis_model_package))

    return vis_dir


def load_motion_encoder(project_root: Path, version: str, device: torch.device):
    vis_dir = setup_imports(project_root, version)
    from utils import instantiate

    config_path = vis_dir / "configs" / f"head_animator_best_{version}.yaml"
    config = OmegaConf.load(config_path)
    if not os.path.isabs(config.resume_ckpt):
        config.resume_ckpt = os.path.normpath(str(vis_dir / config.resume_ckpt))

    module_cls = instantiate(config.model, instantiate_module=False)
    model = module_cls(config=config)
    checkpoint = torch.load(config.resume_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval().to(device)
    return model.motion_encoder.eval().to(device)


def iter_video_frames(video_path: Path, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(rgb)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
    finally:
        cap.release()


def unwrap_encoder_output(output):
    if hasattr(output, "detach"):
        return output
    if isinstance(output, (tuple, list)):
        tensor_items = [item for item in output if hasattr(item, "detach")]
        if not tensor_items:
            raise TypeError(f"motion_encoder returned {type(output)} without tensor items")
        latent_candidates = [item for item in tensor_items if item.ndim >= 2 and item.shape[-1] == 512]
        return latent_candidates[0] if latent_candidates else tensor_items[0]
    raise TypeError(f"Unsupported motion_encoder output type: {type(output)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="0506")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    video_path = (project_root / args.video).resolve() if not Path(args.video).is_absolute() else Path(args.video)
    output_path = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    motion_encoder = load_motion_encoder(project_root, args.version, device)

    latents = []
    batch = []
    with torch.no_grad():
        for image in iter_video_frames(video_path, args.max_frames):
            batch.append(transform(image.convert("RGB")))
            if len(batch) >= args.batch_size:
                x = torch.stack(batch, dim=0).to(device)
                latent = unwrap_encoder_output(motion_encoder(x)).detach().cpu().numpy()
                latents.append(latent)
                batch.clear()
        if batch:
            x = torch.stack(batch, dim=0).to(device)
            latent = unwrap_encoder_output(motion_encoder(x)).detach().cpu().numpy()
            latents.append(latent)

    if not latents:
        raise RuntimeError(f"No frames decoded from {video_path}")

    motion = np.concatenate(latents, axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        random_data=motion,
        motion_latent=motion,
        video_id=output_path.stem,
        video_path=str(video_path),
        ref_img_path=str(output_path.with_suffix(".png")),
    )
    print(f"[motion] wrote {output_path} shape={motion.shape}")


if __name__ == "__main__":
    main()
