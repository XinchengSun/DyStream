#!/usr/bin/env python3
"""Compatibility entry point for DyStream video-to-motion-latent extraction.

The actual implementation lives in:

    scripts/extract_video_motion_latents.py

This wrapper keeps the shorter `video_to_latent.py` command name used in
reports and notes, while avoiding a second copy of the extraction logic.
"""

from pathlib import Path
import runpy


def main() -> None:
    script = Path(__file__).resolve().parent / "scripts" / "extract_video_motion_latents.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing extractor script: {script}")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
