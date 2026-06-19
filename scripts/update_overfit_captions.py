import argparse
import json
from pathlib import Path


CAPTIONS = {
    "overfit_sample_001": (
        "adult male speaker, frontal cropped talking-head video, "
        "indoor interview lighting, neutral expression, English speech"
    ),
    "overfit_sample_002": (
        "adult female speaker, frontal cropped talking-head video, "
        "indoor interview lighting, neutral expression, English speech"
    ),
}


def update_file(path: Path) -> None:
    if not path.exists():
        print(f"[skip] missing {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for item in data:
        video_id = item.get("video_id")
        if video_id in CAPTIONS and item.get("caption") != CAPTIONS[video_id]:
            item["caption"] = CAPTIONS[video_id]
            changed += 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] {path}: updated {changed} caption(s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "data_json/overfit_items.json",
            "data_json/overfit_train.json",
            "data_json/overfit_test.json",
        ],
    )
    args = parser.parse_args()
    for path in args.paths:
        update_file(Path(path))


if __name__ == "__main__":
    main()
