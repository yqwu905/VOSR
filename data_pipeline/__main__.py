"""python -m data_pipeline {run,index,demo}."""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import validate_config
from .pipeline import Pipeline, atomic_json, json_text


def index_images(args):
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if not root.is_dir():
        raise ValueError("--root must be an image directory")
    if output.exists():
        raise ValueError("index output already exists")
    if args.hr_root and not args.teacher_checkpoint:
        raise ValueError("precomputed HR indexing requires --teacher-checkpoint provenance")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for directory, folders, files in os.walk(root):
            folders.sort()
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                    continue
                relative = path.relative_to(root)
                sample = {"id": relative.as_posix(), "source": args.source, "image_path": str(path)}
                if args.hr_root:
                    hr = Path(args.hr_root).resolve() / relative.with_suffix(".png")
                    if not hr.is_file():
                        raise FileNotFoundError(f"precomputed HR missing: {hr}")
                    sample.update(hr_path=str(hr), teacher={"model": args.teacher,
                                  "checkpoint": args.teacher_checkpoint})
                handle.write(json_text(sample) + "\n")
                count += 1
    temporary.replace(output)
    return {"indexed": count, "manifest": str(output)}


def create_demo(directory):
    """A CPU-only plumbing demonstration; all images stay in the review queue."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("input.jsonl", "config.json", "source.png"):
        if (directory / name).exists():
            raise ValueError("demo files already exist; choose a new directory")
    image = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    draw.text((30, 60), "TEXT SR 2026", fill="black", font=font)
    bbox = draw.textbbox((30, 60), "TEXT SR 2026", font=font)
    image.save(directory / "source.png")
    sample = {"id": "demo-001", "source": "synthetic_demo", "image_path": "source.png",
              "annotations": [{"bbox": list(bbox), "text": "TEXT SR 2026", "trusted": True}]}
    (directory / "input.jsonl").write_text(json_text(sample) + "\n", encoding="utf-8")
    config = validate_config({
        "long_edge": 1024,
        "quality": {"ocr_min_confidence": 0.9, "original_min_confidence": 0.95,
                    "iqa_thresholds": {"clarity": {"threshold": 0.5, "direction": "higher"},
                                       "naturalness": {"threshold": 0.5, "direction": "higher"},
                                       "artifacts": {"threshold": 0.5, "direction": "lower"}}},
        "backends": {"teacher": {"type": "demo_teacher"}, "ocr": {"type": "demo_ocr"},
                     "iqa": {"type": "demo_iqa"}, "glyph": {"type": "human"}},
        "augmentation": {"review_samples": True},
    })
    atomic_json(directory / "config.json", config)
    return Pipeline(config, directory / "input.jsonl", directory / "output").run()


def main():
    parser = argparse.ArgumentParser(description="Text SR data enhancement and cleaning")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="stream a multi-source JSONL manifest")
    run.add_argument("--input", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--resume", action="store_true")
    index = commands.add_parser("index", help="index a source tree without loading images")
    index.add_argument("--root", required=True)
    index.add_argument("--source", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--hr-root")
    index.add_argument("--teacher", choices=("tadisr", "vosr2"), default="vosr2")
    index.add_argument("--teacher-checkpoint")
    demo = commands.add_parser("demo", help="CPU plumbing demo; never exports training images")
    demo.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            result = Pipeline(config, args.input, args.output, args.resume).run()
        elif args.command == "index":
            result = index_images(args)
        else:
            result = create_demo(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Data errors are visible to schedulers even though valid records finish.
        return 2 if result.get("crops", {}).get("error", 0) else 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"data_pipeline: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
