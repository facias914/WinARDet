#!/usr/bin/env python3
"""Convert DOTA polygon annotations to the TSV format used by Data/DOTA2.

The output annotation JSON keeps the existing shape:

    <image_tsv_offset>\t{"boxes": [{"bbox": [x0, y0, ..., x3, y3], "phrase": "..."}]}

By default the rotated box is stored in ``bbox`` so it stays compatible with the
existing DOTA2 TSV wrapper, while the value contains the 8 polygon coordinates.
The first column of both images.tsv and annotations.tsv is the byte offset of the
corresponding image row in images.tsv, matching the format expected by
GroundingTSVDataset.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DOTA2_poly images/annfiles into DOTA2-style TSV files."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/root/data-fs/Rex-Omni/Data/DOTA2_poly"),
        help="Source DOTA polygon dataset directory containing images/ and annfiles/.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/root/data-fs/Rex-Omni/Data/DOTA2_Poly"),
        help="Destination directory for images.tsv and annotations.tsv.",
    )
    parser.add_argument(
        "--bbox-key",
        default="bbox",
        help="JSON key used for the 8-value rotated box. Use rbbox if your loader expects it.",
    )
    parser.add_argument(
        "--keep-difficulty",
        action="store_true",
        help="Keep the DOTA difficulty value in each annotation object.",
    )
    parser.add_argument(
        "--write-image-lineidx",
        action="store_true",
        help="Also create images.tsv.lineidx. Existing DOTA2 only has annotations.tsv.lineidx.",
    )
    return parser.parse_args()


def iter_images(image_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def normalize_number(value: str) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def parse_annfile(
    ann_path: Path,
    bbox_key: str,
    keep_difficulty: bool,
) -> dict[str, list[dict[str, object]]]:
    boxes: list[dict[str, object]] = []
    if not ann_path.exists():
        return {"boxes": boxes}

    with ann_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"{ann_path}:{line_no}: expected at least 9 fields, got {len(parts)}")

            coords = [normalize_number(v) for v in parts[:8]]
            phrase = parts[8]
            difficulty = int(float(parts[9])) if len(parts) > 9 else 0
            box: dict[str, object] = {bbox_key: coords, "phrase": phrase}
            if keep_difficulty:
                box["difficulty"] = difficulty
            boxes.append(box)

    return {"boxes": boxes}


def write_lineidx(tsv_path: Path, lineidx_path: Path) -> None:
    offset = 0
    with tsv_path.open("rb") as src, lineidx_path.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(f"{offset}\n")
            offset += len(line)


def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def convert(args: argparse.Namespace) -> None:
    image_dir = args.src / "images"
    ann_dir = args.src / "annfiles"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing annotation directory: {ann_dir}")

    images = list(iter_images(image_dir))
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")

    args.dst.mkdir(parents=True, exist_ok=True)
    images_tsv = args.dst / "images.tsv"
    annotations_tsv = args.dst / "annotations.tsv"

    total_boxes = 0
    saved_images = 0
    skipped_images = 0
    image_offset = 0

    with images_tsv.open("wb") as image_out, annotations_tsv.open("wb") as ann_out:
        for image_path in images:
            ann_path = ann_dir / f"{image_path.stem}.txt"
            annotation = parse_annfile(
                ann_path=ann_path,
                bbox_key=args.bbox_key,
                keep_difficulty=args.keep_difficulty,
            )
            if not annotation["boxes"]:
                skipped_images += 1
                continue

            image_key = str(image_offset)
            image_line = f"{image_key}\t{image_to_base64(image_path)}\n".encode("utf-8")
            ann_line = (
                f"{image_key}\t"
                f"{json.dumps(annotation, ensure_ascii=False, separators=(',', ':'))}\n"
            ).encode("utf-8")

            image_out.write(image_line)
            ann_out.write(ann_line)

            image_offset += len(image_line)
            total_boxes += len(annotation["boxes"])
            saved_images += 1

    write_lineidx(annotations_tsv, args.dst / "annotations.tsv.lineidx")
    if args.write_image_lineidx:
        write_lineidx(images_tsv, args.dst / "images.tsv.lineidx")

    print(f"Converted {saved_images} images and {total_boxes} boxes")
    print(f"Skipped {skipped_images} images without annotations")
    print(f"Wrote {images_tsv}")
    print(f"Wrote {annotations_tsv}")
    print(f"Wrote {annotations_tsv}.lineidx")


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
