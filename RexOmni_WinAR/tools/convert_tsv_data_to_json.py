"""
Reverse of convert_json_data_to_tsv.py:
Restore a TSV dataset (images.tsv + annotations.tsv + annotations.tsv.lineidx)
back to an image folder and a JSONL annotation file.

Output JSONL format (one JSON per line):
    {"image_name": "xxx.jpg", "annotation": {"boxes": [{"bbox": [...], "phrase": "..."}]}}
"""

import argparse
import json
import os
from base64 import b64decode
from io import BytesIO

from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Convert TSV dataset back to image folder + JSONL annotations"
    )
    parser.add_argument(
        "--image_tsv", type=str, required=True,
        help="Path to images.tsv",
    )
    parser.add_argument(
        "--ann_tsv", type=str, required=True,
        help="Path to annotations.tsv",
    )
    parser.add_argument(
        "--ann_lineidx", type=str, required=True,
        help="Path to annotations.tsv.lineidx",
    )
    parser.add_argument(
        "--save_image_dir", type=str, required=True,
        help="Output directory for restored images",
    )
    parser.add_argument(
        "--save_json_path", type=str, required=True,
        help="Output path for JSONL annotation file",
    )
    parser.add_argument(
        "--image_format", type=str, default="jpg",
        help="Format to save images (jpg, png, etc.)",
    )
    args = parser.parse_args()

    os.makedirs(args.save_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_json_path) or ".", exist_ok=True)

    line_offsets = []
    with open(args.ann_lineidx, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                line_offsets.append(int(line))

    print(f"Found {len(line_offsets)} samples in lineidx")

    img_handle = open(args.image_tsv, "r")
    ann_handle = open(args.ann_tsv, "r")
    json_out = open(args.save_json_path, "w", encoding="utf-8")

    saved = 0
    for idx, ann_offset in enumerate(tqdm(line_offsets, desc="Restoring")):
        ann_handle.seek(ann_offset)
        ann_line = ann_handle.readline().strip()
        parts = ann_line.split("\t", 1)
        if len(parts) < 2:
            continue
        img_line_idx = int(parts[0])
        annotation = json.loads(parts[1])

        img_handle.seek(img_line_idx)
        img_line = img_handle.readline().strip()
        img_parts = img_line.split("\t", 1)
        if len(img_parts) < 2:
            continue
        img_b64 = img_parts[1]
        if img_b64.startswith("b'"):
            img_b64 = img_b64[2:-1]

        img = Image.open(BytesIO(b64decode(img_b64))).convert("RGB")

        image_name = f"{idx:06d}.{args.image_format}"
        img.save(os.path.join(args.save_image_dir, image_name))

        record = {"image_name": image_name, "annotation": annotation}
        json_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        saved += 1

    img_handle.close()
    ann_handle.close()
    json_out.close()

    print(f"Done. Restored {saved} images to {args.save_image_dir}")
    print(f"Annotations saved to {args.save_json_path}")


if __name__ == "__main__":
    main()
