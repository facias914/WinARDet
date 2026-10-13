import argparse
import base64
import io
import json
import os
from base64 import b64decode
from io import BytesIO

import numpy as np
from mmengine import Config
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm


class TSVBase(Dataset):
    """Base class for TSV dataset. This class is used to load image and annotations from TSV file.

    Args:
        img_tsv_file (str): The path to the image TSV file.
        ann_tsv_file (str): The path to the annotation TSV file.
        ann_lineidx_file (str): The path to the annotation lineidx file.
        num_workers (int): The number of workers.
        data_ratio (float, optional): The ratio of data to use. Defaults to 1.0.
        filter_empty (bool): If filter the samples without annotations. When training, set it to True.
        dataset_type (str): The data source.
    """

    def __init__(
        self,
        img_tsv_file: str,
        ann_tsv_file: str,
        ann_lineidx_file: str,
        num_samples: int,
    ):

        self.data = []
        f = open(ann_lineidx_file)
        for line in f:
            self.data.append(int(line.strip()))
        self.data = self.data[:num_samples]
        # sample data
        self.img_handle = None
        self.ann_handle = None
        self.ann_tsv_file = ann_tsv_file
        self.img_tsv_file = img_tsv_file
        print(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann_line_idx = self.data[idx]
        if self.ann_handle is None:
            self.ann_handle = open(self.ann_tsv_file)
        self.ann_handle.seek(ann_line_idx)
        img_line_idx, ann = self.ann_handle.readline().strip().split("\t")
        if self.img_handle is None:
            self.img_handle = open(self.img_tsv_file)
        img_line_idx = int(img_line_idx)
        self.img_handle.seek(img_line_idx)
        img = self.img_handle.readline().strip().split("\t")[1]
        if img.startswith("b'"):
            img = img[1:-1]
        img = BytesIO(b64decode(img))
        img = Image.open(img).convert("RGB")
        ann = json.loads(ann)
        return img, ann, img_line_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSON format data to TSV format"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to JSON file (one JSON object per line, each with 'image_name' and 'annotation' keys)",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        required=True,
        help="Root path where images are stored",
    )
    parser.add_argument(
        "--save_image_tsv_path",
        type=str,
        required=True,
        help="Output path for image TSV file",
    )
    parser.add_argument(
        "--save_ann_tsv_path",
        type=str,
        required=True,
        help="Output path for annotation TSV file",
    )
    parser.add_argument(
        "--save_ann_lineidx_path",
        type=str,
        required=True,
        help="Output path for annotation lineidx file",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    for path in [
        args.save_image_tsv_path,
        args.save_ann_tsv_path,
        args.save_ann_lineidx_path,
    ]:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    # Open output files
    f_image_tsv = open(args.save_image_tsv_path, "w")
    f_anno_tsv = open(args.save_ann_tsv_path, "w")
    f_anno_tsv_lineidx = open(args.save_ann_lineidx_path, "w")

    # Track line indices
    img_lineidx = 0
    anno_lineidx = 0

    # Read JSON file line by line
    with open(args.json_file, "r", encoding="utf-8") as json_f:
        for line in tqdm(json_f, desc="Converting JSON to TSV"):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse JSON line
                data = json.loads(line)

                # Extract image_name and annotation
                if "image_name" not in data or "annotation" not in data:
                    print(
                        f"Warning: Missing 'image_name' or 'annotation' key in line, skipping..."
                    )
                    continue

                image_name = data["image_name"]
                annotation = data["annotation"]

                # Construct full image path
                image_path = os.path.join(args.image_root_path, image_name)

                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}, skipping...")
                    continue

                # Load and validate image, then convert to base64
                try:
                    # Validate image can be opened
                    image = Image.open(image_path).convert("RGB")
                    # Convert to base64 directly
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                except Exception as e:
                    print(
                        f"Warning: Failed to load image {image_path}: {e}, skipping..."
                    )
                    continue

                # Write image to TSV (image_lineidx, base64_encoded_image)
                image_line = f"{img_lineidx}\t{img_base64}\n"
                image_length = len(image_line.encode("utf-8"))
                f_image_tsv.write(image_line)

                # Write annotation to TSV (image_lineidx, annotation_json)
                annotation_line = (
                    f"{img_lineidx}\t{json.dumps(annotation, ensure_ascii=False)}\n"
                )
                anno_length = len(annotation_line.encode("utf-8"))
                f_anno_tsv.write(annotation_line)

                # Write annotation lineidx
                f_anno_tsv_lineidx.write(f"{anno_lineidx}\n")

                # Update line indices
                img_lineidx += image_length
                anno_lineidx += anno_length

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON line: {e}, skipping...")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error processing line: {e}, skipping...")
                continue

    # Close files
    f_image_tsv.close()
    f_anno_tsv.close()
    f_anno_tsv_lineidx.close()

    print(f"Conversion completed!")
    print(f"Image TSV saved to: {args.save_image_tsv_path}")
    print(f"Annotation TSV saved to: {args.save_ann_tsv_path}")
    print(f"Annotation lineidx saved to: {args.save_ann_lineidx_path}")
