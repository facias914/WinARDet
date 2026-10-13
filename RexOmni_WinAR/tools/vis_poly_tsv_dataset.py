import argparse
import json
import os
import random
from base64 import b64decode
from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = "/root/data-fs/Rex-Omni/Data/DOTA2_Poly"


class PolyTSVDataset:
    """Simple TSV dataset loader for polygon annotations.

    The DOTA2_Poly TSV files use the first column as a sample id. Some older
    Rex-Omni TSV files use that column as the image TSV byte offset. This loader
    supports both by indexing images.tsv ids first and falling back to seek().
    """

    def __init__(
        self,
        img_tsv_file: str,
        ann_tsv_file: str,
        ann_lineidx_file: str,
    ):
        self.img_tsv_file = img_tsv_file
        self.ann_tsv_file = ann_tsv_file
        self.ann_lineidx_file = ann_lineidx_file

        self.ann_offsets = []
        with open(ann_lineidx_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.ann_offsets.append(int(line))

        self.image_offsets = self._build_image_offset_index()
        self.img_handle = None
        self.ann_handle = None

    def _build_image_offset_index(self) -> Dict[str, int]:
        offsets = {}
        offset = 0
        with open(self.img_tsv_file, "rb") as f:
            for raw_line in f:
                sample_id = raw_line.split(b"	", 1)[0].decode("utf-8")
                offsets[sample_id] = offset
                offset += len(raw_line)
        return offsets

    def __len__(self):
        return len(self.ann_offsets)

    def _read_image_by_key(self, image_key: str) -> Image.Image:
        if self.img_handle is None:
            self.img_handle = open(self.img_tsv_file, "r", encoding="utf-8")

        if image_key in self.image_offsets:
            image_offset = self.image_offsets[image_key]
        else:
            image_offset = int(image_key)

        self.img_handle.seek(image_offset)
        parts = self.img_handle.readline().rstrip("\n").split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid image TSV row for key {image_key}")

        img = parts[1]
        if img.startswith("b'") or img.startswith("b\""):
            img = img[2:-1]
        return Image.open(BytesIO(b64decode(img))).convert("RGB")

    def load_image_and_anno(self, idx: int) -> Tuple[Image.Image, Dict]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        if self.ann_handle is None:
            self.ann_handle = open(self.ann_tsv_file, "r", encoding="utf-8")

        self.ann_handle.seek(self.ann_offsets[idx])
        parts = self.ann_handle.readline().rstrip("\n").split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid annotation TSV row at index {idx}")

        image_key, ann = parts
        image = self._read_image_by_key(image_key)
        data_dict = json.loads(ann)
        return image, data_dict


def polygon_from_annotation(box_info: Dict, polygon_key: str) -> List[float]:
    polygon = box_info.get(polygon_key)
    if polygon is None:
        polygon = box_info.get("polygon", box_info.get("rbbox"))
    if polygon is None:
        raise ValueError(f"No polygon field found in annotation: {box_info}")
    if len(polygon) != 8:
        raise ValueError(f"Expected 8 polygon coordinates, got {len(polygon)}: {polygon}")
    return polygon


def polygon_points(polygon: List[float]) -> np.ndarray:
    return np.asarray(polygon, dtype=float).reshape(4, 2)


def build_phrase_colors(boxes_info: List[Dict]):
    phrases = sorted(
        set(box.get("phrase") or box.get("caption", "") for box in boxes_info)
    )
    if not phrases:
        return {}
    color_values = np.linspace(0, 1, len(phrases))
    colors = [plt.cm.tab20(value) for value in color_values]
    return {phrase: colors[i] for i, phrase in enumerate(phrases)}


def visualize_sample(
    image: Image.Image,
    data_dict: Dict,
    save_path: str = None,
    polygon_key: str = "bbox",
    font_size: int = 10,
    linewidth: float = 2.0,
    alpha: float = 0.20,
    dpi: int = 150,
    show: bool = False,
):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)

    boxes_info = data_dict.get("boxes", [])
    if not boxes_info:
        print("Warning: no boxes found in annotation")

    phrase_to_color = build_phrase_colors(boxes_info)

    for box_info in boxes_info:
        try:
            polygon = polygon_from_annotation(box_info, polygon_key)
        except ValueError as exc:
            print(f"Warning: skip invalid polygon: {exc}")
            continue

        points = polygon_points(polygon)
        phrase = box_info.get("phrase") or box_info.get("caption", "")
        color = phrase_to_color.get(phrase, "red")

        patch = patches.Polygon(
            points,
            closed=True,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(patch)

        outline = patches.Polygon(
            points,
            closed=True,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(outline)
        ax.scatter(points[:, 0], points[:, 1], s=10, c=[color], edgecolors="white", linewidths=0.5)

        label_x = float(np.min(points[:, 0]))
        label_y = float(np.min(points[:, 1]))
        ax.text(
            label_x,
            label_y - 4,
            phrase,
            fontsize=font_size,
            color="white",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.8,
            ),
        )

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig, ax


def visualize_samples(
    dataset: PolyTSVDataset,
    indices: List[int],
    output_dir: str = None,
    polygon_key: str = "bbox",
    font_size: int = 10,
    linewidth: float = 2.0,
    alpha: float = 0.20,
    dpi: int = 150,
    show: bool = False,
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx in indices:
        print(f"Visualizing sample {idx}...")
        try:
            image, data_dict = dataset.load_image_and_anno(idx)
            save_path = os.path.join(output_dir, f"sample_{idx}.png") if output_dir else None
            visualize_sample(
                image,
                data_dict,
                save_path=save_path,
                polygon_key=polygon_key,
                font_size=font_size,
                linewidth=linewidth,
                alpha=alpha,
                dpi=dpi,
                show=show,
            )
        except Exception as exc:
            print(f"Error visualizing sample {idx}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DOTA-style polygon annotations from a TSV dataset."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing images.tsv, annotations.tsv, and annotations.tsv.lineidx.",
    )
    parser.add_argument("--img_tsv_file", type=str, default=None, help="Path to images.tsv.")
    parser.add_argument("--ann_tsv_file", type=str, default=None, help="Path to annotations.tsv.")
    parser.add_argument(
        "--ann_lineidx_file",
        type=str,
        default=None,
        help="Path to annotations.tsv.lineidx.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(DEFAULT_DATA_ROOT, "vis_poly_samples"),
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random samples to visualize.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to visualize. Overrides --num_samples.",
    )
    parser.add_argument("--polygon_key", type=str, default="bbox", help="Annotation key for polygon coordinates.")
    parser.add_argument("--font_size", type=int, default=10, help="Font size for labels.")
    parser.add_argument("--linewidth", type=float, default=2.0, help="Polygon outline width.")
    parser.add_argument("--alpha", type=float, default=0.20, help="Polygon fill alpha.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved visualizations.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def main():
    args = parse_args()

    img_tsv_file = args.img_tsv_file or os.path.join(args.data_root, "images.tsv")
    ann_tsv_file = args.ann_tsv_file or os.path.join(args.data_root, "annotations.tsv")
    ann_lineidx_file = args.ann_lineidx_file or os.path.join(args.data_root, "annotations.tsv.lineidx")

    print("Loading dataset...")
    dataset = PolyTSVDataset(img_tsv_file, ann_tsv_file, ann_lineidx_file)
    print(f"Dataset loaded: {len(dataset)} samples")

    if args.indices:
        sample_indices = args.indices
        print(f"Visualizing specified indices: {sample_indices}")
    else:
        random.seed(args.seed)
        sample_indices = list(range(len(dataset)))
        random.shuffle(sample_indices)
        sample_indices = sample_indices[: args.num_samples]
        print(f"Randomly sampling {len(sample_indices)} samples")

    visualize_samples(
        dataset,
        sample_indices,
        output_dir=args.output_dir,
        polygon_key=args.polygon_key,
        font_size=args.font_size,
        linewidth=args.linewidth,
        alpha=args.alpha,
        dpi=args.dpi,
        show=args.show,
    )

    if args.output_dir:
        print(f"Visualization complete! Check {args.output_dir} for saved images.")
    else:
        print("Visualization complete!")


if __name__ == "__main__":
    main()
