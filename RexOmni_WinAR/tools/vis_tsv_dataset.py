import argparse
import json
import os
import random
from base64 import b64decode
from io import BytesIO
from typing import Dict, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TSVDataset(Dataset):
    """Simple TSV Dataset for loading images and annotations.

    Args:
        img_tsv_file: Path to the image TSV file
        ann_tsv_file: Path to the annotation TSV file
        ann_lineidx_file: Path to the annotation line index file
    """

    def __init__(
        self,
        img_tsv_file: str,
        ann_tsv_file: str,
        ann_lineidx_file: str,
    ):
        super(TSVDataset, self).__init__()

        self.data = []
        with open(ann_lineidx_file) as f:
            for line in f:
                self.data.append(int(line.strip()))

        self.img_handle = None
        self.ann_handle = None
        self.img_tsv_file = img_tsv_file
        self.ann_tsv_file = ann_tsv_file

    def __len__(self):
        return len(self.data)

    def load_image_and_anno(self, idx: int) -> Tuple[Image.Image, Dict]:
        """Load image and annotation for given index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (PIL Image, annotation dictionary)
        """
        try:
            ann_line_idx = self.data[idx]
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            idx = (idx + 1) % len(self.data)
            ann_line_idx = self.data[idx]

        if self.ann_handle is None:
            self.ann_handle = open(self.ann_tsv_file)

        self.ann_handle.seek(ann_line_idx)
        try:
            img_line_idx, ann = self.ann_handle.readline().strip().split("\t")
        except Exception as e:
            print(f"Error loading annotation for index {idx}: {e}")
            return self.load_image_and_anno((idx + 1) % len(self.data))

        try:
            img_line_idx = int(img_line_idx)
            if self.img_handle is None:
                self.img_handle = open(self.img_tsv_file)
            self.img_handle.seek(img_line_idx)
            img = self.img_handle.readline().strip().split("\t")[1]
            if img.startswith("b'"):
                img = img[1:-1]
            img = BytesIO(b64decode(img))
            image = Image.open(img).convert("RGB")
            data_dict = json.loads(ann)
        except Exception as e:
            print(f"Error loading image for index {idx}: {e}")
            return self.load_image_and_anno((idx + 1) % len(self.data))

        return image, data_dict

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict]:
        image_pil, data_dict = self.load_image_and_anno(idx)
        return image_pil, data_dict


def convert_bbox_format(bbox: list, from_format: str, to_format: str) -> list:
    """Convert bounding box between different formats.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2] or [x, y, width, height]
        from_format: Source format ('xyxy' or 'xywh')
        to_format: Target format ('xyxy' or 'xywh')

    Returns:
        Converted bounding box coordinates
    """
    if from_format == to_format:
        return bbox

    if from_format == "xyxy" and to_format == "xywh":
        # Convert from [x1, y1, x2, y2] to [x, y, width, height]
        x, y = bbox[0], bbox[1]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return [x, y, width, height]
    elif from_format == "xywh" and to_format == "xyxy":
        # Convert from [x, y, width, height] to [x1, y1, x2, y2]
        x1, y1 = bbox[0], bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        return [x1, y1, x2, y2]
    else:
        raise ValueError(f"Unsupported conversion from {from_format} to {to_format}")


def visualize_sample(
    image: Image.Image,
    data_dict: Dict,
    box_format: str = "xywh",
    save_path: str = None,
    font_size: int = 12,
    show: bool = True,
):
    """Visualize a sample with bounding boxes/points and phrases.

    Args:
        image: PIL Image to visualize
        data_dict: Annotation dictionary containing boxes/points and phrases
        box_format: Format of bounding boxes ('xywh' or 'xyxy'), only used for box data
        save_path: Path to save the visualization (optional)
        font_size: Font size for text labels
        show: Whether to display the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)

    # Check if this is box or point data
    boxes_info = data_dict.get("boxes", [])
    points_info = data_dict.get("points", [])

    if not boxes_info and not points_info:
        print("Warning: No boxes or points found in annotation")
        ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
        if show:
            plt.show()
        return fig, ax

    # Extract unique phrases and assign colors
    if boxes_info:
        unique_phrases = list(
            set(
                [
                    box_info.get("phrase") or box_info.get("caption", "")
                    for box_info in boxes_info
                ]
            )
        )
    else:  # points_info
        unique_phrases = list(
            set(
                [
                    point_info.get("phrase") or point_info.get("caption", "")
                    for point_info in points_info
                ]
            )
        )

    # Use a colormap with good contrast
    if len(unique_phrases) > 0:
        color_values = np.linspace(0, 1, len(unique_phrases))
        colors = [plt.cm.tab20(val) for val in color_values]
    else:
        colors = []
    phrase_to_color = {phrase: colors[i] for i, phrase in enumerate(unique_phrases)}

    # Draw bounding boxes if present
    if boxes_info:
        for box_info in boxes_info:
            bbox = box_info["bbox"]
            phrase = box_info.get("phrase") or box_info.get("caption", "")

            # Convert bbox to xywh format for visualization (matplotlib Rectangle expects xywh)
            if box_format == "xyxy":
                bbox_xywh = convert_bbox_format(bbox, "xyxy", "xywh")
            else:  # box_format == "xywh"
                bbox_xywh = bbox

            # Get color for this phrase
            color = phrase_to_color.get(phrase, "red")

            # Create rectangle patch
            rect = patches.Rectangle(
                (bbox_xywh[0], bbox_xywh[1]),
                bbox_xywh[2],
                bbox_xywh[3],
                linewidth=3,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            # Add text label with better visibility
            ax.text(
                bbox_xywh[0],
                bbox_xywh[1] - 5,
                phrase,
                fontsize=font_size,
                color="white",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1,
                ),
            )

    # Draw points if present
    if points_info:
        for point_info in points_info:
            point = point_info["point"]
            phrase = point_info.get("phrase") or point_info.get("caption", "")

            # Get color for this phrase
            color = phrase_to_color.get(phrase, "red")

            # Draw point as a circle
            x, y = point[0], point[1]
            circle = patches.Circle(
                (x, y),
                radius=8,  # Point size
                linewidth=3,
                edgecolor="white",
                facecolor=color,
                alpha=0.9,
            )
            ax.add_patch(circle)

            # Add text label with better visibility
            ax.text(
                x + 15,
                y,
                phrase,
                fontsize=font_size,
                color="white",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1,
                ),
            )

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


def visualize_samples(
    dataset: TSVDataset,
    indices: list,
    box_format: str = "xywh",
    save_dir: str = None,
    font_size: int = 12,
    show: bool = False,
):
    """Visualize multiple samples.

    Args:
        dataset: TSVDataset instance
        indices: List of sample indices to visualize
        box_format: Format of bounding boxes ('xywh' or 'xyxy')
        save_dir: Directory to save visualizations
        font_size: Font size for text labels
        show: Whether to display each plot
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for idx in indices:
        print(f"Visualizing sample {idx}...")
        try:
            image, data_dict = dataset.load_image_and_anno(idx)
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"sample_{idx}.png")
            visualize_sample(
                image,
                data_dict,
                box_format=box_format,
                save_path=save_path,
                font_size=font_size,
                show=show,
            )
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TSV dataset with bounding boxes/points and phrases. "
        "Supports both Grounding (boxes) and Pointing (points) data formats."
    )
    parser.add_argument(
        "--img_tsv_file",
        type=str,
        required=True,
        help="Path to the image TSV file",
    )
    parser.add_argument(
        "--ann_tsv_file",
        type=str,
        required=True,
        help="Path to the annotation TSV file",
    )
    parser.add_argument(
        "--ann_lineidx_file",
        type=str,
        required=True,
        help="Path to the annotation line index file",
    )
    parser.add_argument(
        "--box_format",
        type=str,
        default="xyxy",
        choices=["xywh", "xyxy"],
        help="Format of bounding boxes: 'xywh' (x, y, width, height) or 'xyxy' (x1, y1, x2, y2). "
        "Only used for Grounding data. Default: xyxy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations. If not specified, images won't be saved.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to visualize. Default: 10",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific indices to visualize. If provided, overrides num_samples.",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=12,
        help="Font size for text labels. Default: 12",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (use only for small num_samples)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling. Default: 42",
    )

    args = parser.parse_args()

    # Create dataset
    print("Loading dataset...")
    dataset = TSVDataset(
        args.img_tsv_file,
        args.ann_tsv_file,
        args.ann_lineidx_file,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Determine indices to visualize
    if args.indices:
        sample_indices = args.indices
        print(f"Visualizing specified indices: {sample_indices}")
    else:
        random.seed(args.seed)
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        sample_indices = all_indices[: args.num_samples]
        print(f"Randomly sampling {len(sample_indices)} samples")

    # Visualize samples
    print("=== Visualizing Samples ===")
    visualize_samples(
        dataset,
        sample_indices,
        box_format=args.box_format,
        save_dir=args.output_dir,
        font_size=args.font_size,
        show=args.show,
    )

    if args.output_dir:
        print(f"\nVisualization complete! Check '{args.output_dir}' for saved images.")
    else:
        print("\nVisualization complete!")


if __name__ == "__main__":
    main()
