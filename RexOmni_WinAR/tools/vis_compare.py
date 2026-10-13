#!/usr/bin/env python3
"""Compare two same-format box, polygon, or point prediction JSONLs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vis_predictions import (
    COLORS,
    FORMATS,
    add_point_legend,
    detect_file_fallback,
    # draw_boxes,
    # draw_points,
    # draw_polygons,
    infer_record_format,
    normalized_data,
    resolve_image_path,
)

from matplotlib.patches import Rectangle, Polygon


def draw_boxes(
    ax,
    image,
    data,
    title,
    color_map,
    show_text: bool = True,
):
    """
    Draw bounding boxes without category names.
    """
    ax.imshow(image)

    for category, boxes in data.items():
        color = color_map[str(category)]

        for box in boxes:
            if len(box) < 4:
                continue

            x1, y1, x2, y2 = box[:4]

            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
            ax.add_patch(rect)

            # 不绘制类别名称
            # ax.text(...)

    if show_text:
        ax.set_title(title)
    ax.axis("off")


def draw_polygons(
    ax,
    image,
    data,
    title,
    color_map,
    show_text: bool = True,
):
    """
    Draw polygons without category names.
    """
    ax.imshow(image)

    for category, polygons in data.items():
        color = color_map[str(category)]

        for polygon in polygons:
            if not polygon:
                continue

            # 支持：
            # [x1, y1, x2, y2, x3, y3, ...]
            # 或
            # [[x1, y1], [x2, y2], ...]
            if isinstance(polygon[0], (list, tuple)):
                points = polygon
            else:
                if len(polygon) < 6 or len(polygon) % 2 != 0:
                    continue

                points = [
                    (polygon[i], polygon[i + 1])
                    for i in range(0, len(polygon), 2)
                ]

            patch = Polygon(
                points,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
            ax.add_patch(patch)

            # 不绘制类别名称
            # ax.text(...)

    if show_text:
        ax.set_title(title)
    ax.axis("off")


def draw_points(
    ax,
    image,
    data,
    title,
    color_map,
    point_size,
    show_text: bool = True,
):
    """
    Draw points without category names.

    Returns:
        valid_points
        malformed_points
        out_of_bounds_points
    """
    ax.imshow(image)

    width, height = image.size

    valid_points = 0
    malformed_points = 0
    out_of_bounds_points = 0

    for category, points in data.items():
        color = color_map[str(category)]

        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                malformed_points += 1
                continue

            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                malformed_points += 1
                continue

            if x < 0 or x >= width or y < 0 or y >= height:
                out_of_bounds_points += 1

            ax.scatter(
                x,
                y,
                s=point_size,
                c=color,
            )

            valid_points += 1

            # 不绘制类别名称
            # ax.text(...)

    if show_text:
        ax.set_title(title)
    ax.axis("off")

    return (
        valid_points,
        malformed_points,
        out_of_bounds_points,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two prediction JSONLs in three columns: ground truth, first "
            "predictions, and second predictions. Box, polygon, and point formats "
            "are detected automatically."
        )
    )
    parser.add_argument(
        "--jsonl-path",
        "--first-jsonl",
        dest="jsonl_path",
        type=Path,
        default=Path("/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/SKU110K/sft_base_oversample300/Results/Sku110k_Grounding/answer.jsonl"),
        help="First prediction JSONL path (default: %(default)s).",
    )
    parser.add_argument(
        "--second-jsonl-path",
        "--second-jsonl",
        dest="second_jsonl_path",
        type=Path,
        default=Path("/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/SKU110K/sft_seg_200_4-6-4_l6_e4_lr10_rescale/Results/SKU110K_Grounding/answer.jsonl"),
        help="Second prediction JSONL path (default: %(default)s).",
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=Path("/home/u1120250365/share/Rex_Omni/Compression/test/SKU110K/images"),
        help="Root directory used to resolve image_path (default: %(default)s).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("/home/u1120250365/Rex-Omni/visualization_compare"),
        help="Directory for rendered comparisons (default: %(default)s).",
    )
    parser.add_argument(
        "--first-label",
        dest="first_prediction_label",
        default="base",
        help="Middle-column label; defaults to the first JSONL filename.",
    )
    parser.add_argument(
        "--second-label",
        dest="second_prediction_label",
        default="compression",
        help="Right-column label; defaults to the second JSONL filename.",
    )
    parser.add_argument(
        "--format",
        dest="prediction_format",
        choices=("auto",) + FORMATS,
        default="auto",
        help="Prediction format shared by both files (default: %(default)s).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based line index in the first JSONL at which rendering starts.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum comparisons to render; by default render all matched images.",
    )
    parser.add_argument(
        "--min-targets",
        type=int,
        default=200,
        help=(
            "Only visualize images whose ground-truth target count is greater "
            "than this value; by default do not filter by target count."
        ),
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
        help="Point marker area in points squared (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution (default: %(default)s).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every N images; use 0 to suppress (default: %(default)s).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Render figures without any text (titles/legends) and suppress "
            "console output; only the drawings are saved."
        ),
    )
    parser.add_argument(
        "--panel-gap",
        type=float,
        default=0.04,
        help=(
            "Horizontal spacing between the three panels in --plot-only mode, "
            "as a fraction of the average panel width (default: %(default)s)."
        ),
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON in {path} at line {line_index + 1}: {exc}")
                continue
            if not isinstance(record, dict):
                print(f"Skipping non-object JSON in {path} at line {line_index + 1}")
                continue
            yield line_index, record


def record_image_path(record: Mapping) -> str:
    return str(
        record.get("image_path")
        or record.get("image")
        or record.get("image_name")
        or ""
    ).strip()


def build_prediction_index(path: Path):
    by_image_path = {}
    by_basename = {}

    for line_index, record in iter_jsonl(path):
        image_path = record_image_path(record)
        if not image_path:
            print(f"Skipping {path} line {line_index + 1}: image_path is missing")
            continue
        if image_path in by_image_path:
            previous_line = by_image_path[image_path][0] + 1
            raise ValueError(
                f"Duplicate image_path {image_path!r} in {path} at lines "
                f"{previous_line} and {line_index + 1}"
            )
        indexed_record = (line_index, record)
        by_image_path[image_path] = indexed_record
        by_basename.setdefault(Path(image_path).name, []).append(indexed_record)

    return by_image_path, by_basename


def find_matching_record(
    image_path: str,
    by_image_path: Dict,
    by_basename: Dict,
) -> Tuple[Optional[Tuple[int, Dict]], Optional[str]]:
    exact_match = by_image_path.get(image_path)
    if exact_match is not None:
        return exact_match, "image_path"

    basename = Path(image_path).name
    basename_matches = by_basename.get(basename, [])
    if len(basename_matches) == 1:
        return basename_matches[0], "basename"
    if len(basename_matches) > 1:
        lines = [line_index + 1 for line_index, _ in basename_matches]
        raise ValueError(
            f"Multiple records named {basename!r} occur in the second JSONL "
            f"at lines {lines}; basename matching is ambiguous"
        )
    return None, None


def detect_shared_format(
    primary_path: Path,
    secondary_path: Path,
    requested_format: str,
) -> str:
    if requested_format != "auto":
        print(f"Using requested format for both JSONLs: {requested_format}")
        return requested_format

    primary_format = detect_file_fallback(primary_path)
    secondary_format = detect_file_fallback(secondary_path)
    if primary_format is None:
        raise ValueError(
            f"Cannot determine one format for the first JSONL: {primary_path}. "
            "Use --format box, --format polygon, or --format point."
        )
    if secondary_format is None:
        raise ValueError(
            f"Cannot determine one format for the second JSONL: {secondary_path}. "
            "Use --format box, --format polygon, or --format point."
        )
    if primary_format != secondary_format:
        raise ValueError(
            "The two JSONLs use different formats: first={}, second={}".format(
                primary_format,
                secondary_format,
            )
        )
    print(f"Detected shared format: {primary_format}")
    return primary_format


def validate_record_format(
    record: Mapping,
    expected_format: str,
    path: Path,
    line_index: int,
) -> None:
    detected_format = infer_record_format(record)
    if detected_format is not None and detected_format != expected_format:
        raise ValueError(
            f"Unexpected {detected_format} record in {path} at line "
            f"{line_index + 1}; expected {expected_format}"
        )


def comparison_color_map(*data_items: Mapping) -> Dict[str, str]:
    categories = sorted(
        {
            str(category)
            for data in data_items
            for category in data
        }
    )
    return {
        category: COLORS[index % len(COLORS)]
        for index, category in enumerate(categories)
    }


def point_comparison_size(image: Image.Image) -> Tuple[float, float]:
    figure_width = 30.0
    panel_width = figure_width / 3
    image_height = panel_width * image.height / image.width
    return figure_width, min(12.0, max(5.0, image_height + 1.5))


def render_comparison(
    image: Image.Image,
    image_path_value: str,
    gt_data: Mapping,
    first_predictions: Mapping,
    second_predictions: Mapping,
    prediction_format: str,
    first_label: str,
    second_label: str,
    point_size: float,
    show_text: bool = True,
    panel_gap: float = 0.04,
) -> Tuple[plt.Figure, int, int]:
    color_map = comparison_color_map(
        gt_data,
        first_predictions,
        second_predictions,
    )
    malformed_points = 0
    out_of_bounds_points = 0

    if prediction_format == "point":
        figure, axes = plt.subplots(
            1, 3,
            figsize=point_comparison_size(image),
            gridspec_kw={"wspace": panel_gap, "hspace": 0.0} if not show_text else None,
        )
        if show_text:
            figure.suptitle(image_path_value, fontsize=16, y=0.985)
        _, gt_bad, gt_outside = draw_points(
            axes[0], image, gt_data, "Ground Truth", color_map, point_size, show_text
        )
        _, first_bad, first_outside = draw_points(
            axes[1],
            image,
            first_predictions,
            f"Predictions ({first_label})",
            color_map,
            point_size,
            show_text,
        )
        _, second_bad, second_outside = draw_points(
            axes[2],
            image,
            second_predictions,
            f"Predictions ({second_label})",
            color_map,
            point_size,
            show_text,
        )
        malformed_points = gt_bad + first_bad + second_bad
        out_of_bounds_points = gt_outside + first_outside + second_outside
        if show_text:
            add_point_legend(figure, color_map)
            figure.tight_layout(rect=(0, 0, 1, 0.91))
        else:
            # Keep a small configurable gap between the three panels while removing outer margins.
            figure.subplots_adjust(
                left=0.0, right=1.0, bottom=0.0, top=1.0,
                wspace=panel_gap, hspace=0.0,
            )
        return figure, malformed_points, out_of_bounds_points

    figure, axes = plt.subplots(
        1, 3,
        figsize=(30, 10),
        gridspec_kw={"wspace": panel_gap, "hspace": 0.0} if not show_text else None,
    )
    if prediction_format == "polygon":
        draw_polygons(
            axes[0], image, gt_data, f"GT: {image_path_value}", color_map, show_text
        )
        draw_polygons(
            axes[1],
            image,
            first_predictions,
            f"{first_label}: {image_path_value}",
            color_map,
            show_text,
        )
        draw_polygons(
            axes[2],
            image,
            second_predictions,
            f"{second_label}: {image_path_value}",
            color_map,
            show_text,
        )
    else:
        draw_boxes(
            axes[0],
            image,
            gt_data,
            f"Ground Truth: {image_path_value}",
            color_map,
            show_text,
        )
        draw_boxes(
            axes[1],
            image,
            first_predictions,
            f"Predictions ({first_label}): {image_path_value}",
            color_map,
            show_text,
        )
        draw_boxes(
            axes[2],
            image,
            second_predictions,
            f"Predictions ({second_label}): {image_path_value}",
            color_map,
            show_text,
        )
    if show_text:
        figure.tight_layout(pad=1.08)
    else:
        # Keep a small configurable gap between the three panels while removing outer margins.
        figure.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0,
            wspace=panel_gap, hspace=0.0,
        )
    return figure, malformed_points, out_of_bounds_points


def validate_options(
    primary_path: Path,
    secondary_path: Path,
    image_root: Path,
    start_index: int,
    max_images: Optional[int],
    min_targets: Optional[int],
    point_size: float,
    dpi: int,
    log_every: int,
    prediction_format: str,
    panel_gap: float,
) -> None:
    if not primary_path.is_file():
        raise FileNotFoundError(primary_path)
    if not secondary_path.is_file():
        raise FileNotFoundError(secondary_path)
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")
    if min_targets is not None and min_targets < 0:
        raise ValueError("min_targets must be non-negative")
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if log_every < 0:
        raise ValueError("log_every must be non-negative")
    if panel_gap < 0:
        raise ValueError("panel_gap must be non-negative")
    if prediction_format not in ("auto",) + FORMATS:
        raise ValueError(
            f"prediction_format must be one of auto, {', '.join(FORMATS)}"
        )


def visualize_results(
    jsonl_path: str,
    second_jsonl_path: str,
    image_folder: str,
    output_folder: str,
    first_prediction_label: Optional[str] = None,
    second_prediction_label: Optional[str] = None,
    start_index: int = 0,
    max_images: Optional[int] = None,
    min_targets: Optional[int] = None,
    point_size: float = 18.0,
    dpi: int = 150,
    log_every: int = 1,
    prediction_format: str = "auto",
    plot_only: bool = False,
    panel_gap: float = 0.04,
) -> None:
    primary_path = Path(jsonl_path)
    secondary_path = Path(second_jsonl_path)
    image_root = Path(image_folder)
    output_root = Path(output_folder)
    validate_options(
        primary_path,
        secondary_path,
        image_root,
        start_index,
        max_images,
        min_targets,
        point_size,
        dpi,
        log_every,
        prediction_format,
        panel_gap,
    )

    shared_format = detect_shared_format(
        primary_path,
        secondary_path,
        prediction_format,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    first_label = first_prediction_label or primary_path.stem
    second_label = second_prediction_label or secondary_path.stem
    secondary_by_path, secondary_by_basename = build_prediction_index(secondary_path)

    saved_count = 0
    missing_secondary_count = 0
    missing_image_count = 0
    basename_match_count = 0
    malformed_point_count = 0
    out_of_bounds_count = 0

    for line_index, primary_record in iter_jsonl(primary_path):
        if line_index < start_index:
            continue
        if max_images is not None and saved_count >= max_images:
            break

        image_path_value = record_image_path(primary_record)
        if not image_path_value:
            print(f"Skipping {primary_path} line {line_index + 1}: image_path is missing")
            missing_image_count += 1
            continue

        matched_record, match_type = find_matching_record(
            image_path_value,
            secondary_by_path,
            secondary_by_basename,
        )
        if matched_record is None:
            missing_secondary_count += 1
            print(f"Skipping {image_path_value}: not found in the second JSONL")
            continue
        secondary_line_index, secondary_record = matched_record
        if match_type == "basename":
            basename_match_count += 1

        validate_record_format(
            primary_record,
            shared_format,
            primary_path,
            line_index,
        )
        validate_record_format(
            secondary_record,
            shared_format,
            secondary_path,
            secondary_line_index,
        )

        image_path = resolve_image_path(image_root, image_path_value)
        if image_path is None:
            print(f"Skipping {image_path_value}: image file was not found")
            missing_image_count += 1
            continue

        gt_data = normalized_data(primary_record, "gt", line_index + 1)

        # Optional density filter: only keep images with more than min_targets GT objects.
        if min_targets is not None:
            target_count = sum(len(targets) for targets in gt_data.values())
            if target_count <= min_targets:
                continue

        first_predictions = normalized_data(
            primary_record,
            "extracted_predictions",
            line_index + 1,
        )
        second_predictions = normalized_data(
            secondary_record,
            "extracted_predictions",
            secondary_line_index + 1,
        )

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            figure, malformed_points, outside_points = render_comparison(
                image,
                image_path_value,
                gt_data,
                first_predictions,
                second_predictions,
                shared_format,
                first_label,
                second_label,
                point_size,
                show_text=not plot_only,
                panel_gap=panel_gap,
            )
            save_name = f"res_{line_index}_{Path(image_path_value).name}"
            figure.savefig(
                output_root / save_name,
                bbox_inches="tight",
                dpi=dpi,
                facecolor="white",
                pad_inches=0.0 if plot_only else 0.1,
            )
            plt.close(figure)

        malformed_point_count += malformed_points
        out_of_bounds_count += outside_points
        saved_count += 1
        if log_every > 0 and saved_count % log_every == 0:
            print(f"Saved {saved_count} comparisons; latest: {save_name}")

    print(
        "Done: saved {} {} comparisons, missing in second JSONL {}, missing "
        "images {}, basename matches {}, malformed points {}, out-of-bounds "
        "points {}. Output: {}".format(
            saved_count,
            shared_format,
            missing_secondary_count,
            missing_image_count,
            basename_match_count,
            malformed_point_count,
            out_of_bounds_count,
            output_root,
        )
    )


def main() -> None:
    args = parse_args()

    def run_visualization() -> None:
        visualize_results(
            jsonl_path=args.jsonl_path,
            second_jsonl_path=args.second_jsonl_path,
            image_folder=args.image_folder,
            output_folder=args.output_folder,
            first_prediction_label=args.first_prediction_label,
            second_prediction_label=args.second_prediction_label,
            start_index=args.start_index,
            max_images=args.max_images,
            min_targets=args.min_targets,
            point_size=args.point_size,
            dpi=args.dpi,
            log_every=args.log_every,
            prediction_format=args.prediction_format,
            plot_only=args.plot_only,
            panel_gap=args.panel_gap,
        )

    if args.plot_only:
        # Suppress all normal stdout messages in plot-only mode.
        with contextlib.redirect_stdout(io.StringIO()):
            run_visualization()
    else:
        run_visualization()


if __name__ == "__main__":
    main()