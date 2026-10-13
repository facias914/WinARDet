#!/usr/bin/env python3
"""Automatically visualize box, polygon, or point prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


COLORS = (
    "#FF3838",
    "#2DFF2D",
    "#2D52FF",
    "#F92FFF",
    "#FFD300",
    "#00D3FF",
    "#00FFD3",
    "#D300FF",
)
FORMATS = ("box", "polygon", "point")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize predictions and ground truth from a JSONL file. The input "
            "format is detected automatically from metadata and coordinate lengths."
        )
    )
    parser.add_argument(
        "--jsonl-path",
        "--input",
        dest="jsonl_path",
        type=Path,
        default=Path("/home/ubuntu/wcx/Rex-Omni/分析冗余率/basew.jsonl"),
        help="Prediction JSONL path (default: %(default)s).",
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=Path("/home/ubuntu/wcx/Locayany/DATA/test/rs-dense200/images"),
        help="Root directory used to resolve image_path (default: %(default)s).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("/home/ubuntu/wcx/Rex-Omni/visualization_out_weight"),
        help="Directory for rendered images (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        dest="prediction_format",
        choices=("auto",) + FORMATS,
        default="auto",
        help="Prediction format. auto detects each record (default: %(default)s).",
    )
    parser.add_argument(
        "--label",
        default="Pred-1",
        help="Prediction-panel title used for polygon visualization.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based JSONL line index at which rendering starts.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to render; by default render all remaining.",
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
        default=None,
        help=(
            "Print progress every N images. Defaults to 20 for point data and 1 "
            "for box/polygon data; use 0 to suppress progress."
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
                print(f"Skipping invalid JSON at line {line_index + 1}: {exc}")
                continue
            if not isinstance(record, dict):
                print(f"Skipping non-object JSON at line {line_index + 1}")
                continue
            yield line_index, record


def coordinate_lengths(record: Mapping) -> set[int]:
    lengths = set()
    for field_name in ("gt", "extracted_predictions"):
        category_data = record.get(field_name)
        if not isinstance(category_data, dict):
            continue
        for instances in category_data.values():
            if not isinstance(instances, list):
                continue
            for instance in instances:
                if isinstance(instance, (list, tuple)):
                    lengths.add(len(instance))
    return lengths


def metadata_format(record: Mapping) -> Optional[str]:
    task_name = str(record.get("task_name", "")).lower()
    question = str(record.get("question", "")).lower()

    if "polygon" in task_name:
        return "polygon"
    if "point" in task_name:
        return "point"

    polygon_phrases = (
        "polygon",
        "oriented bounding box",
        "four-point",
        "x2, y2",
        "x3, y3",
    )
    if any(phrase in question for phrase in polygon_phrases):
        return "polygon"
    if "point coordinate" in question or "point location" in question:
        return "point"
    if "bounding box" in question or "x1, y1" in question:
        return "box"
    if "detection" in task_name or task_name == "common_object_detection":
        return "box"
    return None


def infer_record_format(record: Mapping) -> Optional[str]:
    metadata_hint = metadata_format(record)
    if metadata_hint is not None:
        return metadata_hint

    lengths = coordinate_lengths(record)
    if any(length >= 8 and length % 2 == 0 for length in lengths):
        return "polygon"

    has_point = 2 in lengths
    has_box = 4 in lengths
    if has_point and has_box:
        return None
    if has_point:
        return "point"
    if has_box:
        return "box"
    return None


def detect_file_fallback(path: Path, sample_limit: int = 200) -> Optional[str]:
    detected = Counter()
    inspected = 0
    for _, record in iter_jsonl(path):
        prediction_format = infer_record_format(record)
        if prediction_format is not None:
            detected[prediction_format] += 1
        inspected += 1
        if inspected >= sample_limit:
            break

    if len(detected) == 1:
        return next(iter(detected))
    return None


def normalized_data(record: Mapping, field_name: str, line_number: int) -> Mapping:
    value = record.get(field_name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        print(
            f"Ignoring {field_name} at line {line_number}: expected an object, "
            f"got {type(value).__name__}"
        )
        return {}
    return value


def resolve_image_path(image_folder: Path, image_path_value: str) -> Optional[Path]:
    image_path = Path(image_path_value)
    candidates = []
    if image_path.is_absolute():
        candidates.append(image_path)
    else:
        candidates.extend((image_folder / image_path, image_folder / image_path.name))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def category_color_map(gt_data: Mapping, prediction_data: Mapping) -> Dict[str, str]:
    categories = sorted(
        {str(category) for category in gt_data}
        | {str(category) for category in prediction_data}
    )
    return {
        category: COLORS[index % len(COLORS)]
        for index, category in enumerate(categories)
    }


def prepare_axis(axis, image: Image.Image, title: str, bold: bool = False) -> None:
    axis.imshow(image)
    axis.set_title(title, fontsize=13 if bold else 15, fontweight="bold" if bold else None)
    axis.axis("off")


def numeric_coordinates(instance: object, count: int) -> Optional[list[float]]:
    if not isinstance(instance, (list, tuple)) or len(instance) < count:
        return None
    try:
        coordinates = [float(value) for value in instance[:count]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coordinates):
        return None
    return coordinates


def draw_boxes(
    axis,
    image: Image.Image,
    box_data: Mapping,
    title: str,
    color_map: Mapping[str, str],
) -> None:
    prepare_axis(axis, image, title)
    for category, boxes in box_data.items():
        if not isinstance(boxes, list):
            continue
        color = color_map.get(str(category), "#FFFFFF")
        for box in boxes:
            coordinates = numeric_coordinates(box, 4)
            if coordinates is None:
                continue
            x0, y0, x1, y1 = coordinates
            axis.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
            )


def polygon_coordinates(instance: object) -> Optional[list[float]]:
    if not isinstance(instance, (list, tuple)):
        return None
    if len(instance) >= 8:
        return numeric_coordinates(instance, 8)
    if len(instance) == 4:
        box = numeric_coordinates(instance, 4)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        return [x0, y0, x1, y0, x1, y1, x0, y1]
    return None


def draw_polygons(
    axis,
    image: Image.Image,
    polygon_data: Mapping,
    title: str,
    color_map: Mapping[str, str],
) -> None:
    prepare_axis(axis, image, title, bold=True)
    for category, polygons in polygon_data.items():
        if not isinstance(polygons, list):
            continue
        category_name = str(category)
        color = color_map.get(category_name, "#FFFFFF")
        for polygon in polygons:
            coordinates = polygon_coordinates(polygon)
            if coordinates is None:
                continue
            points = [
                (coordinates[index], coordinates[index + 1])
                for index in range(0, 8, 2)
            ]
            axis.add_patch(
                patches.Polygon(
                    points,
                    closed=True,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.16,
                )
            )
            axis.add_patch(
                patches.Polygon(
                    points,
                    closed=True,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
            )
            axis.text(
                min(point[0] for point in points),
                min(point[1] for point in points),
                category_name,
                color="white",
                fontsize=9,
                bbox=dict(facecolor=color, alpha=0.75, edgecolor="none"),
            )


def valid_points(
    instances: object,
    image_width: int,
    image_height: int,
) -> Tuple[list[Tuple[float, float]], int, int]:
    if not isinstance(instances, list):
        return [], 1, 0

    points = []
    malformed = 0
    out_of_bounds = 0
    for instance in instances:
        coordinates = numeric_coordinates(instance, 2)
        if coordinates is None or not isinstance(instance, (list, tuple)) or len(instance) != 2:
            malformed += 1
            continue
        x, y = coordinates
        if not (0 <= x < image_width and 0 <= y < image_height):
            out_of_bounds += 1
            continue
        points.append((x, y))
    return points, malformed, out_of_bounds


def draw_points(
    axis,
    image: Image.Image,
    point_data: Mapping,
    title: str,
    color_map: Mapping[str, str],
    point_size: float,
) -> Tuple[int, int, int]:
    axis.imshow(image)
    axis.axis("off")
    axis.set_xlim(0, image.width)
    axis.set_ylim(image.height, 0)
    axis.margins(0)

    total = 0
    malformed = 0
    out_of_bounds = 0
    for category, instances in point_data.items():
        category_name = str(category)
        points, bad_count, outside_count = valid_points(
            instances,
            image.width,
            image.height,
        )
        malformed += bad_count
        out_of_bounds += outside_count
        if not points:
            continue
        x_values, y_values = zip(*points)
        axis.scatter(
            x_values,
            y_values,
            s=point_size,
            c=color_map.get(category_name, "#FFFFFF"),
            marker="o",
            edgecolors="#111111",
            linewidths=0.45,
            alpha=0.9,
            zorder=2,
        )
        total += len(points)

    axis.set_title(f"{title} ({total} points)", fontsize=15)
    return total, malformed, out_of_bounds


def point_figure_size(image: Image.Image) -> Tuple[float, float]:
    figure_width = 20.0
    panel_width = figure_width / 2
    image_height = panel_width * image.height / image.width
    return figure_width, min(12.0, max(5.0, image_height + 1.5))


def add_point_legend(figure, color_map: Mapping[str, str]) -> None:
    if not color_map:
        return
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="#111111",
            markersize=7,
            label=category,
        )
        for category, color in color_map.items()
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(len(handles), 8),
        frameon=False,
    )


def output_name(line_index: int, image_path_value: str, prediction_format: str) -> str:
    image_name = Path(image_path_value).name
    if prediction_format == "point":
        return f"res_{line_index:05d}_{Path(image_name).stem}.jpg"
    return f"res_{line_index}_{image_name}"


def render_record(
    image: Image.Image,
    image_path_value: str,
    gt_data: Mapping,
    prediction_data: Mapping,
    prediction_format: str,
    label: str,
    point_size: float,
) -> Tuple[plt.Figure, int, int]:
    color_map = category_color_map(gt_data, prediction_data)
    malformed_points = 0
    out_of_bounds_points = 0

    if prediction_format == "point":
        figure, axes = plt.subplots(1, 2, figsize=point_figure_size(image))
        figure.suptitle(image_path_value, fontsize=16, y=0.985)
        _, gt_bad, gt_outside = draw_points(
            axes[0], image, gt_data, "Ground Truth", color_map, point_size
        )
        _, pred_bad, pred_outside = draw_points(
            axes[1], image, prediction_data, "Predictions", color_map, point_size
        )
        malformed_points = gt_bad + pred_bad
        out_of_bounds_points = gt_outside + pred_outside
        add_point_legend(figure, color_map)
        figure.tight_layout(rect=(0, 0, 1, 0.91))
        return figure, malformed_points, out_of_bounds_points

    figure, axes = plt.subplots(1, 2, figsize=(20, 10))
    if prediction_format == "polygon":
        draw_polygons(axes[0], image, gt_data, f"GT: {image_path_value}", color_map)
        draw_polygons(
            axes[1],
            image,
            prediction_data,
            f"{label}: {image_path_value}",
            color_map,
        )
    else:
        draw_boxes(
            axes[0],
            image,
            gt_data,
            f"Ground Truth: {image_path_value}",
            color_map,
        )
        draw_boxes(
            axes[1],
            image,
            prediction_data,
            f"Predictions: {image_path_value}",
            color_map,
        )
    figure.tight_layout()
    return figure, malformed_points, out_of_bounds_points


def validate_options(
    jsonl_path: Path,
    image_folder: Path,
    start_index: int,
    max_images: Optional[int],
    point_size: float,
    dpi: int,
    log_every: Optional[int],
    prediction_format: str,
) -> None:
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")
    if not image_folder.is_dir():
        raise NotADirectoryError(f"Image folder does not exist: {image_folder}")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if log_every is not None and log_every < 0:
        raise ValueError("log_every must be non-negative")
    if prediction_format not in ("auto",) + FORMATS:
        raise ValueError(
            f"prediction_format must be one of auto, {', '.join(FORMATS)}"
        )


def visualize_results(
    jsonl_path,
    image_folder,
    output_folder,
    label: str = "Pred-1",
    start_index: int = 0,
    max_images: Optional[int] = None,
    point_size: float = 18.0,
    dpi: int = 150,
    log_every: Optional[int] = None,
    prediction_format: str = "auto",
) -> None:
    jsonl_path = Path(jsonl_path)
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    validate_options(
        jsonl_path,
        image_folder,
        start_index,
        max_images,
        point_size,
        dpi,
        log_every,
        prediction_format,
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    fallback_format = None
    if prediction_format == "auto":
        fallback_format = detect_file_fallback(jsonl_path)
        if fallback_format is not None:
            print(f"Detected file format: {fallback_format}")
        else:
            print("Detected a mixed or ambiguous file; inferring each record separately")
    else:
        print(f"Using requested format: {prediction_format}")

    saved_count = 0
    missing_image_count = 0
    unknown_format_count = 0
    malformed_point_count = 0
    out_of_bounds_count = 0
    format_counts = Counter()

    for line_index, record in iter_jsonl(jsonl_path):
        if line_index < start_index:
            continue
        if max_images is not None and saved_count >= max_images:
            break

        record_format = prediction_format
        if prediction_format == "auto":
            record_format = infer_record_format(record) or fallback_format
        if record_format not in FORMATS:
            print(
                f"Skipping line {line_index + 1}: cannot determine whether its "
                "coordinates are box, polygon, or point data"
            )
            unknown_format_count += 1
            continue

        image_path_value = str(
            record.get("image_path")
            or record.get("image")
            or record.get("image_name")
            or ""
        ).strip()
        if not image_path_value:
            print(f"Skipping line {line_index + 1}: image_path is missing")
            missing_image_count += 1
            continue

        image_path = resolve_image_path(image_folder, image_path_value)
        if image_path is None:
            print(f"Skipping line {line_index + 1}: cannot find {image_path_value}")
            missing_image_count += 1
            continue

        gt_data = normalized_data(record, "gt", line_index + 1)
        prediction_data = normalized_data(
            record,
            "extracted_predictions",
            line_index + 1,
        )
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            figure, malformed_points, outside_points = render_record(
                image,
                image_path_value,
                gt_data,
                prediction_data,
                record_format,
                label,
                point_size,
            )
            save_name = output_name(line_index, image_path_value, record_format)
            figure.savefig(
                output_folder / save_name,
                bbox_inches="tight",
                dpi=dpi,
                facecolor="white",
            )
            plt.close(figure)

        malformed_point_count += malformed_points
        out_of_bounds_count += outside_points
        format_counts[record_format] += 1
        saved_count += 1

        interval = log_every
        if interval is None:
            interval = 20 if record_format == "point" else 1
        if interval > 0 and saved_count % interval == 0:
            print(f"Saved {saved_count} images; latest: {save_name} ({record_format})")

    format_summary = ", ".join(
        f"{name}={format_counts[name]}" for name in FORMATS if format_counts[name]
    ) or "none"
    print(
        "Done: saved {} ({}), missing images {}, unknown formats {}, malformed "
        "points {}, out-of-bounds points {}. Output: {}".format(
            saved_count,
            format_summary,
            missing_image_count,
            unknown_format_count,
            malformed_point_count,
            out_of_bounds_count,
            output_folder,
        )
    )


def main() -> None:
    args = parse_args()
    visualize_results(
        jsonl_path=args.jsonl_path,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        label=args.label,
        start_index=args.start_index,
        max_images=args.max_images,
        point_size=args.point_size,
        dpi=args.dpi,
        log_every=args.log_every,
        prediction_format=args.prediction_format,
    )


if __name__ == "__main__":
    main()
