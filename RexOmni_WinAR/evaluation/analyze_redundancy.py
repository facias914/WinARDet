#!/usr/bin/env python3
"""Analyze exact duplicate-box redundancy and over-prediction in prediction JSONL.

Definitions per image:
    N_pred = number of predicted boxes across all categories
    N_gt   = number of GT boxes across all categories
    N_dup  = N_pred - number of unique predicted boxes
    R_dup  = N_dup / N_pred
    R_over = max(0, N_pred - N_gt) / N_gt

Horizontal and rotated polygon boxes are detected automatically. Duplicate
boxes are counted by exact category + full-coordinate equality after rounding
coordinates to --round-digits decimal places.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT = "/home/u1120250365/Locany/work_dirs/dense/sft_winar/Results/rs-dense200/hybrid_compressed/eval_results.jsonl"
BOX_COORDINATE_COUNTS = {"horizontal": 4, "rotated": 8}


BoxKey = Tuple[str, Tuple[float, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-image and aggregate redundancy metrics for detection JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Prediction JSONL path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional path to save per-image metrics as CSV.",
    )
    parser.add_argument(
        "--round_digits",
        type=int,
        default=6,
        help="Round box coordinates before exact duplicate matching. Default: 6.",
    )
    parser.add_argument(
        "--box_format",
        "--format",
        choices=("auto", "horizontal", "rotated"),
        default="auto",
        help=(
            "Box format. Default: auto (detect from task metadata and coordinate "
            "lengths). Use horizontal for xyxy or rotated for 4-point polygons."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Print top-k images by R_dup and R_over. Default: 10.",
    )
    return parser.parse_args()


def _is_box_like(value: Any, coordinate_count: int = 4) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < coordinate_count:
        return False
    try:
        [float(x) for x in value[:coordinate_count]]
    except (TypeError, ValueError):
        return False
    return True


def _iter_boxes(
    category_to_boxes: Any, coordinate_count: int = 4
) -> Iterable[Tuple[str, List[float]]]:
    """Yield category and coordinates from a category -> boxes dict.

    Horizontal dense detection files use:
        {"category": [[x0, y0, x1, y1], ...]}

    Rotated detection files use four polygon vertices:
        {"category": [[x0, y0, x1, y1, x2, y2, x3, y3], ...]}

    This function is intentionally conservative: unsupported entries are skipped
    rather than guessed.
    """
    if not isinstance(category_to_boxes, dict):
        return

    for category, boxes in category_to_boxes.items():
        if boxes is None:
            continue
        if _is_box_like(boxes, coordinate_count):
            yield str(category), [float(x) for x in boxes[:coordinate_count]]
            continue
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if _is_box_like(box, coordinate_count):
                yield str(category), [float(x) for x in box[:coordinate_count]]


def _get_gt(entry: Dict[str, Any]) -> Any:
    gt = entry.get("gt", {})
    if isinstance(gt, dict) and isinstance(gt.get("gt_mask"), dict):
        return gt["gt_mask"]
    return gt


def _metadata_box_format(entry: Dict[str, Any]) -> Optional[str]:
    task_name = str(entry.get("task_name", "")).lower()
    question = str(entry.get("question", "")).lower()

    if "polygon" in task_name:
        return "rotated"

    rotated_phrases = (
        "polygon",
        "rotated bounding box",
        "oriented bounding box",
        "four-point",
        "x2, y2",
        "x3, y3",
    )
    if any(phrase in question for phrase in rotated_phrases):
        return "rotated"

    if "bounding box" in question or "x1, y1" in question:
        return "horizontal"
    if "detection" in task_name or task_name == "common_object_detection":
        return "horizontal"
    return None


def _iter_coordinate_lengths(category_to_boxes: Any) -> Iterable[int]:
    if not isinstance(category_to_boxes, dict):
        return

    for boxes in category_to_boxes.values():
        if boxes is None:
            continue
        if _is_box_like(boxes):
            yield len(boxes)
            continue
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if _is_box_like(box):
                yield len(box)


def infer_entry_box_format(entry: Dict[str, Any]) -> Optional[str]:
    metadata_hint = _metadata_box_format(entry)
    if metadata_hint is not None:
        return metadata_hint

    lengths = set(_iter_coordinate_lengths(entry.get("extracted_predictions", {})))
    lengths.update(_iter_coordinate_lengths(_get_gt(entry)))
    if any(length >= 8 for length in lengths):
        return "rotated"
    if any(length >= 4 for length in lengths):
        return "horizontal"
    return None


def detect_file_box_format(
    path: Path, requested_format: str, sample_limit: int = 200
) -> str:
    if requested_format != "auto":
        return requested_format

    detected = set()
    inspected = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            box_format = infer_entry_box_format(entry)
            if box_format is not None:
                detected.add(box_format)
            inspected += 1
            if len(detected) > 1 or inspected >= sample_limit:
                break

    if len(detected) > 1:
        raise ValueError(
            "Both horizontal and rotated records were detected. "
            "Use --box_format to select the intended format."
        )
    if detected:
        return detected.pop()

    print(
        "Warning: could not detect box format from the first "
        f"{inspected} valid records; falling back to horizontal."
    )
    return "horizontal"


def _box_key(category: str, box: List[float], round_digits: int) -> BoxKey:
    return category, tuple(round(float(x), round_digits) for x in box)


def analyze_entry(
    entry: Dict[str, Any], round_digits: int, box_format: str = "horizontal"
) -> Dict[str, Any]:
    coordinate_count = BOX_COORDINATE_COUNTS[box_format]
    pred_boxes = list(
        _iter_boxes(entry.get("extracted_predictions", {}), coordinate_count)
    )
    gt_boxes = list(_iter_boxes(_get_gt(entry), coordinate_count))

    pred_keys = [_box_key(category, box, round_digits) for category, box in pred_boxes]
    n_pred = len(pred_keys)
    n_unique_pred = len(set(pred_keys))
    n_dup = n_pred - n_unique_pred
    n_gt = len(gt_boxes)

    r_dup = (n_dup / n_pred) if n_pred > 0 else 0.0
    r_over = (max(0, n_pred - n_gt) / n_gt) if n_gt > 0 else None

    return {
        "image_path": entry.get("image_path", ""),
        "dataset_name": entry.get("dataset_name", ""),
        "n_pred": n_pred,
        "n_unique_pred": n_unique_pred,
        "n_dup": n_dup,
        "n_gt": n_gt,
        "r_dup": r_dup,
        "r_over": r_over,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "dataset_name",
        "n_pred",
        "n_unique_pred",
        "n_dup",
        "n_gt",
        "r_dup",
        "r_over",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    box_format = detect_file_box_format(input_path, args.box_format)

    rows: List[Dict[str, Any]] = []
    bad_lines = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(entry, dict):
                bad_lines += 1
                continue
            rows.append(analyze_entry(entry, args.round_digits, box_format))

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)

    r_dup_values = [float(row["r_dup"]) for row in rows]
    r_over_values = [float(row["r_over"]) for row in rows if row["r_over"] is not None]
    zero_gt_images = sum(1 for row in rows if row["r_over"] is None)

    total_pred = sum(int(row["n_pred"]) for row in rows)
    total_unique_pred = sum(int(row["n_unique_pred"]) for row in rows)
    total_dup = sum(int(row["n_dup"]) for row in rows)
    total_gt = sum(int(row["n_gt"]) for row in rows)

    micro_r_dup = (total_dup / total_pred) if total_pred > 0 else 0.0
    micro_r_over = (max(0, total_pred - total_gt) / total_gt) if total_gt > 0 else 0.0

    print(f"Input: {input_path}")
    print(
        f"Box format: {box_format} "
        f"({BOX_COORDINATE_COUNTS[box_format]} coordinates per box)"
    )
    print(f"Images: {len(rows)}")
    if bad_lines:
        print(f"Bad JSON lines skipped: {bad_lines}")
    if zero_gt_images:
        print(f"Images with N_gt=0 skipped for macro R_over: {zero_gt_images}")
    print()
    print("Macro average over images:")
    print(f"  mean R_dup  = {_mean(r_dup_values):.6f}")
    print(f"  mean R_over = {_mean(r_over_values):.6f}")
    print()
    print("Micro aggregate over all boxes:")
    print(f"  total N_pred        = {total_pred}")
    print(f"  total N_unique_pred = {total_unique_pred}")
    print(f"  total N_dup         = {total_dup}")
    print(f"  total N_gt          = {total_gt}")
    print(f"  global R_dup        = {micro_r_dup:.6f}")
    print(f"  global R_over       = {micro_r_over:.6f}")

    if args.output_csv is not None:
        print(f"\nPer-image CSV saved to: {args.output_csv}")

    topk = max(0, int(args.topk))
    if topk > 0 and rows:
        print(f"\nTop {topk} images by R_dup:")
        for row in sorted(rows, key=lambda x: (x["r_dup"], x["n_dup"]), reverse=True)[:topk]:
            print(
                "  "
                f"{_format_rate(row['r_dup'])} "
                f"N_dup={row['n_dup']} N_pred={row['n_pred']} N_gt={row['n_gt']} "
                f"{row['image_path']}"
            )

        print(f"\nTop {topk} images by R_over:")
        rows_with_over = [row for row in rows if row["r_over"] is not None]
        for row in sorted(rows_with_over, key=lambda x: (x["r_over"], x["n_pred"]), reverse=True)[:topk]:
            print(
                "  "
                f"{_format_rate(row['r_over'])} "
                f"N_pred={row['n_pred']} N_gt={row['n_gt']} N_dup={row['n_dup']} "
                f"{row['image_path']}"
            )


if __name__ == "__main__":
    main()
