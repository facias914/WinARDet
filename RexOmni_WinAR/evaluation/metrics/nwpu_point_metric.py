#!/usr/bin/env python3
"""Evaluate NWPU-Crowd point localization from Rex inference JSONL output.

The localization matching and aggregation follow the official NWPU-Crowd
evaluation code. Ground-truth sigma/level metadata is recovered exactly from
the official val_gt_loc.txt through source_annotation_index stored in the
NPUW_Point COCO annotations.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import spatial as scipy_spatial


NUM_LEVELS = 6


@dataclass
class GroundTruthSample:
    image_path: str
    points: np.ndarray
    sigma_small: np.ndarray
    sigma_large: np.ndarray
    levels: np.ndarray


@dataclass
class MatchCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_by_level: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LEVELS, dtype=np.int64)
    )
    fn_by_level: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LEVELS, dtype=np.int64)
    )

    def add(self, other: "MatchCounts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tp_by_level += other.tp_by_level
        self.fn_by_level += other.fn_by_level


@dataclass
class PredictionStats:
    records: int = 0
    duplicate_records: int = 0
    direct_points: int = 0
    boxes_converted_to_points: int = 0
    ignored_values: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Rex JSONL predictions with the official NWPU-Crowd "
            "point-localization matching and counting metrics."
        )
    )
    parser.add_argument(
        "--pred-jsonl",
        "--pred_jsonl",
        dest="pred_jsonl",
        type=Path,
        required=True,
        help="Inference JSONL produced by inference_detection_ddp.py.",
    )
    parser.add_argument(
        "--gt-jsonl",
        "--gt_jsonl",
        dest="gt_jsonl",
        type=Path,
        required=True,
        help="NPUW_Point annotations_rex.json (JSONL).",
    )
    parser.add_argument(
        "--coco-json",
        "--coco_json",
        dest="coco_json",
        type=Path,
        required=True,
        help="NPUW_Point annotations_coco.json with source annotation metadata.",
    )
    parser.add_argument(
        "--official-gt",
        "--official_gt",
        dest="official_gt",
        type=Path,
        required=True,
        help="Official NWPU-Crowd eval/val_gt_loc.txt.",
    )
    parser.add_argument(
        "--output-json",
        "--output_json",
        dest="output_json",
        type=Path,
        help="Optional path for a machine-readable metric summary.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["auto", "point", "box-center"],
        default="auto",
        help=(
            "How to interpret extracted_predictions coordinates. auto accepts "
            "both [x,y] and [x1,y1,x2,y2]; boxes are converted to centers."
        ),
    )
    parser.add_argument(
        "--category",
        help=(
            "Only use predictions under this category key (case-insensitive). "
            "By default all keys are flattened, matching the category-free official task."
        ),
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["first", "last", "error"],
        default="first",
        help="How to handle duplicate image_path records caused by DDP sampler padding.",
    )
    parser.add_argument(
        "--allow-extra-predictions",
        action="store_true",
        help="Ignore prediction image paths absent from the GT JSONL instead of failing.",
    )
    parser.add_argument(
        "--source-point-tolerance",
        type=float,
        default=1.0,
        help="Maximum source-point rounding error allowed when mapping official metadata.",
    )
    parser.add_argument(
        "--gt-point-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance when validating annotations_rex points against COCO point_xy.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress periodic evaluation progress.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} does not exist: {resolved}")
    return resolved


def image_key(path: object) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"Invalid image_path: {path!r}")
    return path.strip().replace("\\", "/").rsplit("/", 1)[-1]


def as_point(value: object, context: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context} must be [x, y], got {value!r}")
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        raise ValueError(f"{context} must contain numeric coordinates, got {value!r}")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must contain numeric coordinates, got {value!r}"
        ) from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{context} contains non-finite coordinates: {value!r}")
    return x, y


def points_array(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def iter_jsonl(path: Path, label: str) -> Iterable[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {label} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{label} line {line_number} must contain a JSON object."
                )
            yield line_number, record


def load_gt_jsonl(path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    order = []
    gt_by_image = {}
    for line_number, record in iter_jsonl(path, "GT JSONL"):
        try:
            key = image_key(record.get("image_path"))
        except ValueError as exc:
            raise ValueError(f"GT JSONL line {line_number}: {exc}") from exc
        if key in gt_by_image:
            raise ValueError(f"Duplicate GT image_path basename {key!r} at line {line_number}.")

        gt = record.get("gt")
        if not isinstance(gt, dict):
            raise ValueError(f"GT JSONL line {line_number}: gt must be a dictionary.")
        sample_points = []
        for category, values in gt.items():
            if not isinstance(values, list):
                raise ValueError(
                    f"GT JSONL line {line_number}: gt[{category!r}] must be a list."
                )
            for point_index, value in enumerate(values):
                sample_points.append(
                    as_point(
                        value,
                        f"GT JSONL line {line_number}, {category}[{point_index}]",
                    )
                )
        order.append(key)
        gt_by_image[key] = points_array(sample_points)
    if not order:
        raise ValueError(f"No GT records found in {path}")
    return order, gt_by_image


def load_official_gt(path: Path) -> Dict[str, np.ndarray]:
    official = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            try:
                values = [int(value) for value in fields]
            except ValueError as exc:
                raise ValueError(
                    f"Official GT line {line_number} contains a non-integer value."
                ) from exc
            if len(values) < 2:
                raise ValueError(f"Official GT line {line_number} is malformed.")
            source_id, count = str(values[0]), values[1]
            payload = values[2:]
            if count < 0 or len(payload) != count * 5:
                raise ValueError(
                    f"Official GT line {line_number}: count={count}, "
                    f"but {len(payload)} payload values were found."
                )
            if source_id in official:
                raise ValueError(f"Duplicate official source image id {source_id!r}.")
            rows = np.asarray(payload, dtype=np.float64).reshape(count, 5)
            if count:
                levels = rows[:, 4]
                if np.any(levels < 0) or np.any(levels >= NUM_LEVELS):
                    raise ValueError(
                        f"Official GT source image {source_id} contains levels outside 0..5."
                    )
            official[source_id] = rows
    if not official:
        raise ValueError(f"No official GT records found in {path}")
    return official


def rounded_point_counter(points: np.ndarray, tolerance: float) -> Counter:
    if tolerance <= 0:
        return Counter((float(x), float(y)) for x, y in points)
    decimals = max(0, int(math.ceil(-math.log10(tolerance))))
    return Counter((round(float(x), decimals), round(float(y), decimals)) for x, y in points)


def build_ground_truth(
    gt_order: Sequence[str],
    gt_jsonl_points: Dict[str, np.ndarray],
    coco_path: Path,
    official: Dict[str, np.ndarray],
    source_point_tolerance: float,
    gt_point_tolerance: float,
) -> Tuple[List[GroundTruthSample], Dict[str, object]]:
    with coco_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO JSON must contain images and annotations lists.")

    image_id_by_key = {}
    image_key_by_id = {}
    for image in images:
        key = image_key(image.get("file_name"))
        image_id = image.get("id")
        if image_id is None:
            raise ValueError(f"COCO image {key!r} is missing id.")
        if key in image_id_by_key:
            raise ValueError(f"Duplicate COCO image basename {key!r}.")
        if image_id in image_key_by_id:
            raise ValueError(f"Duplicate COCO image id {image_id!r}.")
        image_id_by_key[key] = image_id
        image_key_by_id[image_id] = key

    missing_coco_images = [key for key in gt_order if key not in image_id_by_key]
    if missing_coco_images:
        raise ValueError(
            f"{len(missing_coco_images)} GT images are absent from COCO JSON; "
            f"examples: {missing_coco_images[:5]}"
        )

    metadata_by_image = defaultdict(list)
    skipped_outside_crop = 0
    source_point_max_error = 0.0
    for annotation in annotations:
        if annotation.get("point_in_crop") is False:
            skipped_outside_crop += 1
            continue
        image_id = annotation.get("image_id")
        if image_id not in image_key_by_id:
            raise ValueError(
                f"COCO annotation id={annotation.get('id')!r} references "
                f"unknown image_id={image_id!r}."
            )
        source_id = str(annotation.get("source_image_id"))
        source_index = annotation.get("source_annotation_index")
        if source_id not in official:
            raise ValueError(
                f"COCO annotation id={annotation.get('id')!r} references "
                f"source image {source_id!r}, absent from official GT."
            )
        if not isinstance(source_index, int) or not (
            0 <= source_index < len(official[source_id])
        ):
            raise ValueError(
                f"COCO annotation id={annotation.get('id')!r} has invalid "
                f"source_annotation_index={source_index!r}."
            )

        crop_point = as_point(
            annotation.get("point_xy"),
            f"COCO annotation id={annotation.get('id')!r} point_xy",
        )
        source_point = as_point(
            annotation.get("source_point_xy"),
            f"COCO annotation id={annotation.get('id')!r} source_point_xy",
        )
        official_row = official[source_id][source_index]
        source_error = math.hypot(
            source_point[0] - official_row[0], source_point[1] - official_row[1]
        )
        source_point_max_error = max(source_point_max_error, source_error)
        if source_error > source_point_tolerance:
            raise ValueError(
                f"COCO annotation id={annotation.get('id')!r} differs from "
                f"official source point by {source_error:.6f} pixels; tolerance is "
                f"{source_point_tolerance}."
            )
        metadata_by_image[image_id].append(
            (
                crop_point[0],
                crop_point[1],
                float(official_row[2]),
                float(official_row[3]),
                int(official_row[4]),
            )
        )

    samples = []
    total_points = 0
    for key in gt_order:
        image_id = image_id_by_key[key]
        metadata = metadata_by_image.get(image_id, [])
        if metadata:
            metadata_array = np.asarray(metadata, dtype=np.float64).reshape(-1, 5)
        else:
            metadata_array = np.empty((0, 5), dtype=np.float64)
        canonical_points = metadata_array[:, :2]
        jsonl_points = gt_jsonl_points[key]
        if canonical_points.shape != jsonl_points.shape or rounded_point_counter(
            canonical_points, gt_point_tolerance
        ) != rounded_point_counter(jsonl_points, gt_point_tolerance):
            raise ValueError(
                f"GT point mismatch for {key!r}: annotations_rex has "
                f"{len(jsonl_points)}, COCO has {len(canonical_points)} valid crop points."
            )
        samples.append(
            GroundTruthSample(
                image_path=key,
                points=canonical_points.copy(),
                sigma_small=metadata_array[:, 2].copy(),
                sigma_large=metadata_array[:, 3].copy(),
                levels=metadata_array[:, 4].astype(np.int64, copy=True),
            )
        )
        total_points += len(canonical_points)

    extra_coco_images = set(image_id_by_key) - set(gt_order)
    stats = {
        "samples": len(samples),
        "points": total_points,
        "empty_samples": sum(len(sample.points) == 0 for sample in samples),
        "outside_crop_annotations_skipped": skipped_outside_crop,
        "source_point_max_error": source_point_max_error,
        "coco_images_not_in_gt_jsonl": len(extra_coco_images),
    }
    return samples, stats


def prediction_coordinate_to_point(
    value: object, mode: str
) -> Optional[Tuple[float, float, str]]:
    if value is None or (isinstance(value, str) and value.casefold() == "none"):
        return None
    if not isinstance(value, (list, tuple)):
        return None
    try:
        coordinates = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in coordinates):
        return None

    if len(coordinates) == 2 and mode in ("auto", "point"):
        return coordinates[0], coordinates[1], "point"
    if len(coordinates) == 4 and mode in ("auto", "box-center"):
        x1, y1, x2, y2 = coordinates
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, "box"
    return None


def extract_prediction_points(
    record: Dict,
    mode: str,
    category_filter: Optional[str],
    stats: PredictionStats,
) -> np.ndarray:
    extracted = record.get("extracted_predictions")
    if extracted is None:
        return points_array([])
    if not isinstance(extracted, dict):
        stats.ignored_values += 1
        return points_array([])

    points = []
    wanted = category_filter.casefold() if category_filter else None
    for category, values in extracted.items():
        if wanted is not None and str(category).casefold() != wanted:
            continue
        if not isinstance(values, list):
            stats.ignored_values += 1
            continue
        for value in values:
            parsed = prediction_coordinate_to_point(value, mode)
            if parsed is None:
                if value is not None and not (
                    isinstance(value, str) and value.casefold() == "none"
                ):
                    stats.ignored_values += 1
                continue
            x, y, source_type = parsed
            points.append((x, y))
            if source_type == "point":
                stats.direct_points += 1
            else:
                stats.boxes_converted_to_points += 1
    return points_array(points)


def load_predictions(
    path: Path,
    mode: str,
    category_filter: Optional[str],
    duplicate_policy: str,
) -> Tuple[Dict[str, np.ndarray], PredictionStats]:
    predictions = {}
    stats = PredictionStats()
    for line_number, record in iter_jsonl(path, "prediction JSONL"):
        stats.records += 1
        try:
            key = image_key(record.get("image_path"))
        except ValueError as exc:
            raise ValueError(f"Prediction JSONL line {line_number}: {exc}") from exc
        if key in predictions:
            stats.duplicate_records += 1
            if duplicate_policy == "error":
                raise ValueError(
                    f"Duplicate prediction image_path basename {key!r} at line {line_number}."
                )
            if duplicate_policy == "first":
                continue
        predictions[key] = extract_prediction_points(
            record, mode, category_filter, stats
        )
    return predictions, stats


def maximum_bipartite_assignment(match_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return matched prediction/GT masks using the official DFS augmenting order."""
    pred_count, gt_count = match_matrix.shape
    required_recursion_limit = pred_count + gt_count + 100
    if sys.getrecursionlimit() < required_recursion_limit:
        sys.setrecursionlimit(required_recursion_limit)
    graph = [np.flatnonzero(match_matrix[index]).tolist() for index in range(pred_count)]
    gt_to_pred = np.full(gt_count, -1, dtype=np.int64)

    def dfs(pred_index: int, visited: np.ndarray) -> bool:
        for gt_index in graph[pred_index]:
            if visited[gt_index]:
                continue
            visited[gt_index] = True
            previous_pred = int(gt_to_pred[gt_index])
            if previous_pred == -1 or dfs(previous_pred, visited):
                gt_to_pred[gt_index] = pred_index
                return True
        return False

    for pred_index in range(pred_count):
        dfs(pred_index, np.zeros(gt_count, dtype=bool))

    matched_gt = gt_to_pred >= 0
    matched_pred = np.zeros(pred_count, dtype=bool)
    matched_pred[gt_to_pred[matched_gt]] = True
    return matched_pred, matched_gt


def match_at_sigma(
    distance_matrix: np.ndarray,
    sigma: np.ndarray,
    levels: np.ndarray,
) -> MatchCounts:
    pred_count, gt_count = distance_matrix.shape
    if gt_count == 0:
        return MatchCounts(fp=pred_count)
    if pred_count == 0:
        fn_by_level = np.bincount(levels, minlength=NUM_LEVELS).astype(np.int64)
        return MatchCounts(fn=gt_count, fn_by_level=fn_by_level)

    match_matrix = distance_matrix <= sigma[np.newaxis, :]
    matched_pred, matched_gt = maximum_bipartite_assignment(match_matrix)
    tp = int(matched_pred.sum())
    tp_by_level = np.bincount(
        levels[matched_gt], minlength=NUM_LEVELS
    ).astype(np.int64)
    fn_by_level = np.bincount(
        levels[~matched_gt], minlength=NUM_LEVELS
    ).astype(np.int64)
    return MatchCounts(
        tp=tp,
        fp=pred_count - tp,
        fn=gt_count - tp,
        tp_by_level=tp_by_level,
        fn_by_level=fn_by_level,
    )


def evaluate_sample(
    prediction_points: np.ndarray,
    sample: GroundTruthSample,
) -> Tuple[MatchCounts, MatchCounts]:
    pred_count = len(prediction_points)
    gt_count = len(sample.points)
    if pred_count == 0 or gt_count == 0:
        empty_distances = np.empty((pred_count, gt_count), dtype=np.float64)
        return (
            match_at_sigma(empty_distances, sample.sigma_small, sample.levels),
            match_at_sigma(empty_distances, sample.sigma_large, sample.levels),
        )
    distances = scipy_spatial.distance_matrix(prediction_points, sample.points, p=2)
    return (
        match_at_sigma(distances, sample.sigma_small, sample.levels),
        match_at_sigma(distances, sample.sigma_large, sample.levels),
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def localization_summary(counts: MatchCounts) -> Dict[str, object]:
    precision = safe_divide(counts.tp, counts.tp + counts.fp)
    recall = safe_divide(counts.tp, counts.tp + counts.fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    level_denominator = counts.tp_by_level + counts.fn_by_level
    level_recall = np.divide(
        counts.tp_by_level,
        level_denominator,
        out=np.zeros(NUM_LEVELS, dtype=np.float64),
        where=level_denominator != 0,
    )
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_by_level": counts.tp_by_level.tolist(),
        "fn_by_level": counts.fn_by_level.tolist(),
        "recall_by_level": level_recall.tolist(),
        "mean_recall_by_level": float(level_recall.mean()),
    }


def evaluate_dataset(
    samples: Sequence[GroundTruthSample],
    predictions: Dict[str, np.ndarray],
    quiet: bool,
) -> Dict[str, object]:
    small_total = MatchCounts()
    large_total = MatchCounts()
    absolute_errors = []
    squared_errors = []
    normalized_absolute_errors = []
    total_predictions = 0

    for index, sample in enumerate(samples, start=1):
        prediction_points = predictions.get(sample.image_path, points_array([]))
        small, large = evaluate_sample(prediction_points, sample)
        small_total.add(small)
        large_total.add(large)

        gt_count = len(sample.points)
        pred_count = len(prediction_points)
        total_predictions += pred_count
        error = abs(gt_count - pred_count)
        absolute_errors.append(error)
        squared_errors.append(error * error)
        if gt_count != 0:
            normalized_absolute_errors.append(error / gt_count)

        if not quiet and (index % 100 == 0 or index == len(samples)):
            print(f"Evaluated {index}/{len(samples)} samples", flush=True)

    return {
        "localization": {
            "small": localization_summary(small_total),
            "large": localization_summary(large_total),
        },
        "counting": {
            "mae": float(np.mean(absolute_errors)) if absolute_errors else 0.0,
            "rmse": float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0,
            "nae": (
                float(np.mean(normalized_absolute_errors))
                if normalized_absolute_errors
                else 0.0
            ),
            "samples_used_for_nae": len(normalized_absolute_errors),
        },
        "totals": {
            "samples": len(samples),
            "gt_points": sum(len(sample.points) for sample in samples),
            "predicted_points": total_predictions,
            "samples_without_gt": sum(len(sample.points) == 0 for sample in samples),
            "samples_without_predictions": sum(
                len(predictions.get(sample.image_path, [])) == 0 for sample in samples
            ),
        },
    }


def format_array(values: Sequence[float]) -> str:
    return "[" + " ".join(f"{value:.8f}" for value in values) + "]"


def print_results(results: Dict[str, object]) -> None:
    small = results["localization"]["small"]
    large = results["localization"]["large"]
    counting = results["counting"]

    print("-----Localization performance-----")
    print(f"AP_small: {small['precision']:.10f}")
    print(f"AR_small: {small['recall']:.10f}")
    print(f"F1m_small: {small['f1']:.10f}")
    print(f"AR_small_category: {format_array(small['recall_by_level'])}")
    print(f"    avg: {small['mean_recall_by_level']:.10f}")
    print(f"AP_large: {large['precision']:.10f}")
    print(f"AR_large: {large['recall']:.10f}")
    print(f"F1m_large: {large['f1']:.10f}")
    print(f"AR_large_category: {format_array(large['recall_by_level'])}")
    print(f"    avg: {large['mean_recall_by_level']:.10f}")
    print("-----Counting performance-----")
    print(f"MAE: {counting['mae']:.10f}")
    print(f"MSE: {counting['rmse']:.10f}")
    print(f"NAE: {counting['nae']:.10f}")


def main() -> None:
    args = parse_args()
    if args.source_point_tolerance < 0 or args.gt_point_tolerance < 0:
        raise SystemExit("Point tolerances must be non-negative.")

    pred_path = require_file(args.pred_jsonl, "Prediction JSONL")
    gt_path = require_file(args.gt_jsonl, "GT JSONL")
    coco_path = require_file(args.coco_json, "COCO JSON")
    official_path = require_file(args.official_gt, "Official localization GT")
    start_time = time.time()

    try:
        gt_order, gt_jsonl_points = load_gt_jsonl(gt_path)
        official = load_official_gt(official_path)
        samples, gt_stats = build_ground_truth(
            gt_order,
            gt_jsonl_points,
            coco_path,
            official,
            args.source_point_tolerance,
            args.gt_point_tolerance,
        )
        predictions, prediction_stats = load_predictions(
            pred_path,
            args.prediction_mode,
            args.category,
            args.duplicate_policy,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Input validation failed: {exc}") from exc

    gt_keys = set(gt_order)
    extra_predictions = sorted(set(predictions) - gt_keys)
    if extra_predictions and not args.allow_extra_predictions:
        raise SystemExit(
            f"Found {len(extra_predictions)} prediction image paths absent from GT; "
            f"examples: {extra_predictions[:5]}. Pass --allow-extra-predictions to ignore them."
        )
    if extra_predictions:
        for key in extra_predictions:
            predictions.pop(key, None)

    missing_predictions = len(gt_keys - set(predictions))
    if prediction_stats.duplicate_records:
        print(
            f"Warning: resolved {prediction_stats.duplicate_records} duplicate prediction "
            f"records with policy={args.duplicate_policy!r}."
        )
    if prediction_stats.ignored_values:
        print(
            f"Warning: ignored {prediction_stats.ignored_values} malformed or "
            f"mode-incompatible prediction values."
        )
    if missing_predictions:
        print(
            f"Warning: {missing_predictions} GT samples have no prediction record; "
            "they are evaluated as empty predictions."
        )

    results = evaluate_dataset(samples, predictions, args.quiet)
    results["inputs"] = {
        "pred_jsonl": str(pred_path),
        "gt_jsonl": str(gt_path),
        "coco_json": str(coco_path),
        "official_gt": str(official_path),
        "prediction_mode": args.prediction_mode,
        "category_filter": args.category,
    }
    results["ground_truth_validation"] = gt_stats
    results["prediction_input"] = {
        "records": prediction_stats.records,
        "duplicates_resolved": prediction_stats.duplicate_records,
        "direct_points": prediction_stats.direct_points,
        "boxes_converted_to_points": prediction_stats.boxes_converted_to_points,
        "ignored_values": prediction_stats.ignored_values,
        "missing_gt_samples": missing_predictions,
        "extra_records_ignored": len(extra_predictions),
    }
    results["elapsed_seconds"] = time.time() - start_time

    print_results(results)
    print("-----Input summary-----")
    print(
        f"Samples: {results['totals']['samples']}, "
        f"GT points: {results['totals']['gt_points']}, "
        f"predicted points: {results['totals']['predicted_points']}"
    )
    print(
        f"Direct point predictions: {prediction_stats.direct_points}, "
        f"box centers: {prediction_stats.boxes_converted_to_points}"
    )

    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Detailed metrics written to: {output_path}")


if __name__ == "__main__":
    main()
