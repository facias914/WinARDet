import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

IOU_THRESHOLD = 0.5
EPS = 1e-12
def normalize_category_name(name: str) -> str:
    return str(name).strip().lower().replace('_', ' ')


def load_jsonl(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _as_float_list(value) -> Optional[List[float]]:
    if value is None or value == 'None':
        return None
    if isinstance(value, dict):
        value = value.get('bbox') or value.get('poly') or value.get('polygon')
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def iter_polygons(items: Iterable) -> Iterable[List[float]]:
    if not isinstance(items, list):
        return
    for item in items:
        coords = _as_float_list(item)
        if coords is None:
            continue
        # Keep the same confidence behavior as other_metric.py: no score filtering.
        # If a 9th score exists, ignore it for recall/precision/F1 counting.
        if len(coords) >= 8:
            yield coords[:8]
        elif len(coords) == 4:
            x0, y0, x1, y1 = coords
            yield [x0, y0, x1, y0, x1, y1, x0, y1]


def _normalize_angle(angle: float) -> float:
    while angle < -math.pi / 2:
        angle += math.pi
    while angle >= math.pi / 2:
        angle -= math.pi
    return angle


def norm_angle(angle: float, angle_version: str) -> float:
    if angle_version == 'le135':
        return (angle + math.pi / 4) % math.pi - math.pi / 4
    if angle_version == 'le90':
        return (angle + math.pi / 2) % math.pi - math.pi / 2
    if angle_version == 'oc':
        while angle <= 0:
            angle += math.pi / 2
        while angle > math.pi / 2:
            angle -= math.pi / 2
        return angle
    raise ValueError(f'Unsupported angle_version: {angle_version}')


def _edge_poly_to_obb(poly: List[float], angle_version: str) -> Optional[List[float]]:
    pts = np.asarray(poly[:8], dtype=np.float64).reshape(4, 2)
    pt1, pt2, pt3, pt4 = pts
    edge1 = float(np.linalg.norm(pt1 - pt2))
    edge2 = float(np.linalg.norm(pt2 - pt3))
    if edge1 < 2 or edge2 < 2:
        return None

    width = max(edge1, edge2)
    height = min(edge1, edge2)
    if edge1 > edge2:
        angle = math.atan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    else:
        angle = math.atan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, angle_version)
    cx = float(pt1[0] + pt3[0]) / 2.0
    cy = float(pt1[1] + pt3[1]) / 2.0
    return [cx, cy, width, height, angle]


def _min_area_poly_to_obb(poly: List[float], angle_version: str) -> Optional[List[float]]:
    pts = np.asarray(poly[:8], dtype=np.float64).reshape(4, 2)
    best = None
    best_area = float('inf')
    for i in range(pts.shape[0]):
        p0 = pts[i]
        p1 = pts[(i + 1) % pts.shape[0]]
        edge = p1 - p0
        edge_len = float(np.hypot(edge[0], edge[1]))
        if edge_len <= EPS:
            continue
        angle = math.atan2(edge[1], edge[0])
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float64)
        rotated = pts @ rot.T
        min_xy = rotated.min(axis=0)
        max_xy = rotated.max(axis=0)
        width, height = max_xy - min_xy
        area = float(width * height)
        if area < best_area:
            center_rot = (min_xy + max_xy) / 2.0
            inv_rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
            center = center_rot @ inv_rot.T
            best = [float(center[0]), float(center[1]), float(width), float(height), angle]
            best_area = area

    if best is None:
        return None
    cx, cy, w, h, angle = best
    if w < 2 or h < 2:
        return None

    if angle_version in {'le90', 'le135'}:
        if w < h:
            w, h = h, w
            angle += math.pi / 2
        angle = norm_angle(angle, angle_version)
    elif angle_version == 'oc':
        while angle <= 0:
            angle += math.pi / 2
            w, h = h, w
        while angle > math.pi / 2:
            angle -= math.pi / 2
            w, h = h, w
    else:
        raise ValueError(f'Unsupported angle_version: {angle_version}')
    return [cx, cy, float(w), float(h), float(angle)]


def poly_to_obb(poly: List[float], angle_version: str) -> Optional[List[float]]:
    if angle_version in {'le90', 'le135'}:
        return _edge_poly_to_obb(poly, angle_version)
    if angle_version == 'oc':
        return _min_area_poly_to_obb(poly, angle_version)
    raise ValueError(f'Unsupported angle_version: {angle_version}')


def build_image_category_obbs(samples: List[dict], angle_version: str):
    gt_records = defaultdict(lambda: defaultdict(dict))
    pred_records = defaultdict(lambda: defaultdict(dict))
    skipped = {'gt': 0, 'pred': 0}

    for idx, sample in enumerate(tqdm(samples, desc='Converting polygons to rotated boxes')):
        image_id = sample.get('image_path') or sample.get('image') or sample.get('image_name') or str(idx)
        gt = sample.get('gt', {})
        pred = sample.get('extracted_predictions', {})

        if isinstance(gt, dict):
            for cat, boxes in gt.items():
                cat_key = normalize_category_name(cat)
                obbs = []
                for poly in iter_polygons(boxes):
                    obb = poly_to_obb(poly, angle_version)
                    if obb is None:
                        skipped['gt'] += 1
                        continue
                    obbs.append(obb)
                if obbs:
                    gt_records[image_id][cat_key] = obbs

        if isinstance(pred, dict):
            for cat, boxes in pred.items():
                cat_key = normalize_category_name(cat)
                obbs = []
                for poly in iter_polygons(boxes):
                    obb = poly_to_obb(poly, angle_version)
                    if obb is None:
                        skipped['pred'] += 1
                        continue
                    obbs.append(obb)
                if obbs:
                    pred_records[image_id][cat_key] = obbs

    return gt_records, pred_records, skipped


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _obb_to_poly(obb: List[float]) -> np.ndarray:
    cx, cy, w, h, angle = obb
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = w / 2.0
    dy = h / 2.0
    corners = np.asarray([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    rot = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return corners @ rot.T + np.asarray([cx, cy], dtype=np.float64)


def _inside(p: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    edge = edge_end - edge_start
    rel = p - edge_start
    return edge[0] * rel[1] - edge[1] * rel[0] >= -1e-9


def _line_intersection(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) <= EPS:
        return p2
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def _convex_intersection(poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
    output = poly1.copy()
    for i in range(poly2.shape[0]):
        clip_start = poly2[i]
        clip_end = poly2[(i + 1) % poly2.shape[0]]
        input_list = output
        if input_list.size == 0:
            break
        output_points = []
        s = input_list[-1]
        for e in input_list:
            if _inside(e, clip_start, clip_end):
                if not _inside(s, clip_start, clip_end):
                    output_points.append(_line_intersection(s, e, clip_start, clip_end))
                output_points.append(e)
            elif _inside(s, clip_start, clip_end):
                output_points.append(_line_intersection(s, e, clip_start, clip_end))
            s = e
        output = np.asarray(output_points, dtype=np.float64)
    return output


def _fallback_single_rotated_iou(obb1: List[float], obb2: List[float]) -> float:
    poly1 = _obb_to_poly(obb1)
    poly2 = _obb_to_poly(obb2)
    area1 = _polygon_area(poly1)
    area2 = _polygon_area(poly2)
    if area1 <= EPS or area2 <= EPS:
        return 0.0
    inter_poly = _convex_intersection(poly1, poly2)
    if inter_poly.size == 0 or inter_poly.shape[0] < 3:
        return 0.0
    inter = _polygon_area(inter_poly)
    union = area1 + area2 - inter
    return float(inter / union) if union > EPS else 0.0


def fallback_rotated_iou_matrix(pred_obbs: List[List[float]], gt_obbs: List[List[float]]) -> np.ndarray:
    mat = np.zeros((len(pred_obbs), len(gt_obbs)), dtype=np.float32)
    for i, pred in enumerate(pred_obbs):
        for j, gt in enumerate(gt_obbs):
            mat[i, j] = _fallback_single_rotated_iou(pred, gt)
    return mat


def rotated_iou_matrix(pred_obbs: List[List[float]], gt_obbs: List[List[float]], clockwise: bool, allow_fallback: bool) -> np.ndarray:
    if not pred_obbs or not gt_obbs:
        return np.zeros((len(pred_obbs), len(gt_obbs)), dtype=np.float32)
    try:
        from mmcv.ops import box_iou_rotated  # type: ignore
        pred_tensor = torch.as_tensor(pred_obbs, dtype=torch.float32)
        gt_tensor = torch.as_tensor(gt_obbs, dtype=torch.float32)
        if torch.cuda.is_available():
            pred_tensor = pred_tensor.cuda()
            gt_tensor = gt_tensor.cuda()
        with torch.no_grad():
            ious = box_iou_rotated(pred_tensor, gt_tensor, aligned=False, clockwise=clockwise)
        return ious.detach().cpu().numpy()
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                'mmcv.ops.box_iou_rotated is unavailable. Install mmcv-full/mmcv with ops, '
                'or pass --allow_fallback for a slower pure Python rotated IoU fallback.'
            ) from exc
        return fallback_rotated_iou_matrix(pred_obbs, gt_obbs)


def match_image_category(gt_obbs: List[List[float]], pred_obbs: List[List[float]], clockwise: bool, allow_fallback: bool) -> int:
    if not gt_obbs or not pred_obbs:
        return 0
    ious = rotated_iou_matrix(pred_obbs, gt_obbs, clockwise=clockwise, allow_fallback=allow_fallback)
    used_preds = set()
    matches = 0
    # Same greedy style as other_metric.py: each GT takes the best unused prediction.
    for gt_idx in range(len(gt_obbs)):
        best_iou = 0.0
        best_pred_idx = -1
        for pred_idx in range(len(pred_obbs)):
            if pred_idx in used_preds:
                continue
            iou = float(ious[pred_idx, gt_idx])
            if iou > best_iou and iou >= IOU_THRESHOLD:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx != -1:
            matches += 1
            used_preds.add(best_pred_idx)
    return matches


def evaluate(gt_records, pred_records, clockwise: bool, allow_fallback: bool) -> dict:
    image_ids = sorted(set(gt_records.keys()) | set(pred_records.keys()))
    total_gt = 0
    total_pred = 0
    total_matches = 0
    per_class = defaultdict(lambda: {'gt_count': 0, 'pred_count': 0, 'matches': 0})

    for image_id in tqdm(image_ids, desc='Evaluating rotated IoU@0.5'):
        gt_by_cat = gt_records.get(image_id, {})
        pred_by_cat = pred_records.get(image_id, {})
        cats = sorted(set(gt_by_cat.keys()) | set(pred_by_cat.keys()))
        for cat in cats:
            gt_obbs = gt_by_cat.get(cat, [])
            pred_obbs = pred_by_cat.get(cat, [])
            gt_count = len(gt_obbs)
            pred_count = len(pred_obbs)
            matches = match_image_category(gt_obbs, pred_obbs, clockwise=clockwise, allow_fallback=allow_fallback)

            total_gt += gt_count
            total_pred += pred_count
            total_matches += matches
            per_class[cat]['gt_count'] += gt_count
            per_class[cat]['pred_count'] += pred_count
            per_class[cat]['matches'] += matches

    micro_recall = total_matches / total_gt if total_gt > 0 else (1.0 if total_pred == 0 else 0.0)
    micro_precision = total_matches / total_pred if total_pred > 0 else (1.0 if total_gt == 0 else 0.0)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    per_class_out = {}
    for cat, stats in sorted(per_class.items()):
        cls_gt = stats['gt_count']
        cls_pred = stats['pred_count']
        cls_matches = stats['matches']
        cls_recall = cls_matches / cls_gt if cls_gt > 0 else (1.0 if cls_pred == 0 else 0.0)
        cls_precision = cls_matches / cls_pred if cls_pred > 0 else (1.0 if cls_gt == 0 else 0.0)
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if cls_precision + cls_recall > 0 else 0.0
        per_class_out[cat] = {
            'gt_count': int(cls_gt),
            'pred_count': int(cls_pred),
            'matches': int(cls_matches),
            'recall': float(cls_recall),
            'precision': float(cls_precision),
            'f1': float(cls_f1),
        }

    macro_classes = [
        stats for stats in per_class_out.values()
        if stats['gt_count'] > 0
    ]
    if not macro_classes:
        macro_classes = list(per_class_out.values())

    if macro_classes:
        precision = float(np.mean([stats['precision'] for stats in macro_classes]))
        recall = float(np.mean([stats['recall'] for stats in macro_classes]))
        f1 = float(np.mean([stats['f1'] for stats in macro_classes]))
    else:
        precision = 1.0 if total_gt == 0 and total_pred == 0 else 0.0
        recall = precision
        f1 = precision

    return {
        'iou_threshold': IOU_THRESHOLD,
        'aggregation': 'category_mean',
        'macro_class_selection': 'classes_with_gt_count_gt_0',
        'macro_class_count': int(len(macro_classes)),
        'gt_count': int(total_gt),
        'pred_count': int(total_pred),
        'matches': int(total_matches),
        'recall': float(recall),
        'precision': float(precision),
        'f1': float(f1),
        'micro_recall': float(micro_recall),
        'micro_precision': float(micro_precision),
        'micro_f1': float(micro_f1),
        'per_class': per_class_out,
    }


def print_results(result: dict) -> None:
    print('\n' + '=' * 90)
    print('FAST ROTATED IOU@0.5 METRIC'.center(90))
    print('=' * 90)
    print(f"GT: {result['gt_count']}  Pred: {result['pred_count']}  Matches: {result['matches']}")
    print(
        f"Category-mean ({result['macro_class_count']} classes)  "
        f"Precision: {result['precision']:.4f}  Recall: {result['recall']:.4f}  F1: {result['f1']:.4f}"
    )
    print(
        f"Micro(total boxes)             "
        f"Precision: {result['micro_precision']:.4f}  Recall: {result['micro_recall']:.4f}  F1: {result['micro_f1']:.4f}"
    )
    print('-' * 90)
    print(f"{'Class':<24} | {'GT':>8} | {'Pred':>8} | {'Match':>8} | {'P':>8} | {'R':>8} | {'F1':>8}")
    print('-' * 90)
    for cat, stats in result['per_class'].items():
        print(
            f"{cat:<24} | {stats['gt_count']:>8} | {stats['pred_count']:>8} | {stats['matches']:>8} | "
            f"{stats['precision']:>8.4f} | {stats['recall']:>8.4f} | {stats['f1']:>8.4f}"
        )
    print('=' * 90)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fast rotated IoU@0.5 recall/precision/F1 metric for Rex-Omni answer.jsonl.'
    )
    parser.add_argument('data_path', nargs='?', help='Path to answer.jsonl.')
    parser.add_argument('output_path', nargs='?', help='Path to save metric JSON.')
    parser.add_argument('--data_path', dest='data_path_flag', help='Path to answer.jsonl.')
    parser.add_argument('--output_path', dest='output_path_flag', help='Path to save metric JSON.')
    parser.add_argument('--angle_version', default='oc', choices=['oc', 'le90', 'le135'])
    parser.add_argument('--counterclockwise', action='store_true', help='Use counter-clockwise convention for mmcv rotated IoU.')
    parser.add_argument('--allow_fallback', action='store_true', help='Use a slower pure Python rotated IoU fallback if mmcv.ops is unavailable.')
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path_flag or args.data_path
    output_path = args.output_path_flag or args.output_path
    if not data_path or not output_path:
        raise SystemExit('Please provide data_path and output_path, either positionally or via --data_path/--output_path.')

    print(f'Loading data from: {data_path}')
    samples = load_jsonl(data_path)
    print(f'Loaded {len(samples)} samples')
    gt_records, pred_records, skipped = build_image_category_obbs(samples, args.angle_version)
    print(f'Skipped invalid boxes: {skipped}')

    result = evaluate(
        gt_records,
        pred_records,
        clockwise=not args.counterclockwise,
        allow_fallback=args.allow_fallback,
    )
    print_results(result)

    payload = {
        'metric': 'fast_rotated_iou_recall_precision_f1',
        'data_path': data_path,
        'iou_threshold': IOU_THRESHOLD,
        'angle_version': args.angle_version,
        'clockwise': not args.counterclockwise,
        'confidence_handling': 'same_as_other_metric_all_predictions_counted_no_score_threshold',
        'skipped': skipped,
        'result': result,
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'Saved metric JSON to: {output_path}')


if __name__ == '__main__':
    main()
