import argparse
import json

import fastevaluate_dota as fe
import numpy as np


def safe_mean(values):
    vals = [v for v in values if v > 0]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


def format_table(rows, headers):
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt_row(row):
        return " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO/LVIS predictions (FastEval) with per-category metrics."
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="/home/ubuntu/wcx/Locayany/DATA/test/rs-dense200/annotations_coco.json",
        help="Path to GT json (COCO/LVIS)",
    )
    parser.add_argument(
        "--pred_tsv",
        type=str,
        default="Mountchicken/Rex-Omni-Eval/eval_results/box_eval/COCO/fast_eval.tsv",
        help="Path to predictions (TSV/JSON)",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="auto",
        choices=["auto", "coco", "lvis"],
        help="Evaluation type: auto (detect from filename), coco, or lvis",
    )
    return parser.parse_args()


def detect_eval_type(gt_path, pred_path, eval_type):
    """Detect evaluation type from file paths if auto is specified."""
    if eval_type != "auto":
        return eval_type

    lvis_indicators = ["lvis", "LVIS"]
    for indicator in lvis_indicators:
        if indicator in gt_path or indicator in pred_path:
            return "lvis"

    return "coco"


def load_category_names(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    categories = gt_data.get("categories", [])
    return [
        {
            "id": category.get("id", idx),
            "name": category.get("name", f"category_{idx}"),
        }
        for idx, category in enumerate(categories)
    ]


def get_metric_value(res, key, idx):
    values = res.get(key, [])
    if idx >= len(values):
        return 0.0
    value = values[idx]
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def print_per_category_results(res, categories):
    rows = []
    category_count = max(
        len(categories),
        len(res.get("precision", [])),
        len(res.get("recall", [])),
        len(res.get("precision50", [])),
        len(res.get("recall50", [])),
        len(res.get("precision95", [])),
        len(res.get("recall95", [])),
    )

    for idx in range(category_count):
        if idx < len(categories):
            cat_id = categories[idx]["id"]
            cat_name = categories[idx]["name"]
        else:
            cat_id = idx
            cat_name = f"category_{idx}"

        precision = get_metric_value(res, "precision", idx)
        recall = get_metric_value(res, "recall", idx)
        precision50 = get_metric_value(res, "precision50", idx)
        recall50 = get_metric_value(res, "recall50", idx)
        precision95 = get_metric_value(res, "precision95", idx)
        recall95 = get_metric_value(res, "recall95", idx)

        rows.append(
            [
                cat_id,
                cat_name,
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1(precision, recall):.4f}",
                f"{precision50:.4f}",
                f"{recall50:.4f}",
                f"{f1(precision50, recall50):.4f}",
                f"{precision95:.4f}",
                f"{recall95:.4f}",
                f"{f1(precision95, recall95):.4f}",
            ]
        )

    headers = [
        "Cat ID",
        "Category",
        "Precision",
        "Recall",
        "F1",
        "P@0.50",
        "R@0.50",
        "F1@0.50",
        "P@0.95",
        "R@0.95",
        "F1@0.95",
    ]
    print("\nPer-category Results")
    print(format_table(rows, headers))


def print_final_results(res):
    avg_precision = safe_mean(res.get("precision", []))
    avg_recall = safe_mean(res.get("recall", []))
    avg_f1 = f1(avg_precision, avg_recall)

    precision50 = safe_mean(res.get("precision50", []))
    recall50 = safe_mean(res.get("recall50", []))
    f1_50 = f1(precision50, recall50)

    precision95 = safe_mean(res.get("precision95", []))
    recall95 = safe_mean(res.get("recall95", []))
    f1_95 = f1(precision95, recall95)

    headers = ["Metric", "Value"]
    rows = [
        ["Avg Precision", f"{avg_precision:.4f}"],
        ["Avg Recall", f"{avg_recall:.4f}"],
        ["Avg F1", f"{avg_f1:.4f}"],
        ["Precision@0.50", f"{precision50:.4f}"],
        ["Recall@0.50", f"{recall50:.4f}"],
        ["F1@0.50", f"{f1_50:.4f}"],
        ["Precision@0.95", f"{precision95:.4f}"],
        ["Recall@0.95", f"{recall95:.4f}"],
        ["F1@0.95", f"{f1_95:.4f}"],
    ]

    print("\nFinal Results")
    print(format_table(rows, headers))


def main():
    args = parse_args()

    eval_type = detect_eval_type(args.gt, args.pred_tsv, args.eval_type)
    print(f"🔍 Detected evaluation type: {eval_type.upper()}")

    categories = load_category_names(args.gt)
    res = fe.evaluate(args.gt, args.pred_tsv, 0, 0, eval_type)

    print_per_category_results(res, categories)
    print_final_results(res)


if __name__ == "__main__":
    main()
