"""Evaluate COCO/LVIS predictions without a per-image maxDets limit."""

import argparse

import fastevaluate_all as fe
import numpy as np


def safe_mean(values):
    values = [value for value in values if value > 0]
    return float(np.mean(values)) if values else 0.0


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def detect_eval_type(gt_path, pred_path, requested):
    if requested != "auto":
        return requested
    return "lvis" if "lvis" in f"{gt_path} {pred_path}".lower() else "coco"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate every prediction per image with FastEval."
    )
    parser.add_argument("--gt", required=True, help="COCO/LVIS ground-truth JSON")
    parser.add_argument("--pred_tsv", required=True, help="Prediction TSV or JSON")
    parser.add_argument(
        "--eval_type", choices=("auto", "coco", "lvis"), default="auto"
    )
    args = parser.parse_args()

    eval_type = detect_eval_type(args.gt, args.pred_tsv, args.eval_type)
    print(f"Detected evaluation type: {eval_type.upper()}")
    result = fe.evaluate(args.gt, args.pred_tsv, 0, 0, eval_type)

    precision = safe_mean(result.get("precision", []))
    recall = safe_mean(result.get("recall", []))
    precision50 = safe_mean(result.get("precision50", []))
    recall50 = safe_mean(result.get("recall50", []))
    precision95 = safe_mean(result.get("precision95", []))
    recall95 = safe_mean(result.get("recall95", []))

    rows = (
        ("Avg Precision", precision),
        ("Avg Recall", recall),
        ("Avg F1", f1(precision, recall)),
        ("Precision@0.50", precision50),
        ("Recall@0.50", recall50),
        ("F1@0.50", f1(precision50, recall50)),
        ("Precision@0.95", precision95),
        ("Recall@0.95", recall95),
        ("F1@0.95", f1(precision95, recall95)),
    )
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}} | {value:.4f}")


if __name__ == "__main__":
    main()
