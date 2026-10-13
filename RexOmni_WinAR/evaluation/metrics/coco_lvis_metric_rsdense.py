import argparse

import fastevaluate_rsdense as fe
import numpy as np


def mean_over_iou(values, metric_name):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise RuntimeError(
            f"FastEval did not return {metric_name!r}. Rebuild "
            "evaluation/fastevaluate_rsdense so the per-image backend is available."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"FastEval returned a non-finite {metric_name} value.")
    return float(np.mean(values))


def format_table(rows, headers):
    # Compute column widths
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
        description=(
            "Evaluate all loaded COCO/LVIS predictions by pooling categories "
            "within each image and averaging image-level precision/recall/F1."
        )
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="/home/u1120250365/share/Rex_Omni/Compression/test/rs-dense200/annotations_coco.json",
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
    """Detect evaluation type from file paths if auto is specified"""
    if eval_type != "auto":
        return eval_type

    # Check for LVIS indicators in paths
    lvis_indicators = ["lvis", "LVIS"]
    for indicator in lvis_indicators:
        if indicator in gt_path or indicator in pred_path:
            return "lvis"

    # Default to COCO
    return "coco"


def main():
    args = parse_args()

    # Detect evaluation type
    eval_type = detect_eval_type(args.gt, args.pred_tsv, args.eval_type)
    print(f"🔍 Detected evaluation type: {eval_type.upper()}")

    res = fe.evaluate(args.gt, args.pred_tsv, 0, 0, eval_type)

    # Matching remains category-aware. The backend pools TP/FP/FN across all
    # categories in each image, computes that image's P/R/F1, and then averages
    # each metric across images. Only the ten IoU thresholds are averaged here.
    image_mean_precision = res.get("image_mean_precision", [])
    image_mean_recall = res.get("image_mean_recall", [])
    image_mean_f1 = res.get("image_mean_f1", [])
    metric_lengths = {
        len(image_mean_precision), len(image_mean_recall), len(image_mean_f1)
    }
    if len(metric_lengths) != 1:
        raise ValueError("Image-mean precision, recall, and F1 have different lengths.")

    avg_precision = mean_over_iou(
        image_mean_precision, "image_mean_precision")
    avg_recall = mean_over_iou(image_mean_recall, "image_mean_recall")
    avg_f1 = mean_over_iou(image_mean_f1, "image_mean_f1")

    # IoU=0.50
    precision50 = float(image_mean_precision[0])
    recall50 = float(image_mean_recall[0])
    f1_50 = float(image_mean_f1[0])

    # IoU=0.95
    precision95 = float(image_mean_precision[-1])
    recall95 = float(image_mean_recall[-1])
    f1_95 = float(image_mean_f1[-1])

    headers = ["Metric", "Value"]
    rows = [
        ["Image-mean Precision", f"{avg_precision:.4f}"],
        ["Image-mean Recall", f"{avg_recall:.4f}"],
        ["Image-mean F1", f"{avg_f1:.4f}"],
        ["Precision@0.50", f"{precision50:.4f}"],
        ["Recall@0.50", f"{recall50:.4f}"],
        ["F1@0.50", f"{f1_50:.4f}"],
        ["Precision@0.95", f"{precision95:.4f}"],
        ["Recall@0.95", f"{recall95:.4f}"],
        ["F1@0.95", f"{f1_95:.4f}"],
    ]

    print(format_table(rows, headers))


if __name__ == "__main__":
    main()
