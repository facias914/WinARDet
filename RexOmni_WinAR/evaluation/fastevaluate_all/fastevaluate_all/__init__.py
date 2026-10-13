"""Fast COCO/LVIS evaluation without per-image prediction truncation."""

from ._fastevaluate_all import evaluate

__all__ = ["evaluate"]
