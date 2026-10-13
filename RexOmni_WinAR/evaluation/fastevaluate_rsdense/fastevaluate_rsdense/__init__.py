"""Fast COCO-style evaluation with category-pooled, per-image metrics."""

from ._fastevaluate_rsdense import evaluate

__all__ = ["evaluate"]
