"""Trainer for segment-weighted NTP with uncompressed causal history."""

from typing import Dict

import torch
from transformers import Trainer

from utils.constants import IGNORE_INDEX


WEIGHTED_SEGMENT_METADATA_KEYS = (
    "weighted_segment_start",
    "weighted_segment_end",
    "weighted_segment_token_count",
    "weighted_num_segments",
    "weighted_answer_token_count",
    "weighted_original_index",
)


class WeightedSegmentNTPTrainer(Trainer):
    """Run standard causal NTP while supervising only the selected answer segment.

    The dataset has already constructed ``prefix + raw history + current
    segment`` inputs and set every non-current label to ``IGNORE_INDEX``.  This
    trainer keeps the segment metadata through collation, validates the loss
    mask, removes metadata before the model call, and otherwise delegates to
    the ordinary Hugging Face causal-LM loss.
    """

    def __init__(self, *args, **kwargs):
        trainer_args = kwargs.get("args")
        if trainer_args is None and len(args) >= 2:
            trainer_args = args[1]
        if trainer_args is not None:
            trainer_args.remove_unused_columns = False
        super().__init__(*args, **kwargs)
        self._last_weighted_segment_stats: Dict[str, float] = {}

    def _pop_and_validate_segment_metadata(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        metadata = {
            key: inputs.pop(key)
            for key in WEIGHTED_SEGMENT_METADATA_KEYS
            if key in inputs
        }
        if "labels" not in inputs:
            raise ValueError("WeightedSegmentNTPTrainer requires labels.")

        labels = inputs["labels"]
        if labels.dim() != 2:
            raise ValueError(f"Expected labels with shape [B, L], got {tuple(labels.shape)}.")
        supervised_counts = labels.ne(IGNORE_INDEX).sum(dim=1)

        expected_counts = metadata.get("weighted_segment_token_count")
        if expected_counts is not None:
            expected_counts = expected_counts.to(
                device=supervised_counts.device, dtype=supervised_counts.dtype
            ).view(-1)
            if expected_counts.numel() != supervised_counts.numel():
                raise ValueError(
                    "weighted_segment_token_count batch size mismatch: "
                    f"{expected_counts.numel()} != {supervised_counts.numel()}."
                )
            if not torch.equal(supervised_counts, expected_counts):
                raise ValueError(
                    "Weighted segment loss-mask mismatch: "
                    f"labels={supervised_counts.tolist()}, expected={expected_counts.tolist()}."
                )

        if torch.any(supervised_counts <= 0):
            raise ValueError(
                f"Every weighted NTP sample must supervise at least one token: "
                f"{supervised_counts.tolist()}."
            )

        self._last_weighted_segment_stats = {
            "mean_supervised_tokens": float(supervised_counts.float().mean().item()),
            "max_supervised_tokens": float(supervised_counts.max().item()),
        }
        if "weighted_num_segments" in metadata:
            self._last_weighted_segment_stats["mean_segments_per_example"] = float(
                metadata["weighted_num_segments"].float().mean().item()
            )
        return metadata

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Trainer may reuse/mutate the input dictionary around compute_loss;
        # make a shallow copy before removing bookkeeping fields.
        model_inputs = dict(inputs)
        self._pop_and_validate_segment_metadata(model_inputs)
        try:
            return super().compute_loss(
                model,
                model_inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        except TypeError:
            # Compatibility with transformers versions whose Trainer does not
            # yet expose num_items_in_batch.
            return super().compute_loss(
                model,
                model_inputs,
                return_outputs=return_outputs,
            )
