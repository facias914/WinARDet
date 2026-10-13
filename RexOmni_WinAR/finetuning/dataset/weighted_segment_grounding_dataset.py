"""Segment-weighted NTP dataset with uncompressed answer history.

This module provides a control dataset for studying the sampling and loss
reweighting introduced by segment-compression training.  The portion of an
answer visible within ``max_length`` produces one virtual sample per segment.
Virtual sample ``j`` uses the original (uncompressed) answer tokens before
segment ``j`` as causal context, and computes language-model loss only on the
in-range part of segment ``j``.
"""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import transformers

from .collator import DataCollatorForSupervisedDataset
from .base_tsv_dataset import GroundingTSVDataset
from utils.constants import IGNORE_INDEX


_QWEN_2_VISUAL_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def _prepare_weighted_segment_tokenizer(tokenizer):
    """Create the dataset-private tokenizer once instead of once per sample."""
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = _QWEN_2_VISUAL_CHAT_TEMPLATE
    return tokenizer


def _preprocess_qwen_2_visual_with_prepared_tokenizer(
    sources,
    tokenizer,
    grid_thw: List,
    visual_type: str = "image",
    system_message: str = "You are a helpful assistant.",
) -> Dict[str, torch.Tensor]:
    """Equivalent Qwen visual preprocessing using an already-owned tokenizer.

    This intentionally mirrors ``tsv_dataset.preprocess_qwen_2_visual`` while
    omitting its per-call tokenizer deepcopy.  It is private to this dataset so
    the shared TSV dataset implementation and all of its callers remain
    unchanged.
    """
    roles = {"human": "user", "gpt": "assistant"}
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    visual_replicate_index = 0
    input_ids, targets = [], []
    for source in sources:
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except Exception:
            print(sources)

        input_id, target = [], []
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except Exception:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for part_index in range(len(parts) - 1):
                        new_parts.append(parts[part_index])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            encode_id = tokenizer.apply_chat_template(
                [{"role": role, "content": content}]
            )
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        if len(input_id) != len(target):
            raise ValueError(f"input/target length mismatch: {len(input_id)} != {len(target)}")
        input_ids.append(input_id)
        targets.append(target)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(targets, dtype=torch.long),
    }


def build_raw_history_segment_view(
    input_ids_full: torch.Tensor,
    labels_full: torch.Tensor,
    segment_start: int,
    segment_end: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Build ``prefix + raw history + current segment`` and a segment-only loss mask.

    Args:
        input_ids_full: Complete tokenized conversation, shape ``[L]``.
        labels_full: Complete causal labels with non-answer positions set to
            ``IGNORE_INDEX``, shape ``[L]``.
        segment_start: Inclusive answer-relative token offset.
        segment_end: Exclusive answer-relative token offset.

    Returns:
        Truncated input ids, masked labels, clamped start/end offsets, and the
        total number of answer tokens.
    """
    if input_ids_full.dim() != 1 or labels_full.dim() != 1:
        raise ValueError("input_ids_full and labels_full must both be one-dimensional.")
    if input_ids_full.numel() != labels_full.numel():
        raise ValueError(
            f"input/label length mismatch: {input_ids_full.numel()} != {labels_full.numel()}."
        )

    answer_positions = torch.nonzero(labels_full.ne(IGNORE_INDEX), as_tuple=False).flatten()
    num_answer_tokens = int(answer_positions.numel())
    if num_answer_tokens == 0:
        raise ValueError("Cannot build a weighted NTP segment without answer labels.")

    start = max(0, min(int(segment_start), num_answer_tokens))
    end = max(start, min(int(segment_end), num_answer_tokens))
    if end <= start:
        raise ValueError(
            f"Empty weighted NTP segment after clamping: [{start}, {end}) of {num_answer_tokens}."
        )

    current_positions = answer_positions[start:end]
    # Future answer tokens must not be visible.  Keep every original token up
    # through the current segment so that all previous answer tokens remain
    # ordinary, uncompressed causal context.
    input_end = int(current_positions[-1].item()) + 1
    input_ids = input_ids_full[:input_end].clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[current_positions] = labels_full[current_positions]
    return input_ids, labels, start, end, num_answer_tokens


class GroundingWeightedSegmentNTPTSVDataset(GroundingTSVDataset):
    """Expand grounding examples into segment-weighted raw-history NTP samples.

    The number of virtual samples is based on the answer tokens visible within
    ``max_length``.  Unlike compressed training, earlier answer segments stay
    in the main model input as their original token ids.  Only the current
    segment contributes to loss, and a segment crossing ``max_length`` keeps
    its in-range portion instead of dropping the complete source example.
    """

    def __init__(
        self,
        *args,
        tokens_per_segment: int = 200,
        max_segment_samples_per_example: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokens_per_segment = int(tokens_per_segment)
        self.max_segment_samples_per_example = int(max_segment_samples_per_example)

        # Slow Qwen tokenizers are expensive to deepcopy.  Keep one isolated,
        # preconfigured tokenizer for both index construction and __getitem__.
        self._preprocess_tokenizer = _prepare_weighted_segment_tokenizer(self.tokenizer)
        self._assistant_count_tokenizer = self._preprocess_tokenizer
        self.segment_index = self._build_weighted_segment_index()

        # Do not carry file handles created during index construction into
        # dataloader worker processes.
        if self.img_handle is not None:
            self.img_handle.close()
            self.img_handle = None
        if self.ann_handle is not None:
            self.ann_handle.close()
            self.ann_handle = None

        print(
            "weighted NTP segment sampling: "
            f"tokens_per_segment={self.tokens_per_segment}, "
            f"max_segment_samples_per_example={self.max_segment_samples_per_example}, "
            f"virtual_samples={len(self.segment_index)}"
        )

    def __len__(self) -> int:
        return len(self.segment_index)

    def _assistant_target_token_count(self, answer: str) -> int:
        encoded = self._assistant_count_tokenizer.apply_chat_template(
            [{"role": "assistant", "content": answer}]
        )
        # The assistant header occupies the first three tokens, matching the
        # masking convention in preprocess_qwen_2_visual.
        return max(0, len(encoded) - 3)

    def _selected_segment_ids(self, num_segments: int) -> List[int]:
        cap = self.max_segment_samples_per_example
        if cap <= 0 or num_segments <= cap:
            return list(range(num_segments))
        if cap == 1:
            return [num_segments - 1]

        # If an explicit cap is requested, cover the complete answer rather
        # than keeping only early segments.  The default cap=0 keeps all.
        return [
            round(sample_idx * (num_segments - 1) / (cap - 1))
            for sample_idx in range(cap)
        ]

    def _merged_image_token_count_for_index(self, image_pil) -> int:
        """Return the number of LLM image tokens without encoding image pixels.

        ``SegGroundingTaskFn`` mirrors Qwen-VL's smart-resize calculation, so
        the cheap path below produces the same merged visual-token count as
        ``process_image_unified``.  Keep an exact processor fallback for other
        task functions.
        """
        processor = self.data_args.image_processor
        smart_resize = getattr(self.task_fn, "_smart_resize_for_visual_tokens", None)
        if callable(smart_resize):
            resized_height, resized_width = smart_resize(
                image_pil.height, image_pil.width
            )
            grid_h = resized_height // int(processor.patch_size)
            grid_w = resized_width // int(processor.patch_size)
            return int(grid_h * grid_w // int(processor.merge_size) ** 2)

        _, grid_thw = self.process_image_unified(image_pil)
        if isinstance(grid_thw, Sequence):
            if len(grid_thw) != 1:
                raise ValueError(
                    "GroundingWeightedSegmentNTPTSVDataset expects one image per sample."
                )
            grid_thw = grid_thw[0]
        return int(grid_thw.prod().item() // int(processor.merge_size) ** 2)

    def _build_weighted_segment_index(self) -> List[Dict[str, object]]:
        segment_index: List[Dict[str, object]] = []
        for sample_idx in range(len(self.data)):
            try:
                image_pil, raw_data = self.load_image_and_anno(sample_idx)
                data_dict = self.convert_raw_data(image_pil, raw_data)
                if data_dict is None or len(data_dict.get("boxes", [])) == 0:
                    continue

                ori_width, ori_height = image_pil.size
                if ori_width < 28 or ori_height < 28:
                    continue
                if self.task_fn is not None:
                    data_dict = self.task_fn(
                        {"id2name": self.id2name, "annotations": data_dict},
                        ori_width,
                        ori_height,
                    )

                # Build the sampling index from the part of the complete NTP
                # sequence that is actually visible under max_length.  This
                # prevents fully out-of-range virtual segments from remaining
                # in the sampler and recursively duplicating earlier samples.
                indexed_conversations = copy.deepcopy(data_dict["conversations"])
                merged_image_tokens = self._merged_image_token_count_for_index(
                    image_pil
                )
                indexed_tokenized = _preprocess_qwen_2_visual_with_prepared_tokenizer(
                    copy.deepcopy([indexed_conversations]),
                    self._preprocess_tokenizer,
                    grid_thw=[merged_image_tokens],
                    visual_type="image",
                    system_message=self.system_message,
                )
                indexed_labels = indexed_tokenized["labels"][0]
                num_answer_tokens = int(indexed_labels.ne(IGNORE_INDEX).sum().item())
                truncated_by_max_length = (
                    indexed_tokenized["input_ids"].shape[1] > self.max_length
                )
                visible_answer_tokens = int(
                    indexed_labels[: self.max_length].ne(IGNORE_INDEX).sum().item()
                )
                if visible_answer_tokens == 0:
                    continue

                num_segments = math.ceil(
                    visible_answer_tokens / self.tokens_per_segment
                )

                for segment_id in self._selected_segment_ids(num_segments):
                    start = segment_id * self.tokens_per_segment
                    # Keep the nominal full segment end.  __getitem__ clamps it
                    # against the answer tokens visible for its sampled prompt,
                    # allowing the boundary segment to contribute partially.
                    end = min(start + self.tokens_per_segment, num_answer_tokens)
                    segment_index.append(
                        {
                            "sample_idx": sample_idx,
                            "segment_start_token": start,
                            "segment_end_token": end,
                            "segment_id": segment_id,
                            "num_segments": num_segments,
                            "num_answer_tokens": num_answer_tokens,
                            "indexed_visible_answer_tokens": visible_answer_tokens,
                            "indexed_conversations": indexed_conversations,
                            "truncated_by_max_length": truncated_by_max_length,
                        }
                    )
            except Exception as exc:
                print(
                    f"error building weighted NTP index {self.dataset_name} "
                    f"id: {sample_idx}. {exc}"
                )
        return segment_index

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        segment_info = self.segment_index[i]
        base_idx = int(segment_info["sample_idx"])

        try:
            image_pil, data_dict = self.load_image_and_anno(base_idx)
        except Exception as exc:
            print(f"error image {self.dataset_name} id: {base_idx}. {exc}")
            return self.__getitem__((i + 1) % len(self.segment_index))

        width, height = image_pil.size
        if width < 28 or height < 28:
            print(f"image size is too small {width}x{height}, skip this image.")
            return self.__getitem__((i + 1) % len(self.segment_index))

        data_dict = self.convert_raw_data(image_pil, data_dict)
        if data_dict is None:
            return self.__getitem__((i + 1) % len(self.segment_index))
        if "boxes" in data_dict and len(data_dict["boxes"]) == 0:
            print(f"no boxes in {self.dataset_name} id: {base_idx}.")
            return self.__getitem__((i + 1) % len(self.segment_index))

        image, grid_thw = self.process_image_unified(image_pil)
        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [
            merged_thw.prod() // self.data_args.image_processor.merge_size**2
            for merged_thw in grid_thw_merged
        ]

        if self.task_fn is not None:
            data_dict = self.task_fn(
                {"id2name": self.id2name, "annotations": data_dict},
                width,
                height,
            )

        # Prompt templates have different token lengths.  Once an example is
        # known to cross max_length, reuse the conversation selected during
        # index construction for all of its virtual segments.  This gives the
        # source example one stable truncation boundary and guarantees that
        # its retained answer prefix is covered continuously exactly once.
        if bool(segment_info.get("truncated_by_max_length", False)):
            data_dict["conversations"] = copy.deepcopy(
                segment_info["indexed_conversations"]
            )

        sources = copy.deepcopy([data_dict["conversations"]])
        tokenized = _preprocess_qwen_2_visual_with_prepared_tokenizer(
            sources,
            self._preprocess_tokenizer,
            grid_thw=grid_thw_merged,
            visual_type="image",
            system_message=self.system_message,
        )
        input_ids_full = tokenized["input_ids"][0]
        labels_full = tokenized["labels"][0]

        actual_answer_tokens = int(labels_full.ne(IGNORE_INDEX).sum().item())
        indexed_answer_tokens = int(segment_info["num_answer_tokens"])
        if actual_answer_tokens != indexed_answer_tokens:
            raise ValueError(
                "Weighted segment-NTP requires deterministic answer ordering: "
                f"indexed {indexed_answer_tokens} answer tokens but __getitem__ produced "
                f"{actual_answer_tokens} for dataset={self.dataset_name}, sample={base_idx}. "
                "Use the same SegGroundingTaskFn ordering as compression training."
            )

        segment_start = int(segment_info["segment_start_token"])
        segment_end = int(segment_info["segment_end_token"])
        visible_answer_tokens = int(
            labels_full[: self.max_length].ne(IGNORE_INDEX).sum().item()
        )

        if segment_start >= visible_answer_tokens:
            # Prompt templates have different token lengths.  If a randomly
            # selected longer prompt pushes this indexed boundary segment fully
            # outside max_length, fall back to the exact conversation used to
            # build the valid segment index instead of sampling another item.
            tokenized = _preprocess_qwen_2_visual_with_prepared_tokenizer(
                copy.deepcopy([segment_info["indexed_conversations"]]),
                self._preprocess_tokenizer,
                grid_thw=grid_thw_merged,
                visual_type="image",
                system_message=self.system_message,
            )
            input_ids_full = tokenized["input_ids"][0]
            labels_full = tokenized["labels"][0]
            actual_answer_tokens = int(labels_full.ne(IGNORE_INDEX).sum().item())
            if actual_answer_tokens != indexed_answer_tokens:
                raise ValueError(
                    "Answer token count changed between weighted segment index "
                    f"construction and retrieval: {indexed_answer_tokens} != "
                    f"{actual_answer_tokens}."
                )
            visible_answer_tokens = int(
                labels_full[: self.max_length].ne(IGNORE_INDEX).sum().item()
            )

        effective_segment_end = min(segment_end, visible_answer_tokens)
        if effective_segment_end <= segment_start:
            raise RuntimeError(
                "Weighted segment index contains a segment fully outside max_length: "
                f"sample={base_idx}, segment=[{segment_start}, {segment_end}), "
                f"visible_answer_tokens={visible_answer_tokens}."
            )

        try:
            input_ids, labels, start, end, num_answer_tokens = build_raw_history_segment_view(
                input_ids_full,
                labels_full,
                segment_start,
                effective_segment_end,
            )
        except ValueError as exc:
            print(f"invalid weighted NTP segment {self.dataset_name} id {base_idx}: {exc}")
            return self.__getitem__((i + 1) % len(self.segment_index))

        if input_ids.numel() > self.max_length:
            raise RuntimeError(
                "Weighted segment truncation invariant failed: "
                f"{input_ids.numel()} > max_length={self.max_length}."
            )

        # Preserve the base model's multimodal RoPE exactly.  Compute Qwen2.5-VL
        # 3-axis MRoPE on the same complete input used by GroundingTSVDataset,
        # then take the causal prefix retained by this virtual segment.  This
        # keeps image temporal/height/width ids and all subsequent text ids
        # byte-for-byte identical to base NTP at every retained position.
        full_position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            input_ids_full.unsqueeze(0),
            torch.stack(grid_thw, dim=0),
        )
        position_ids = full_position_ids[..., : input_ids.numel()].contiguous()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "pixel_values": image,
            "image_grid_thw": grid_thw,
            "weighted_segment_start": torch.tensor(start, dtype=torch.long),
            "weighted_segment_end": torch.tensor(end, dtype=torch.long),
            "weighted_segment_token_count": torch.tensor(end - start, dtype=torch.long),
            "weighted_num_segments": torch.tensor(segment_info["num_segments"], dtype=torch.long),
            "weighted_answer_token_count": torch.tensor(num_answer_tokens, dtype=torch.long),
            "weighted_original_index": torch.tensor(base_idx, dtype=torch.long),
        }


@dataclass
class WeightedSegmentNTPDataCollator(DataCollatorForSupervisedDataset):
    """Base multimodal collator plus weighted-segment bookkeeping fields."""

    tokenizer: transformers.PreTrainedTokenizer
    include_task_info: bool = False

    def __call__(self, instances):
        batch = super().__call__(instances)
        for key in (
            "weighted_segment_start",
            "weighted_segment_end",
            "weighted_segment_token_count",
            "weighted_num_segments",
            "weighted_answer_token_count",
            "weighted_original_index",
        ):
            values = [
                value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.long)
                for value in (instance[key] for instance in instances)
            ]
            batch[key] = torch.stack(values, dim=0).view(-1)
        return batch
