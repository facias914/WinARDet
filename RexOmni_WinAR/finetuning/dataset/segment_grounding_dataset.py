import copy
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List

import torch
import transformers

from .base_tsv_dataset import GroundingTSVDataset, preprocess_qwen_2_visual
from utils.constants import IGNORE_INDEX


class GroundingSegmentTSVDataset(GroundingTSVDataset):
    """Grounding TSV dataset expanded into answer-segment virtual samples.

    Each dataset item keeps only ``prefix + current_segment`` as model input,
    while also carrying full answer token ids for history-memory compression in
    the trainer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assistant_count_tokenizer = copy.deepcopy(self.tokenizer)
        self._assistant_count_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        self.max_tokens_per_segment = getattr(self.data_args, "max_tokens_per_segment", 300)
        self.use_coord_soft_loss = getattr(self.data_args, "use_coord_soft_loss", False)
        self.coord_vocab_size = getattr(self.data_args, "coord_vocab_size", 1000)
        self.coord_token_id_to_value = self._build_coord_token_id_to_value()
        self.box_start_token_id = self.tokenizer.convert_tokens_to_ids("<|box_start|>")
        self.box_end_token_id = self.tokenizer.convert_tokens_to_ids("<|box_end|>")
        print(f"max_tokens_per_segment:{self.max_tokens_per_segment}")
        self.segment_index = self._build_segment_index()

        if self.img_handle is not None:
            self.img_handle.close()
            self.img_handle = None
        if self.ann_handle is not None:
            self.ann_handle.close()
            self.ann_handle = None

    def __len__(self):
        return len(self.segment_index)

    def _build_coord_token_id_to_value(self) -> Dict[int, int]:
        coord_token_id_to_value = {}
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        for coord_value in range(self.coord_vocab_size):
            token_id = self.tokenizer.convert_tokens_to_ids(f"<{coord_value}>")
            if token_id is None or token_id == unk_token_id:
                continue
            coord_token_id_to_value[int(token_id)] = coord_value
        return coord_token_id_to_value

    def _empty_coord_metadata(self, length: int) -> Dict[str, torch.Tensor]:
        return {
            "coord_mask": torch.zeros(length, dtype=torch.bool),
            "coord_values": torch.zeros(length, dtype=torch.long),
            "coord_roles": torch.zeros(length, dtype=torch.long),
            "box_widths": torch.zeros(length, dtype=torch.float32),
            "box_heights": torch.zeros(length, dtype=torch.float32),
        }

    def _build_answer_coord_metadata(self, answer_token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        metadata = self._empty_coord_metadata(answer_token_ids.numel())
        if not self.use_coord_soft_loss:
            return metadata

        inside_box = False
        current_box = []
        token_ids = [int(token_id) for token_id in answer_token_ids.tolist()]
        for idx, token_id in enumerate(token_ids):
            if token_id == self.box_start_token_id:
                inside_box = True
                current_box = []
                continue
            if token_id == self.box_end_token_id:
                inside_box = False
                current_box = []
                continue
            if not inside_box:
                continue
            coord_value = self.coord_token_id_to_value.get(token_id)
            if coord_value is None:
                continue

            current_box.append((idx, coord_value))
            if len(current_box) != 4:
                continue

            x0, y0, x1, y1 = [value for _, value in current_box]
            width = x1 - x0
            height = y1 - y0
            if width > 0 and height > 0:
                for role, (coord_idx, value) in enumerate(current_box):
                    metadata["coord_mask"][coord_idx] = True
                    metadata["coord_values"][coord_idx] = value
                    metadata["coord_roles"][coord_idx] = role
                    metadata["box_widths"][coord_idx] = float(width)
                    metadata["box_heights"][coord_idx] = float(height)
            current_box = []
        return metadata

    def _build_segment_coord_metadata(
        self,
        answer_coord_metadata: Dict[str, torch.Tensor],
        seg_start: int,
        seg_end: int,
        prefix_len: int,
    ) -> Dict[str, torch.Tensor]:
        segment_len = seg_end - seg_start
        prefix_metadata = self._empty_coord_metadata(prefix_len)
        segment_metadata = {
            key: value[seg_start:seg_end]
            for key, value in answer_coord_metadata.items()
        }
        if segment_len <= 0:
            segment_metadata = self._empty_coord_metadata(0)
        return {
            key: torch.cat([prefix_metadata[key], segment_metadata[key]], dim=0)
            for key in prefix_metadata
        }

    def _assistant_target_token_count(self, answer: str) -> int:
        encode_id = self._assistant_count_tokenizer.apply_chat_template(
            [{"role": "assistant", "content": answer}]
        )
        return max(0, len(encode_id) - 3)

    def _build_segment_index(self) -> List[Dict[str, int]]:
        segment_index = []
        for sample_idx in range(len(self.data)):
            try:
                image_pil, raw_data = self.load_image_and_anno(sample_idx)
                data_dict = self.convert_raw_data(image_pil, raw_data)
                if data_dict is None or len(data_dict.get("boxes", [])) == 0:
                    continue
                ori_width, ori_height = image_pil.size
                if self.task_fn is not None:
                    data_dict = self.task_fn(
                        {"id2name": self.id2name, "annotations": data_dict},
                        ori_width,
                        ori_height,
                    )
                answer = data_dict["conversations"][-1].get(
                    "value", data_dict["conversations"][-1].get("content", "")
                )
                num_answer_tokens = self._assistant_target_token_count(answer)
                num_segments = max(1, math.ceil(num_answer_tokens / self.max_tokens_per_segment))
                for segment_id in range(num_segments):
                    seg_start = segment_id * self.max_tokens_per_segment
                    seg_end = min(seg_start + self.max_tokens_per_segment, num_answer_tokens)
                    segment_index.append(
                        {
                            "sample_idx": sample_idx,
                            "segment_start_token": seg_start,
                            "segment_end_token": seg_end,
                            "segment_id": segment_id,
                            "num_segments": num_segments,
                        }
                    )
            except Exception as exc:
                print(f"error building segment index {self.dataset_name} id: {sample_idx}. {exc}")
        return segment_index

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        segment_info = self.segment_index[i]
        base_idx = segment_info["sample_idx"]

        try:
            image_pil, data_dict = self.load_image_and_anno(base_idx)
        except Exception as e:
            print(f"error image {self.dataset_name} id: {base_idx}. {e}")
            return self.__getitem__((i + 1) % len(self.segment_index))
        w, h = image_pil.size
        if w < 28 or h < 28:
            print(f"image size is too small {w}x{h}, skip this image.")
            return self.__getitem__((i + 1) % len(self.segment_index))

        data_dict = self.convert_raw_data(image_pil, data_dict)
        if data_dict is None:
            return self.__getitem__((i + 1) % len(self.segment_index))
        if "boxes" in data_dict and len(data_dict["boxes"]) == 0:
            print(f"no boxes in {self.dataset_name} id: {base_idx}.")
            return self.__getitem__((i + 1) % len(self.segment_index))

        ori_width, ori_height = image_pil.size
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
                ori_width,
                ori_height,
            )

        sources = copy.deepcopy([e["conversations"] for e in [data_dict]])
        tokenized = preprocess_qwen_2_visual(
            sources,
            self.tokenizer,
            grid_thw=grid_thw_merged,
            visual_type="image",
            system_message=self.system_message,
        )

        input_ids_full = tokenized["input_ids"][0]
        labels_full = tokenized["labels"][0]
        answer_positions = torch.nonzero(labels_full.ne(IGNORE_INDEX), as_tuple=False).flatten()
        if answer_positions.numel() == 0:
            return self.__getitem__((i + 1) % len(self.segment_index))

        answer_start_abs = int(answer_positions[0].item())
        answer_token_ids = labels_full[answer_positions].to(dtype=torch.long)
        answer_coord_metadata = self._build_answer_coord_metadata(answer_token_ids)

        seg_start = int(segment_info["segment_start_token"])
        seg_end = int(segment_info["segment_end_token"])
        seg_start = max(0, min(seg_start, int(answer_positions.numel())))
        seg_end = max(seg_start, min(seg_end, int(answer_positions.numel())))
        selected_answer_positions = answer_positions[seg_start:seg_end]
        if selected_answer_positions.numel() == 0:
            return self.__getitem__((i + 1) % len(self.segment_index))

        prefix_ids = input_ids_full[:answer_start_abs]
        segment_ids = input_ids_full[selected_answer_positions]
        input_ids = torch.cat([prefix_ids, segment_ids], dim=0)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[prefix_ids.size(0) :] = segment_ids

        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            input_ids.unsqueeze(0),
            torch.stack(grid_thw, dim=0),
        )

        data_item = dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            pixel_values=image,
            image_grid_thw=grid_thw,
            context_segment_start=torch.tensor(seg_start, dtype=torch.long),
            context_segment_end=torch.tensor(seg_end, dtype=torch.long),
            context_num_segments=torch.tensor(segment_info["num_segments"], dtype=torch.long),
            context_original_index=torch.tensor(base_idx, dtype=torch.long),
            context_answer_token_ids=answer_token_ids,
        )
        if self.use_coord_soft_loss:
            data_item.update(
                self._build_segment_coord_metadata(
                    answer_coord_metadata,
                    seg_start,
                    seg_end,
                    prefix_ids.size(0),
                )
            )
        return data_item


@dataclass
class SegmentDataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    include_task_info: bool = False

    def __call__(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        images = list(
            itertools.chain(
                *(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
            )
        )
        if len(images) != 0:
            batch["pixel_values"] = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance)
                )
            )
            batch["image_grid_thw"] = torch.stack(grid_thw, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None

        for key in (
            "context_segment_start",
            "context_segment_end",
            "context_num_segments",
            "context_original_index",
        ):
            if key in instances[0]:
                vals = [
                    v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.long)
                    for v in (instance[key] for instance in instances)
                ]
                batch[key] = torch.stack(vals, dim=0).view(-1)

        if "context_answer_token_ids" in instances[0]:
            answer_ids = torch.nn.utils.rnn.pad_sequence(
                [instance["context_answer_token_ids"] for instance in instances],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            answer_mask = answer_ids.ne(self.tokenizer.pad_token_id)
            batch["context_answer_token_ids"] = answer_ids
            batch["context_answer_mask"] = answer_mask

        if "coord_mask" in instances[0]:
            batch["coord_mask"] = torch.nn.utils.rnn.pad_sequence(
                [instance["coord_mask"] for instance in instances],
                batch_first=True,
                padding_value=False,
            )
            for key in ("coord_values", "coord_roles"):
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [instance[key] for instance in instances],
                    batch_first=True,
                    padding_value=0,
                )
            for key in ("box_widths", "box_heights"):
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [instance[key] for instance in instances],
                    batch_first=True,
                    padding_value=0.0,
                )

        if self.include_task_info:
            batch["task_type"] = [instance.get("task_type", "unknown") for instance in instances]
            batch["dataset_name"] = [instance.get("dataset_name", None) for instance in instances]

        return batch
