# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import math
import os
from base64 import b64decode
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

# from . import torch_functional as VF
import verl.utils.torch_functional as VF


def get_rope_index(
    processor: "Qwen2VLProcessor",
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen2-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1546
    """
    spatial_merge_size = processor.image_processor.merge_size
    tokens_per_second = 2
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|vision_start|>"
    )
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(
            3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device
        )  # (3, seqlen)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0

                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            )
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
            )

    return position_ids


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image) -> ImageObject:

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(
                image.height * resize_factor
            )
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(
                image.height * resize_factor
            )
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class TSVRLHFDataset(Dataset, ImageProcessMixin):
    """
    TSV dataset for RLHF training that supports both GRPO format and grounding format.
    """

    def __init__(
        self,
        image_tsv_file,
        anno_tsv_file,
        anno_idx_file,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int = 4096,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        reward_name: str = "box_iou",
        dataset_name: str = "custom_grounding",
        task_fn=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts
        self.reward_name = reward_name
        self.dataset_name = dataset_name

        # Build task_fn if provided
        if task_fn is not None:
            from engine.registry import BUILDER

            self.task_fn = BUILDER.build(task_fn)
        else:
            self.task_fn = None

        self.dataset = []
        f = open(anno_idx_file)
        for line in f:
            self.dataset.append(int(line.strip()))

        self.img_handle = None
        self.ann_handle = None
        self.img_tsv_file = image_tsv_file
        self.ann_tsv_file = anno_tsv_file

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        # For grounding data, use the prompt field directly
        prompt_str: str = example.get("prompt", "")
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        content_list = []
        for i, content in enumerate(prompt_str.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})

            if content:
                content_list.append({"type": "text", "text": content})

        return [{"role": "user", "content": content_list}]

    def __len__(self):
        return len(self.dataset)

    def load_image_and_anno(self, idx):
        ann_line_idx = self.dataset[idx]
        if self.ann_handle is None:
            self.ann_handle = open(self.ann_tsv_file)
        self.ann_handle.seek(ann_line_idx)

        img_line_idx, ann = self.ann_handle.readline().strip().split("\t")
        img_line_idx = int(img_line_idx)
        if self.img_handle is None:
            self.img_handle = open(self.img_tsv_file)
        self.img_handle.seek(img_line_idx)
        img = self.img_handle.readline().strip().split("\t")[1]
        if img.startswith("b'"):
            img = img[1:-1]
        img = BytesIO(b64decode(img))
        image = Image.open(img).convert("RGB")
        data_dict = json.loads(ann)
        return image, data_dict

    def _convert_grounding_to_grpo_format(self, raw_data_dict, image_size=None):
        """
        Convert grounding format data to GRPO format.

        Grounding format (from tsv_dataset.py):
        {
            "boxes": [[x0, y0, x1, y1], ...]  # numpy array or list
            "labels": ["tree", "third", ...]
            "size": (height, width)
        }

        Target GRPO format:
        {
            "raw_image_size": [width, height],
            "resized_image_size": [resized_width, resized_height],
            "answer": {
                "tree": {"boxes": [[x0, y0, x1, y1], ...]},
                "third": {"boxes": [[x0, y0, x1, y1], ...]}
            },
            "reward_name": "box_iou",
            "dataset_name": "custom_grounding"
        }
        """
        # Group boxes by label
        boxes_by_label = defaultdict(list)
        boxes = raw_data_dict.get("boxes", [])
        labels = raw_data_dict.get("labels", [])

        # Handle both numpy array and list
        for box, label in zip(boxes, labels):
            # Convert to list if numpy array
            box_list = box.tolist() if isinstance(box, np.ndarray) else box
            boxes_by_label[label].append(box_list)

        # Construct answer dict in GRPO format
        answer_dict = {
            label: {"boxes": boxes_list} for label, boxes_list in boxes_by_label.items()
        }

        # Get image size
        if image_size is not None:
            height, width = image_size
        else:
            size = raw_data_dict.get("size", (800, 800))
            if isinstance(size, (list, tuple)):
                height, width = size
            else:
                height, width = 800, 800

        grpo_answer = {
            "raw_image_size": [width, height],
            "resized_image_size": [width, height],  # Same as raw for now
            "answer": answer_dict,
            "reward_name": "box_iou",
            "dataset_name": raw_data_dict.get("dataset_name", "custom_grounding"),
        }

        return grpo_answer

    def _convert_raw_tsv_to_grounding(self, data_dict):
        """
        Convert raw TSV annotation format to grounding format.

        Raw TSV format:
        {
            "boxes": [
                {"bbox": [x0, y0, x1, y1], "phrase": "tree"},
                {"bbox": [x0, y0, x1, y1], "phrase": "third"},
                ...
            ]
        }

        Converted to grounding format:
        {
            "boxes": [[x0, y0, x1, y1], ...],
            "labels": ["tree", "third", ...]
        }
        """
        boxes = []
        labels = []

        if "boxes" in data_dict:
            for anno in data_dict["boxes"]:
                boxes.append(anno.get("bbox", []))
                labels.append(anno.get("phrase", anno.get("caption", "")))

        return boxes, labels

    def __getitem__(self, index):

        image_pil, data_dict = self.load_image_and_anno(index)

        # Convert raw data to grounding format
        raw_boxes, labels = self._convert_raw_tsv_to_grounding(data_dict)
        img_width, img_height = image_pil.size
        image_size = (img_height, img_width)

        grounding_data = {
            "boxes": np.array(raw_boxes),
            "labels": labels,
            "size": image_size,
        }

        # Use task_fn to build conversations if provided
        if self.task_fn is not None:
            example = {"annotations": grounding_data}
            example_with_conversations = self.task_fn(example, img_width, img_height)
            # Extract conversation from human to get the prompt
            conversations = example_with_conversations.get("conversations", [])
            if conversations:
                # Get the user's message (question)
                for conv in conversations:
                    if conv.get("from") == "human":
                        prompt = conv.get("value", "")
                        data_dict["prompt"] = prompt
                        break

        # Convert to GRPO format
        grpo_answer = self._convert_grounding_to_grpo_format(grounding_data, image_size)
        # Store the converted answer
        data_dict["answer"] = json.dumps(grpo_answer)

        messages = self._build_messages(data_dict)

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        images = [self.process_image(image_pil)]
        model_inputs = self.processor(
            images, [prompt], add_special_tokens=False, return_tensors="pt"
        )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        data_dict["multi_modal_data"] = {"image": images}
        data_dict["multi_modal_inputs"] = dict(model_inputs)

        if (
            self.processor is not None
            and self.processor.image_processor.__class__.__name__
            == "Qwen2VLImageProcessor"
        ):
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(
                attention_mask.cumsum(dim=0) - 1, min=0, max=None
            )  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )
        data_dict["input_ids"] = input_ids
        data_dict["attention_mask"] = attention_mask
        data_dict["position_ids"] = position_ids
        data_dict["raw_prompt_ids"] = raw_prompt_ids
        data_dict["ground_truth"] = json.loads(data_dict.pop("answer"))
        data_dict["ground_truth"]["reward_name"] = self.reward_name
        return data_dict


class MosaicVQARLHFDataset(Dataset, ImageProcessMixin):
    """JSONL-based dataset for mosaic VQA GRPO training.

    Each line of the annotation file is a JSON object:
        {
            "image_name": "mosaic_00000.jpg",
            "sub_images": [
                {
                    "image_name": "orig.jpg",
                    "top_left": [x, y],
                    "bottom_right": [x, y],
                    "categories": ["cat1", ...]
                },
                ...  (4 quadrants)
            ]
        }

    The dataset collects all unique categories across the four quadrants to
    build the detection prompt (via ``task_fn``).  The per-quadrant category
    information is forwarded in ``ground_truth["sub_images"]`` so the
    ``MosaicVQARewardFunction`` can evaluate predictions against each quadrant.
    """

    def __init__(
        self,
        anno_jsonl_file: str,
        image_dir: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int = 4096,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        reward_name: str = "mosaic_vqa",
        dataset_name: str = "mosaic_vqa",
        task_fn=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts
        self.reward_name = reward_name
        self.dataset_name = dataset_name
        self.image_dir = image_dir

        if task_fn is not None:
            from engine.registry import BUILDER
            self.task_fn = BUILDER.build(task_fn)
        else:
            self.task_fn = None

        self.samples = []
        with open(anno_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

    def __len__(self):
        return len(self.samples)

    def _build_messages(self, data_dict):
        prompt_str = data_dict.get("prompt", "")
        if self.format_prompt:
            tpl = Template(self.format_prompt.strip())
            prompt_str = tpl.render(content=prompt_str)

        content_list = []
        for i, content in enumerate(prompt_str.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})
            if content:
                content_list.append({"type": "text", "text": content})
        return [{"role": "user", "content": content_list}]

    def _convert_to_grpo_format(self, sub_images, image_size, image_name):
        h, w = image_size
        return {
            "raw_image_size": [w, h],
            "resized_image_size": [w, h],
            "sub_images": [
                {
                    "top_left": s["top_left"],
                    "bottom_right": s["bottom_right"],
                    "categories": s["categories"],
                }
                for s in sub_images
            ],
            "reward_name": self.reward_name,
            "dataset_name": self.dataset_name,
            "image_name": image_name,
        }

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample["image_name"]
        sub_images = sample["sub_images"]

        img_path = os.path.join(self.image_dir, image_name)
        image_pil = Image.open(img_path).convert("RGB")
        img_w, img_h = image_pil.size

        fake_boxes = []
        labels = []
        for sub in sub_images:
            tl = sub["top_left"]
            br = sub["bottom_right"]
            for cat in sub["categories"]:
                fake_boxes.append([tl[0], tl[1], br[0], br[1]])
                labels.append(cat)

        grounding_data = {
            "boxes": np.array(fake_boxes) if fake_boxes else np.zeros((0, 4)),
            "labels": labels,
            "size": (img_h, img_w),
        }

        data_dict: Dict[str, Any] = {}

        if self.task_fn is not None:
            example = {"annotations": grounding_data}
            example = self.task_fn(example, img_w, img_h)
            for conv in example.get("conversations", []):
                if conv.get("from") == "human":
                    data_dict["prompt"] = conv["value"]
                    break

        grpo_answer = self._convert_to_grpo_format(
            sub_images, (img_h, img_w), image_name
        )
        data_dict["answer"] = json.dumps(grpo_answer)

        messages = self._build_messages(data_dict)
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        images = [self.process_image(image_pil)]
        model_inputs = self.processor(
            images, [prompt], add_special_tokens=False, return_tensors="pt"
        )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        data_dict["multi_modal_data"] = {"image": images}
        data_dict["multi_modal_inputs"] = dict(model_inputs)

        if (
            self.processor is not None
            and self.processor.image_processor.__class__.__name__
            == "Qwen2VLImageProcessor"
        ):
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )
        else:
            position_ids = torch.clip(
                attention_mask.cumsum(dim=0) - 1, min=0, max=None
            )

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} > {self.max_prompt_length}."
                )

        data_dict["input_ids"] = input_ids
        data_dict["attention_mask"] = attention_mask
        data_dict["position_ids"] = position_ids
        data_dict["raw_prompt_ids"] = raw_prompt_ids
        data_dict["ground_truth"] = json.loads(data_dict.pop("answer"))
        data_dict["ground_truth"]["reward_name"] = self.reward_name
        data_dict["boxes"] = None
        return data_dict


if __name__ == "__main__":

    import random
    from typing import Dict, List

    import transformers
    from transformers import AutoProcessor

    GROUNDING_SINGLE_REGION_STAGE_XYXY = [
        "Detect [OBJ].",
        "detect [OBJ].",
        "detect [OBJ].",
        "detect [OBJ].",
        "detect [OBJ].",
        "detect [OBJ].",
        "Please detect [OBJ] in this image."
        "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "Please detect [OBJ] in this image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "Find [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "Detect [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "Locate [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "Identify [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "Please locate [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "What is the location of [OBJ]? Return its bounding box as [x0, y0, x1, y1].",
        "Mark the region where [OBJ] appears using [x0, y0, x1, y1] format.",
        "Can you find [OBJ] in this picture? Give the coordinates as [x0, y0, x1, y1].",
        "Highlight [OBJ] in the image and output its bounding box in [x0, y0, x1, y1].",
        "Indicate where [OBJ] is located with bounding box coordinates [x0, y0, x1, y1].",
        "Show me the bounding box for [OBJ] in [x0, y0, x1, y1] format.",
        "Return the bounding box coordinates for [OBJ] in the image.",
        "Give the coordinates of the box around [OBJ] using [x0, y0, x1, y1] format.",
        "Determine the bounding box of [OBJ] and return it as [x0, y0, x1, y1].",
        "Identify the bounding box location of [OBJ] using the format [x0, y0, x1, y1].",
        "detect [OBJ].",
        "please detect [OBJ] in this image.",
        "detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "please detect [OBJ] in this image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "find [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "detect [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "locate [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "identify [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "please locate [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        "what is the location of [OBJ]? Return its bounding box as [x0, y0, x1, y1].",
        "mark the region where [OBJ] appears using [x0, y0, x1, y1] format.",
        "can you find [OBJ] in this picture? Give the coordinates as [x0, y0, x1, y1].",
        "highlight [OBJ] in the image and output its bounding box in [x0, y0, x1, y1].",
        "indicate where [OBJ] is located with bounding box coordinates [x0, y0, x1, y1].",
        "show me the bounding box for [OBJ] in [x0, y0, x1, y1] format.",
        "return the bounding box coordinates for [OBJ] in the image.",
        "give the coordinates of the box around [OBJ] using [x0, y0, x1, y1] format.",
        "determine the bounding box of [OBJ] and return it as [x0, y0, x1, y1].",
        "identify the bounding box location of [OBJ] using the format [x0, y0, x1, y1].",
    ]

    class GroundingTaskFn(object):
        """This is for detection dataset tsv training

        Args:
            task_prompts (list[str]): A list of prompts to random choose from.
            image_min_pixels (int): The minimal number of pixels for the resized image.
            image_max_pixels (int): The maximal number of pixels for the resized image.
            extra_categories (List[str], optional): A list of all possible category names. Used for negative sampling.
                If None, only positive examples will be used. Default: None.

        Returns:
            - dict: Dict with the following keys:
                "conversations" (List): [
                    {
                        "from": "human",
                        "value": "<image>\nCan you detect the dog, cat in this image? Answer the question in json format."
                    },
                    {
                        "from": "gpt",
                        "value": "<|object_ref_start|>dog<|object_ref_end|><|box_start|>x0y0x1y1, x0y0x1y1<|box_end|>, <|object_ref_start|>cat<|object_ref_end|><|box_start|>None<|box_end|>"
                    },
                ]
                # ! Note: The coordinates are now normalized to [0, 999] bins. If coord_to_word_map is provided,
                # ! the coordinates will be mapped to corresponding word tokens. Otherwise, they remain as integers.
                # ! For negative examples, the box coordinates will be "None".
        """

        def __init__(
            self,
            task_prompts,
            image_min_pixels,
            image_max_pixels,
            extra_categories: List = None,
            **kwargs,
        ):
            self.min_pixels = image_min_pixels
            self.max_pixels = image_max_pixels
            self.task_prompts = task_prompts
            self.extra_categories = extra_categories

            if extra_categories is not None:
                # modify category name
                self.extra_categories = [
                    self.modify_cateogry_name(category_name)
                    for category_name in extra_categories
                ]

        def modify_cateogry_name(self, category_name: str):
            """Process the category name to be more readable.

            Args:
                region_map (Dict): Region map from the input.
            """
            try:
                if "/" in category_name:
                    # If the category name contains '/', replace it with '_'
                    category_name = category_name.split("/")[0]
                category_name = category_name.replace("_", " ").replace(",", "")
            except Exception as e:
                raise ValueError(f"Error modifying category name: {category_name}")
            return category_name

        def convert_boxes_from_absolute_to_normalized_bins(
            self, gt_boxes, ori_width, ori_height
        ):
            """Convert boxes from absolute coordinates to normalized bins (0-999) and map to words.

            Args:
                gt_boxes: List of boxes in absolute coordinates
                ori_width: Original image width
                ori_height: Original image height

            Returns:
                List of boxes with coordinates mapped to words
            """
            # Step 1: Convert to normalized bins
            normalized_gt_boxes = []
            for box in gt_boxes:
                # Normalize coordinates to [0, 1] range
                x0, y0, x1, y1 = box
                x0_norm = x0 / ori_width
                x1_norm = x1 / ori_width
                y0_norm = y0 / ori_height
                y1_norm = y1 / ori_height

                # Clip to [0, 1] range
                x0_norm = max(0.0, min(1.0, x0_norm))
                x1_norm = max(0.0, min(1.0, x1_norm))
                y0_norm = max(0.0, min(1.0, y0_norm))
                y1_norm = max(0.0, min(1.0, y1_norm))

                # Convert to bins [0, 999]
                x0_bin = int(x0_norm * 999)
                y0_bin = int(y0_norm * 999)
                x1_bin = int(x1_norm * 999)
                y1_bin = int(y1_norm * 999)

                # Ensure bins are in valid range [0, 999]
                x0_bin = max(0, min(999, x0_bin))
                y0_bin = max(0, min(999, y0_bin))
                x1_bin = max(0, min(999, x1_bin))
                y1_bin = max(0, min(999, y1_bin))

                normalized_gt_boxes.append([x0_bin, y0_bin, x1_bin, y1_bin])

            # Step 2: Sort boxes based on x0
            normalized_gt_boxes.sort(key=lambda box: box[0])

            # Step 3: Map to words using coord_to_word_map
            word_mapped_boxes = []
            for box in normalized_gt_boxes:
                x0_bin, y0_bin, x1_bin, y1_bin = box
                # check if x1 > x0 and y1 > y0
                if x1_bin < x0_bin or y1_bin < y0_bin:
                    print(
                        f"x1_bin <= x0_bin or y1_bin <= y0_bin: {x1_bin} <= {x0_bin} or {y1_bin} <= {y0_bin}"
                    )
                    print(f"box: {box}")
                    print(f"normalized_gt_boxes: {normalized_gt_boxes}")
                    print(f"ori_width: {ori_width}, ori_height: {ori_height}")
                word_mapped_boxes.append(
                    "".join(
                        [
                            f"<{x0_bin}>",
                            f"<{y0_bin}>",
                            f"<{x1_bin}>",
                            f"<{y1_bin}>",
                        ]
                    )
                )

            return word_mapped_boxes

        def compose_answer(self, qa_pair):
            """
            Compose the answer for the question.
            """
            answer = []
            for category_name, bboxes in qa_pair.items():
                # Check if this is a negative example (no boxes)
                if bboxes is None or len(bboxes) == 0:
                    # Negative example - object not found in image
                    answer.append(
                        f"<|object_ref_start|>{category_name}<|object_ref_end|><|box_start|>None<|box_end|>"
                    )
                else:
                    # Positive example - format bboxes for output
                    bbox_strings = []
                    for bbox in bboxes:
                        # Join the four coordinate words directly without spaces
                        bbox_strings.append(bbox)
                    bboxes_formatted = ",".join(bbox_strings)

                    answer.append(
                        f"<|object_ref_start|>{category_name}<|object_ref_end|><|box_start|>{bboxes_formatted}<|box_end|>"
                    )
            return ", ".join(answer)

        def step1_compose_qa_pair(self, annotations, ori_width, ori_height):
            """
            Compose a dict for qa pair. The key is the cateogry name and the value is a list of bbox coordinates after resized.

            Args:
                annotations (Dict): Annotation dict with the following keys:
                    {
                        "boxes": List[List[float]], a list of bbox coordinates in xyxy format
                        "labels": List[int], a list of category ids
                        "size": Tuple[int, int], the original size of the image
                    }

            Returns:
                Dict: A dict for qa pair. The key is the cateogry name and the value is a list of bbox coordinates after resized.
            """
            # Get positive categories from current image
            positive_categories = set()
            qa_pair = {}

            for box, label in zip(annotations["boxes"], annotations["labels"]):
                category_name = self.modify_cateogry_name(label)

                positive_categories.add(category_name)
                if category_name not in qa_pair:
                    qa_pair[category_name] = []
                qa_pair[category_name].append(box)

            # Add negative categories if extra_categories is provided
            if self.extra_categories is not None:
                # Get negative categories (categories not in current image)
                negative_categories = set(self.extra_categories) - positive_categories

                # Add all negative categories with None boxes
                for cat in negative_categories:
                    qa_pair[cat] = None

            # Convert positive boxes to normalized bins
            for category_name, bboxes in qa_pair.items():
                if bboxes is not None:  # Only process positive examples
                    qa_pair[category_name] = (
                        self.convert_boxes_from_absolute_to_normalized_bins(
                            bboxes, ori_width, ori_height
                        )
                    )

            # last step, shuffle qa_pair
            qa_pair = dict(sorted(qa_pair.items(), key=lambda x: random.random()))

            return qa_pair

        def __call__(self, example, ori_width, ori_height):
            """
            example (dict): Example from a detection tsv dataset.
                {
                    "image_lineidx" (int): The line index of the image in the tsv file.
                    "annotations" (Dict): Annotation dict with the following keys:
                        {
                            "boxes": List[List[float]], a list of bbox coordinates in xyxy format
                            "labels": List[int], a list of category ids
                            "size": Tuple[int, int], the original size of the image
                        }
                },
            ori_width (int): The original width of the image.
            ori_height (int): The original height of the image.
            """
            # step1 build qa pair
            qa_pair = self.step1_compose_qa_pair(
                example["annotations"], ori_width, ori_height
            )

            # step2 build conversation
            question = random.choice(self.task_prompts)
            conversations = []
            questioned_categories = list(qa_pair.keys())
            question = question.replace("[OBJ]", ", ".join(questioned_categories))

            # conversation from human
            conversation_from_human = {}
            conversation_from_human["from"] = "human"
            conversation_from_human["value"] = f"<image>\n{question}"
            conversations.append(conversation_from_human)
            # conversation from gpt
            conversation_from_gpt = {}
            conversation_from_gpt["from"] = "gpt"
            answer = self.compose_answer(qa_pair)
            conversation_from_gpt["value"] = answer
            conversations.append(conversation_from_gpt)
            example["conversations"] = conversations
            return example

    task_fn = dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=16 * 28 * 28,
        image_max_pixels=2560 * 28 * 28,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/comp_robot/jiangqing/projects/2023/research/R1/QwenSFTOfficial/open_source/IDEA-Research/Rex-Omni",
        padding_side="right",
        use_fast=False,
    )

    processor = AutoProcessor.from_pretrained(
        "/comp_robot/jiangqing/projects/2023/research/R1/QwenSFTOfficial/open_source/IDEA-Research/Rex-Omni",
    )

    dataset = TSVRLHFDataset(
        "/comp_robot/jiangqing/projects/2023/research/R1/QwenSFTOfficial/open_source/Mountchicken/Rex-Omni-Finetune-ToyData/SFT_Grounding_data.images.tsv",
        "/comp_robot/jiangqing/projects/2023/research/R1/QwenSFTOfficial/open_source/Mountchicken/Rex-Omni-Finetune-ToyData/SFT_Grounding_data.annotations.tsv",
        "/comp_robot/jiangqing/projects/2023/research/R1/QwenSFTOfficial/open_source/Mountchicken/Rex-Omni-Finetune-ToyData/SFT_Grounding_data.annotations.tsv.lineidx",
        tokenizer,
        processor,
        min_pixels=16 * 28 * 28,
        max_pixels=2560 * 28 * 28,
        reward_name="box_iou",
        task_fn=task_fn,
    )

    print(dataset[0])
