"""Length-based oversampling for ordinary pointing SFT."""

import copy
import math
from typing import Dict, List

from .base_tsv_dataset import PointingTSVDataset


_QWEN_2_VISUAL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


class PointingOversampleTSVDataset(PointingTSVDataset):
    """Repeat complete pointing-SFT examples by assistant-token length.

    An example with ``N`` assistant target tokens appears
    ``max(1, ceil(N / tokens_per_segment))`` times. Each repeated item keeps
    the complete ordinary pointing SFT sequence and its complete causal-LM loss
    mask; no answer segmentation or context compression is applied.
    """

    def __init__(self, *args, tokens_per_segment: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_per_segment = int(tokens_per_segment)
        if self.tokens_per_segment <= 0:
            raise ValueError("tokens_per_segment must be positive.")

        self._assistant_count_tokenizer = copy.deepcopy(self.tokenizer)
        self._assistant_count_tokenizer.chat_template = _QWEN_2_VISUAL_CHAT_TEMPLATE

        base_data = list(self.data)
        self.oversample_index = self._build_oversample_index()

        # PointingTSVDataset reads annotation offsets directly from self.data.
        # Repeating those lightweight offsets preserves inherited loading,
        # point conversion, preprocessing, and retry behavior.
        self.data = [
            base_data[int(sample_info["sample_idx"])]
            for sample_info in self.oversample_index
        ]

        # Index construction opens TSV handles in the parent dataset. Workers
        # must open their own handles after the dataset is forked/spawned.
        if self.img_handle is not None:
            self.img_handle.close()
            self.img_handle = None
        if self.ann_handle is not None:
            self.ann_handle.close()
            self.ann_handle = None

        max_repeat = max(
            (int(item["num_segments"]) for item in self.oversample_index),
            default=0,
        )
        print(
            "ordinary pointing SFT length oversampling: "
            f"tokens_per_segment={self.tokens_per_segment}, "
            f"base_samples={len(base_data)}, "
            f"virtual_samples={len(self.oversample_index)}, "
            f"max_repeat={max_repeat}"
        )

    def _assistant_target_token_count(self, answer: str) -> int:
        encoded = self._assistant_count_tokenizer.apply_chat_template(
            [{"role": "assistant", "content": answer}]
        )
        # preprocess_qwen_2_visual masks the three-token assistant header.
        return max(0, len(encoded) - 3)

    def _build_oversample_index(self) -> List[Dict[str, int]]:
        oversample_index: List[Dict[str, int]] = []
        for sample_idx in range(len(self.data)):
            try:
                image_pil, raw_data = self.load_image_and_anno(sample_idx)
                data_dict = self.convert_raw_data(image_pil, raw_data)
                if data_dict is None or len(data_dict.get("points", [])) == 0:
                    continue

                ori_width, ori_height = image_pil.size
                if self.task_fn is not None:
                    data_dict = self.task_fn(
                        {"id2name": self.id2name, "annotations": data_dict},
                        ori_width,
                        ori_height,
                    )

                conversations = data_dict["conversations"]
                answer_message = conversations[-1]
                answer = answer_message.get(
                    "value", answer_message.get("content", "")
                )
                num_answer_tokens = self._assistant_target_token_count(answer)
                num_segments = max(
                    1,
                    math.ceil(num_answer_tokens / self.tokens_per_segment),
                )

                for repeat_id in range(num_segments):
                    oversample_index.append(
                        {
                            "sample_idx": sample_idx,
                            "repeat_id": repeat_id,
                            "num_segments": num_segments,
                            "num_answer_tokens": num_answer_tokens,
                        }
                    )
            except Exception as exc:
                print(
                    f"error building ordinary pointing SFT oversample index "
                    f"{self.dataset_name} id: {sample_idx}. {exc}"
                )
        return oversample_index
