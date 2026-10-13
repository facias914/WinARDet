import math
import random
from typing import Dict, List


class SegGroundingTaskFn(object):
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
        self.use_visual_token_order_box_sort = kwargs.get(
            "use_visual_token_order_box_sort", False
        )
        self.visual_token_order_grid_size = kwargs.get("visual_token_order_grid_size", 32)
        self.visual_patch_size = kwargs.get("visual_patch_size", 14)
        self.visual_merge_size = kwargs.get("visual_merge_size", 2)
        self._current_image_size = None

        if self.visual_token_order_grid_size <= 0:
            raise ValueError("visual_token_order_grid_size must be positive.")
        if self.visual_patch_size <= 0:
            raise ValueError("visual_patch_size must be positive.")
        if self.visual_merge_size <= 0:
            raise ValueError("visual_merge_size must be positive.")

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
            # if "/" in category_name:
            #     # If the category name contains '/', replace it with '_'
            #     category_name = category_name.split("/")[0]
            category_name = category_name.replace("_", " ").replace(",", "")
        except Exception as e:
            raise ValueError(f"Error modifying category name: {category_name}")
        return category_name

    def sort_boxes_for_autoregressive_answer(self, normalized_gt_boxes):
        """Sort boxes to define the autoregressive detection assignment order.

        Qwen2.5-VL visual tokens are flattened row-major in image space: width
        changes fastest, then height. When enabled, this assigns each box center
        to the exact LLM-side visual-token grid for the current image.
        """
        if not self.use_visual_token_order_box_sort:
            return sorted(normalized_gt_boxes, key=lambda box: box[0])

        return self.sort_boxes_for_autoregressive_answer_by_visual_token_grid(
            normalized_gt_boxes
        )

    def _round_by_factor(self, value, factor):
        return round(value / factor) * factor

    def _ceil_by_factor(self, value, factor):
        return math.ceil(value / factor) * factor

    def _floor_by_factor(self, value, factor):
        return math.floor(value / factor) * factor

    def _smart_resize_for_visual_tokens(self, height, width):
        """Mirror Qwen-VL smart resize for image_grid_thw computation."""
        factor = self.visual_patch_size * self.visual_merge_size
        min_pixels = self.min_pixels
        max_pixels = self.max_pixels

        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image size: {width}x{height}")
        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"Absolute aspect ratio must be smaller than 200, got {width}x{height}."
            )

        resized_height = max(factor, self._round_by_factor(height, factor))
        resized_width = max(factor, self._round_by_factor(width, factor))

        if resized_height * resized_width > max_pixels:
            scale = math.sqrt((height * width) / max_pixels)
            resized_height = self._floor_by_factor(height / scale, factor)
            resized_width = self._floor_by_factor(width / scale, factor)
        elif resized_height * resized_width < min_pixels:
            scale = math.sqrt(min_pixels / (height * width))
            resized_height = self._ceil_by_factor(height * scale, factor)
            resized_width = self._ceil_by_factor(width * scale, factor)

        resized_height = max(factor, resized_height)
        resized_width = max(factor, resized_width)
        return int(resized_height), int(resized_width)

    def _get_current_processed_image_grid(self):
        if self._current_image_size is None:
            raise ValueError(
                "Current image size is required for exact visual-token grid sorting. "
                "Call convert_boxes_from_absolute_to_normalized_bins first."
            )

        ori_width, ori_height = self._current_image_size
        resized_height, resized_width = self._smart_resize_for_visual_tokens(
            ori_height, ori_width
        )
        grid_h = resized_height // (self.visual_patch_size * self.visual_merge_size)
        grid_w = resized_width // (self.visual_patch_size * self.visual_merge_size)
        return {
            "ori_width": ori_width,
            "ori_height": ori_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "grid_h": max(1, grid_h),
            "grid_w": max(1, grid_w),
        }

    def _get_current_llm_visual_grid_size(self):
        grid = self._get_current_processed_image_grid()
        return grid["grid_h"], grid["grid_w"]

    def sort_boxes_for_autoregressive_answer_by_visual_token_grid(
        self, normalized_gt_boxes
    ):
        """Sort boxes by the exact LLM-side visual token grid for this image.

        This has the same interface as sort_boxes_for_autoregressive_answer.
        It uses the current image size recorded by the caller, mirrors Qwen-VL
        resize alignment, derives the same H x W visual-token grid used after
        spatial merge, and sorts boxes in row-major visual token order.
        """
        grid = self._get_current_processed_image_grid()
        grid_h = grid["grid_h"]
        grid_w = grid["grid_w"]
        resized_height = grid["resized_height"]
        resized_width = grid["resized_width"]
        visual_token_size = self.visual_patch_size * self.visual_merge_size

        def sort_key(box):
            x0, y0, x1, y1 = box
            cx_bin = (x0 + x1) * 0.5
            cy_bin = (y0 + y1) * 0.5

            # Map answer coordinate bins back onto the processed image plane,
            # then assign by the same patch*merge grid seen by the visual encoder.
            cx_processed = (cx_bin / 999.0) * resized_width
            cy_processed = (cy_bin / 999.0) * resized_height
            grid_x = max(0, min(grid_w - 1, int(cx_processed // visual_token_size)))
            grid_y = max(0, min(grid_h - 1, int(cy_processed // visual_token_size)))
            token_index = grid_y * grid_w + grid_x
            local_x = cx_processed - grid_x * visual_token_size
            local_y = cy_processed - grid_y * visual_token_size
            area = max(0, x1 - x0) * max(0, y1 - y0)
            return (token_index, local_y, local_x, area, y0, x0)

        return sorted(normalized_gt_boxes, key=sort_key)

    def sort_boxes_for_autoregressive_answer_by_fixed_visual_grid(
        self, normalized_gt_boxes
    ):
        """Legacy fixed normalized-grid sorter kept for analysis/debug use."""
        grid_size = self.visual_token_order_grid_size
        cell_size = 1000.0 / grid_size

        def sort_key(box):
            x0, y0, x1, y1 = box
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            grid_x = max(0, min(grid_size - 1, int(cx // cell_size)))
            grid_y = max(0, min(grid_size - 1, int(cy // cell_size)))
            token_index = grid_y * grid_size + grid_x
            local_x = cx - grid_x * cell_size
            local_y = cy - grid_y * cell_size
            area = max(0, x1 - x0) * max(0, y1 - y0)
            return (token_index, local_y, local_x, area, y0, x0)

        return sorted(normalized_gt_boxes, key=sort_key)

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
        self._current_image_size = (ori_width, ori_height)
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

        # Step 2: Sort boxes to match the autoregressive answer assignment order.
        normalized_gt_boxes = self.sort_boxes_for_autoregressive_answer(
            normalized_gt_boxes
        )

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
                    "boxes": List[List[float] or None], a list of bbox coordinates in xyxy format (None for negative samples)
                    "labels": List[str], a list of category names
                    "size": Tuple[int, int], the original size of the image
                }

        Returns:
            Dict: A dict for qa pair. The key is the cateogry name and the value is a list of bbox coordinates after resized.
        """
        # Get positive and negative categories from current image
        positive_categories = {}  # category -> list of boxes
        negative_categories = set()  # categories with None boxes

        for box, label in zip(annotations["boxes"], annotations["labels"]):
            category_name = self.modify_cateogry_name(label)

            if box is not None:
                # Positive sample: valid box
                if category_name not in positive_categories:
                    positive_categories[category_name] = []
                positive_categories[category_name].append(box)
            else:
                # Negative sample: box is None
                negative_categories.add(category_name)

        # Build qa_pair
        qa_pair = {}

        # Add positive categories with their boxes
        for category_name, boxes in positive_categories.items():
            qa_pair[category_name] = boxes

        # Add negative categories with None
        for category_name in negative_categories:
            if (
                category_name not in qa_pair
            ):  # Only add if not already a positive sample
                qa_pair[category_name] = None

        # Add extra negative categories if provided
        if self.extra_categories is not None:
            all_present_categories = (
                set(positive_categories.keys()) | negative_categories
            )
            extra_negative_categories = (
                set(self.extra_categories) - all_present_categories
            )
            for cat in extra_negative_categories:
                qa_pair[cat] = None

        # Convert positive boxes to normalized bins
        for category_name, bboxes in qa_pair.items():
            if bboxes is not None:  # Only process positive examples
                qa_pair[category_name] = (
                    self.convert_boxes_from_absolute_to_normalized_bins(
                        bboxes, ori_width, ori_height
                    )
                )

        # Keep category order deterministic and shared by prompt/answer.
        qa_pair = dict(sorted(qa_pair.items(), key=lambda x: x[0].casefold()))

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
