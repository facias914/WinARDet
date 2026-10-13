import random
from typing import Dict, List


class PointingTaskFn(object):
    """This is for pointing dataset tsv training

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
                    "value": "<image>\nCan you point to the dog, cat in this image?"
                },
                {
                    "from": "gpt",
                    "value": "<|object_ref_start|>dog<|object_ref_end|><|box_start|><123><456>,<789><890><|box_end|>, <|object_ref_start|>cat<|object_ref_end|><|box_start|>None<|box_end|>"
                },
            ]
            # ! Note: The coordinates are now normalized to [0, 999] bins.
            # ! For negative examples, the point coordinates will be "None".
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
            category_name (str): Category name from the input.
        """
        try:
            if "/" in category_name:
                # If the category name contains '/', replace it with '_'
                category_name = category_name.split("/")[0]
            category_name = category_name.replace("_", " ").replace(",", "")
        except Exception as e:
            raise ValueError(f"Error modifying category name: {category_name}")
        return category_name

    def convert_points_from_absolute_to_normalized_bins(
        self, gt_points, ori_width, ori_height
    ):
        """Convert points from absolute coordinates to normalized bins (0-999) and map to words.

        Args:
            gt_points: List of points in absolute coordinates [x, y]
            ori_width: Original image width
            ori_height: Original image height

        Returns:
            List of points with coordinates mapped to words
        """
        # Step 1: Convert to normalized bins
        normalized_gt_points = []
        for point in gt_points:
            # Normalize coordinates to [0, 1] range
            x, y = point
            x_norm = x / ori_width
            y_norm = y / ori_height

            # Clip to [0, 1] range
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))

            # Convert to bins [0, 999]
            x_bin = int(x_norm * 999)
            y_bin = int(y_norm * 999)

            # Ensure bins are in valid range [0, 999]
            x_bin = max(0, min(999, x_bin))
            y_bin = max(0, min(999, y_bin))

            normalized_gt_points.append([x_bin, y_bin])

        # Step 2: Sort points based on x
        normalized_gt_points.sort(key=lambda point: point[0])

        # Step 3: Map to words
        word_mapped_points = []
        for point in normalized_gt_points:
            x_bin, y_bin = point
            word_mapped_points.append("".join([f"<{x_bin}>", f"<{y_bin}>"]))

        return word_mapped_points

    def compose_answer(self, qa_pair):
        """
        Compose the answer for the question.
        """
        answer = []
        for category_name, points in qa_pair.items():
            # Check if this is a negative example (no points)
            if points is None or len(points) == 0:
                # Negative example - object not found in image
                answer.append(
                    f"<|object_ref_start|>{category_name}<|object_ref_end|><|box_start|>None<|box_end|>"
                )
            else:
                # Positive example - format points for output
                point_strings = []
                for point in points:
                    # Join the two coordinate words directly without spaces
                    point_strings.append(point)
                points_formatted = ",".join(point_strings)

                answer.append(
                    f"<|object_ref_start|>{category_name}<|object_ref_end|><|box_start|>{points_formatted}<|box_end|>"
                )
        return ", ".join(answer)

    def step1_compose_qa_pair(self, annotations, ori_width, ori_height):
        """
        Compose a dict for qa pair. The key is the category name and the value is a list of point coordinates after resized.

        Args:
            annotations (Dict): Annotation dict with the following keys:
                {
                    "points": List[List[float] or None], a list of point coordinates in xy format (None for negative samples)
                    "labels": List[str or int], a list of category names or ids
                    "size": Tuple[int, int], the original size of the image
                }

        Returns:
            Dict: A dict for qa pair. The key is the category name and the value is a list of point coordinates after resized.
        """
        # Get positive and negative categories from current image
        positive_categories = {}  # category -> list of points
        negative_categories = set()  # categories with None points

        for point, label in zip(annotations["points"], annotations["labels"]):
            category_name = self.modify_cateogry_name(label)
            
            if point is not None:
                # Positive sample: valid point
                if category_name not in positive_categories:
                    positive_categories[category_name] = []
                positive_categories[category_name].append(point)
            else:
                # Negative sample: point is None
                negative_categories.add(category_name)

        # Build qa_pair
        qa_pair = {}
        
        # Add positive categories with their points
        for category_name, points in positive_categories.items():
            qa_pair[category_name] = points
        
        # Add negative categories with None
        for category_name in negative_categories:
            if category_name not in qa_pair:  # Only add if not already a positive sample
                qa_pair[category_name] = None

        # Add extra negative categories if provided
        if self.extra_categories is not None:
            all_present_categories = set(positive_categories.keys()) | negative_categories
            extra_negative_categories = set(self.extra_categories) - all_present_categories
            for cat in extra_negative_categories:
                qa_pair[cat] = None

        # Convert positive points to normalized bins
        for category_name, points in qa_pair.items():
            if points is not None:  # Only process positive examples
                qa_pair[category_name] = (
                    self.convert_points_from_absolute_to_normalized_bins(
                        points, ori_width, ori_height
                    )
                )

        # last step, shuffle qa_pair
        qa_pair = dict(sorted(qa_pair.items(), key=lambda x: random.random()))

        return qa_pair

    def __call__(self, example, ori_width, ori_height):
        """
        example (dict): Example from a pointing tsv dataset.
            {
                "image_lineidx" (int): The line index of the image in the tsv file.
                "annotations" (Dict): Annotation dict with the following keys:
                    {
                        "points": List[List[float]], a list of point coordinates in xy format
                        "labels": List[str or int], a list of category names or ids
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
