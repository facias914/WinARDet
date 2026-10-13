from .sacom_grounding_task import SegGroundingTaskFn


class SegPointingTaskFn(SegGroundingTaskFn):
    """Segment-compression task function for point grounding.

    Point coordinates are expected as [x, y] in absolute image pixels and are
    converted to normalized 0-999 bins.
    """

    def modify_cateogry_name(self, category_name: str):
        try:
            if "/" in category_name:
                category_name = category_name.split("/")[0]
            category_name = category_name.replace("_", " ").replace(",", "")
        except Exception as exc:
            raise ValueError(f"Error modifying category name: {category_name}") from exc
        return category_name

    def sort_points_for_autoregressive_answer(self, normalized_gt_points):
        if not self.use_visual_token_order_box_sort:
            return sorted(normalized_gt_points, key=lambda point: point[0])

        return self.sort_points_for_autoregressive_answer_by_visual_token_grid(
            normalized_gt_points
        )

    def sort_points_for_autoregressive_answer_by_visual_token_grid(
        self, normalized_gt_points
    ):
        grid = self._get_current_processed_image_grid()
        grid_h = grid["grid_h"]
        grid_w = grid["grid_w"]
        resized_height = grid["resized_height"]
        resized_width = grid["resized_width"]
        visual_token_size = self.visual_patch_size * self.visual_merge_size

        def sort_key(point):
            x_bin, y_bin = point
            x_processed = (x_bin / 999.0) * resized_width
            y_processed = (y_bin / 999.0) * resized_height
            grid_x = max(0, min(grid_w - 1, int(x_processed // visual_token_size)))
            grid_y = max(0, min(grid_h - 1, int(y_processed // visual_token_size)))
            token_index = grid_y * grid_w + grid_x
            local_x = x_processed - grid_x * visual_token_size
            local_y = y_processed - grid_y * visual_token_size
            return (token_index, local_y, local_x, y_bin, x_bin)

        return sorted(normalized_gt_points, key=sort_key)

    def sort_points_for_autoregressive_answer_by_fixed_visual_grid(
        self, normalized_gt_points
    ):
        grid_size = self.visual_token_order_grid_size
        cell_size = 1000.0 / grid_size

        def sort_key(point):
            x_bin, y_bin = point
            grid_x = max(0, min(grid_size - 1, int(x_bin // cell_size)))
            grid_y = max(0, min(grid_size - 1, int(y_bin // cell_size)))
            token_index = grid_y * grid_size + grid_x
            local_x = x_bin - grid_x * cell_size
            local_y = y_bin - grid_y * cell_size
            return (token_index, local_y, local_x, y_bin, x_bin)

        return sorted(normalized_gt_points, key=sort_key)

    def convert_points_from_absolute_to_normalized_bins(
        self, gt_points, ori_width, ori_height
    ):
        self._current_image_size = (ori_width, ori_height)
        normalized_gt_points = []
        for point in gt_points:
            if len(point) != 2:
                raise ValueError(
                    f"Point annotation must contain 2 coordinates, got {len(point)}: {point}"
                )

            x, y = point
            x_norm = max(0.0, min(1.0, x / ori_width))
            y_norm = max(0.0, min(1.0, y / ori_height))
            x_bin = max(0, min(999, int(x_norm * 999)))
            y_bin = max(0, min(999, int(y_norm * 999)))
            normalized_gt_points.append([x_bin, y_bin])

        normalized_gt_points = self.sort_points_for_autoregressive_answer(
            normalized_gt_points
        )

        word_mapped_points = []
        for point in normalized_gt_points:
            x_bin, y_bin = point
            word_mapped_points.append("".join([f"<{x_bin}>", f"<{y_bin}>"]))

        return word_mapped_points

    def step1_compose_qa_pair(self, annotations, ori_width, ori_height):
        positive_categories = {}
        negative_categories = set()

        for point, label in zip(annotations["points"], annotations["labels"]):
            category_name = self.modify_cateogry_name(label)

            if point is not None:
                if category_name not in positive_categories:
                    positive_categories[category_name] = []
                positive_categories[category_name].append(point)
            else:
                negative_categories.add(category_name)

        qa_pair = {}
        for category_name, points in positive_categories.items():
            qa_pair[category_name] = points

        for category_name in negative_categories:
            if category_name not in qa_pair:
                qa_pair[category_name] = None

        if self.extra_categories is not None:
            all_present_categories = (
                set(positive_categories.keys()) | negative_categories
            )
            extra_negative_categories = (
                set(self.extra_categories) - all_present_categories
            )
            for cat in extra_negative_categories:
                qa_pair[cat] = None

        for category_name, points in qa_pair.items():
            if points is not None:
                qa_pair[category_name] = (
                    self.convert_points_from_absolute_to_normalized_bins(
                        points, ori_width, ori_height
                    )
                )

        return dict(sorted(qa_pair.items(), key=lambda x: x[0].casefold()))
