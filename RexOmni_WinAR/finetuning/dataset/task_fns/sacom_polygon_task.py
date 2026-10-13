from .sacom_grounding_task import SegGroundingTaskFn


class SegPolygonTaskFn(SegGroundingTaskFn):
    """Segment-compression task function for polygon grounding.

    Polygon coordinates are expected as [x0, y0, x1, y1, x2, y2, x3, y3]
    in absolute image pixels and are converted to normalized 0-999 bins.
    """

    def sort_boxes_for_autoregressive_answer(self, normalized_gt_boxes):
        if not self.use_visual_token_order_box_sort:
            return sorted(normalized_gt_boxes, key=lambda polygon: min(polygon[0::2]))

        return self.sort_boxes_for_autoregressive_answer_by_visual_token_grid(
            normalized_gt_boxes
        )

    def sort_boxes_for_autoregressive_answer_by_visual_token_grid(
        self, normalized_gt_boxes
    ):
        grid = self._get_current_processed_image_grid()
        grid_h = grid["grid_h"]
        grid_w = grid["grid_w"]
        resized_height = grid["resized_height"]
        resized_width = grid["resized_width"]
        visual_token_size = self.visual_patch_size * self.visual_merge_size

        def sort_key(polygon):
            xs = polygon[0::2]
            ys = polygon[1::2]
            cx_bin = sum(xs) / len(xs)
            cy_bin = sum(ys) / len(ys)

            cx_processed = (cx_bin / 999.0) * resized_width
            cy_processed = (cy_bin / 999.0) * resized_height
            grid_x = max(0, min(grid_w - 1, int(cx_processed // visual_token_size)))
            grid_y = max(0, min(grid_h - 1, int(cy_processed // visual_token_size)))
            token_index = grid_y * grid_w + grid_x
            local_x = cx_processed - grid_x * visual_token_size
            local_y = cy_processed - grid_y * visual_token_size
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            area = max(0, width) * max(0, height)
            return (token_index, local_y, local_x, area, min(ys), min(xs))

        return sorted(normalized_gt_boxes, key=sort_key)

    def sort_boxes_for_autoregressive_answer_by_fixed_visual_grid(
        self, normalized_gt_boxes
    ):
        grid_size = self.visual_token_order_grid_size
        cell_size = 1000.0 / grid_size

        def sort_key(polygon):
            xs = polygon[0::2]
            ys = polygon[1::2]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            grid_x = max(0, min(grid_size - 1, int(cx // cell_size)))
            grid_y = max(0, min(grid_size - 1, int(cy // cell_size)))
            token_index = grid_y * grid_size + grid_x
            local_x = cx - grid_x * cell_size
            local_y = cy - grid_y * cell_size
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            area = max(0, width) * max(0, height)
            return (token_index, local_y, local_x, area, min(ys), min(xs))

        return sorted(normalized_gt_boxes, key=sort_key)

    def convert_boxes_from_absolute_to_normalized_bins(
        self, gt_boxes, ori_width, ori_height
    ):
        self._current_image_size = (ori_width, ori_height)
        normalized_gt_polygons = []
        for polygon in gt_boxes:
            if len(polygon) != 8:
                raise ValueError(
                    f"Polygon annotation must contain 8 coordinates, got {len(polygon)}: {polygon}"
                )

            normalized_polygon = []
            for idx, coord in enumerate(polygon):
                limit = ori_width if idx % 2 == 0 else ori_height
                coord_norm = coord / limit
                coord_norm = max(0.0, min(1.0, coord_norm))
                coord_bin = int(coord_norm * 999)
                coord_bin = max(0, min(999, coord_bin))
                normalized_polygon.append(coord_bin)

            normalized_gt_polygons.append(normalized_polygon)

        normalized_gt_polygons = self.sort_boxes_for_autoregressive_answer(
            normalized_gt_polygons
        )

        word_mapped_polygons = []
        for polygon in normalized_gt_polygons:
            word_mapped_polygons.append("".join(f"<{coord}>" for coord in polygon))

        return word_mapped_polygons
