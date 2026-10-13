from .grounding_task import GroundingTaskFn


class PolygonTaskFn(GroundingTaskFn):
    """Task function for polygon grounding with rotated boxes.

    Polygon coordinates are expected as [x0, y0, x1, y1, x2, y2, x3, y3]
    in absolute image pixels and are converted to normalized 0-999 bins.
    The answer keeps the same object and box delimiters as grounding tasks.
    """

    def convert_boxes_from_absolute_to_normalized_bins(
        self, gt_boxes, ori_width, ori_height
    ):
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

        normalized_gt_polygons.sort(key=lambda polygon: min(polygon[0::2]))

        word_mapped_polygons = []
        for polygon in normalized_gt_polygons:
            word_mapped_polygons.append("".join(f"<{coord}>" for coord in polygon))

        return word_mapped_polygons
