import torch

from .segment_grounding_dataset import GroundingSegmentTSVDataset, SegmentDataCollatorForSupervisedDataset


class SegmentPolygonTSVDataset(GroundingSegmentTSVDataset):
    """Segmented TSV dataset for rotated boxes represented by 4 polygon points.

    It reuses GroundingSegmentTSVDataset compression/segment mechanics and adapts
    the annotation schema plus coordinate soft-loss metadata from 4-coordinate
    horizontal boxes to 8-coordinate polygons.
    """

    def __init__(self, *args, polygon_key="bbox", **kwargs):
        self.polygon_key = polygon_key
        super().__init__(*args, **kwargs)

    def convert_raw_data(self, image_pil, data_dict):
        polygons = []
        labels = []
        for anno in data_dict["boxes"]:
            polygon = anno.get(self.polygon_key, anno.get("polygon", anno.get("rbbox")))
            phrase = anno.get("phrase", anno.get("caption", None))

            if polygon is None:
                polygons.append(None)
            else:
                if len(polygon) != 8:
                    raise ValueError(
                        f"Polygon annotation must contain 8 coordinates, got {len(polygon)}: {polygon}"
                    )
                polygons.append(polygon)

            labels.append(phrase)

        return dict(
            boxes=polygons,
            labels=labels,
            size=(image_pil.height, image_pil.width),
        )

    def _build_answer_coord_metadata(self, answer_token_ids: torch.Tensor):
        metadata = self._empty_coord_metadata(answer_token_ids.numel())
        if not self.use_coord_soft_loss:
            return metadata

        inside_box = False
        current_polygon = []
        token_ids = [int(token_id) for token_id in answer_token_ids.tolist()]
        for idx, token_id in enumerate(token_ids):
            if token_id == self.box_start_token_id:
                inside_box = True
                current_polygon = []
                continue
            if token_id == self.box_end_token_id:
                inside_box = False
                current_polygon = []
                continue
            if not inside_box:
                continue

            coord_value = self.coord_token_id_to_value.get(token_id)
            if coord_value is None:
                continue

            current_polygon.append((idx, coord_value))
            if len(current_polygon) != 8:
                continue

            xs = [value for role, (_, value) in enumerate(current_polygon) if role % 2 == 0]
            ys = [value for role, (_, value) in enumerate(current_polygon) if role % 2 == 1]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            if width > 0 and height > 0:
                for role, (coord_idx, value) in enumerate(current_polygon):
                    metadata["coord_mask"][coord_idx] = True
                    metadata["coord_values"][coord_idx] = value
                    metadata["coord_roles"][coord_idx] = role
                    metadata["box_widths"][coord_idx] = float(width)
                    metadata["box_heights"][coord_idx] = float(height)
            current_polygon = []

        return metadata


SegmentPolygonDataCollatorForSupervisedDataset = SegmentDataCollatorForSupervisedDataset
