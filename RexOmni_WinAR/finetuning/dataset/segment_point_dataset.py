import torch

from .segment_grounding_dataset import (
    GroundingSegmentTSVDataset,
    SegmentDataCollatorForSupervisedDataset,
)


class SegmentPointingTSVDataset(GroundingSegmentTSVDataset):
    """Segmented TSV dataset for point annotations.

    It reuses GroundingSegmentTSVDataset compression/segment mechanics and
    adapts the raw annotation schema from boxes to points.
    """

    def convert_raw_data(self, image_pil, data_dict):
        points = []
        labels = []
        for anno in data_dict["points"]:
            point = anno.get("point")
            phrase = anno.get("phrase", anno.get("caption", None))

            if point is None:
                points.append(None)
            else:
                if len(point) != 2:
                    raise ValueError(
                        f"Point annotation must contain 2 coordinates, got {len(point)}: {point}"
                    )
                points.append(point)

            labels.append(phrase)

        return dict(
            points=points,
            boxes=points,
            labels=labels,
            size=(image_pil.height, image_pil.width),
        )

    def _build_answer_coord_metadata(self, answer_token_ids: torch.Tensor):
        metadata = self._empty_coord_metadata(answer_token_ids.numel())
        if not self.use_coord_soft_loss:
            return metadata

        inside_box = False
        current_point = []
        token_ids = [int(token_id) for token_id in answer_token_ids.tolist()]
        for idx, token_id in enumerate(token_ids):
            if token_id == self.box_start_token_id:
                inside_box = True
                current_point = []
                continue
            if token_id == self.box_end_token_id:
                inside_box = False
                current_point = []
                continue
            if not inside_box:
                continue

            coord_value = self.coord_token_id_to_value.get(token_id)
            if coord_value is None:
                continue

            current_point.append((idx, coord_value))
            if len(current_point) != 2:
                continue

            for role, (coord_idx, value) in enumerate(current_point):
                metadata["coord_mask"][coord_idx] = True
                metadata["coord_values"][coord_idx] = value
                metadata["coord_roles"][coord_idx] = role
                metadata["box_widths"][coord_idx] = 1.0
                metadata["box_heights"][coord_idx] = 1.0
            current_point = []

        return metadata


SegmentPointingDataCollatorForSupervisedDataset = SegmentDataCollatorForSupervisedDataset
