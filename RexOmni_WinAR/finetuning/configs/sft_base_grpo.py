from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

grounding_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/u1120250365/share/Rex_Omni/Compression/train/DOTA_SODA_GRPO/images.tsv",
    anno_tsv_file="/home/u1120250365/share/Rex_Omni/Compression/train/DOTA_SODA_GRPO/annotations.tsv",
    anno_idx_file="/home/u1120250365/share/Rex_Omni/Compression/train/DOTA_SODA_GRPO/annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="rexomni_grounding_data",
    reward_name="box_iou",
)

train_dataset = [
    grounding_data
]
