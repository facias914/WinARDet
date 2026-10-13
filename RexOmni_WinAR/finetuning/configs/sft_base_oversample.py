from dataset import (
    ConcatDataset,
    DataCollatorForSupervisedDataset,
    GroundingOversampleTSVDataset
)
from dataset.task_fns import SegGroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import GROUNDING_SINGLE_REGION_STAGE_XYXY

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28
max_length=20000
model_name_or_path = "/home/u1120250365/Rex-Omni/Rex-Omni"


# Match the virtual-sample weighting used by segment-compression training.
# A full ordinary-SFT example is repeated ceil(answer_tokens / 200) times.
max_tokens_per_segment = 300

use_visual_token_order_box_sort = False
visual_token_order_grid_size = 28
visual_patch_size = 14
visual_merge_size = 2


grounding_data_dota2 = dict(
    type=GroundingOversampleTSVDataset,
    img_tsv_file="/home/share/Rex_Omni/Compression/train/DOTA2/images.tsv",
    ann_tsv_file="/home/share/Rex_Omni/Compression/train/DOTA2/annotations.tsv",
    ann_lineidx_file="/home/share/Rex_Omni/Compression/train/DOTA2/annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_length=max_length,
    tokens_per_segment=max_tokens_per_segment,
    task_fn=dict(
        type=SegGroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
        use_visual_token_order_box_sort=use_visual_token_order_box_sort,
        visual_token_order_grid_size=visual_token_order_grid_size,
        visual_patch_size=visual_patch_size,
        visual_merge_size=visual_merge_size,
    ),
    dataset_name="rexomni_grounding_data",
)

grounding_data_sodaa = dict(
    type=GroundingOversampleTSVDataset,
    img_tsv_file="/home/share/Rex_Omni/Compression/train/SODA-A/images.tsv",
    ann_tsv_file="/home/share/Rex_Omni/Compression/train/SODA-A/annotations.tsv",
    ann_lineidx_file="/home/share/Rex_Omni/Compression/train/SODA-A/annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_length=max_length,
    tokens_per_segment=max_tokens_per_segment,
    task_fn=dict(
        type=SegGroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
        use_visual_token_order_box_sort=use_visual_token_order_box_sort,
        visual_token_order_grid_size=visual_token_order_grid_size,
        visual_patch_size=visual_patch_size,
        visual_merge_size=visual_merge_size,
    ),
    dataset_name="rexomni_grounding_data",
)


train_dataset = dict(
    type=ConcatDataset,
    datasets=[grounding_data_dota2, grounding_data_sodaa],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
