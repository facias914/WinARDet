from dataset import (
    ConcatDataset,
    GroundingSegmentTSVDataset,
    SegmentDataCollatorForSupervisedDataset,
)
from dataset.task_fns import SegGroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import GROUNDING_SINGLE_REGION_STAGE_XYXY

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28
max_length=20000
model_name_or_path = "/home/u1120250365/Rex-Omni/Rex-Omni"

use_context_compression = True
compression_train_mode = "segment_causal"
max_tokens_per_segment = 300
compressor_passthrough_tokens_per_side = 6
num_memory_tokens = 8
compressor_hidden_dim = None
compressor_num_layers = 6
compressor_num_heads = 8
detach_compressor_inputs = True
history_compression_no_grad = False
lambda_cls = 0.0
lambda_cov = 0.0

use_coord_soft_loss = False
lambda_coord_soft = 0.3
coord_soft_alpha = 0.025
coord_soft_sigma_min = 0.5
coord_soft_sigma_max = 0.8
coord_vocab_size = 1000

use_visual_token_order_box_sort = False
visual_token_order_grid_size = 28
visual_patch_size = 14
visual_merge_size = 2


grounding_data_dota2 = dict(
    type=GroundingSegmentTSVDataset,
    img_tsv_file="/home/share/Rex_Omni/Compression/train/DOTA2/images.tsv",
    ann_tsv_file="/home/share/Rex_Omni/Compression/train/DOTA2/annotations.tsv",
    ann_lineidx_file="/home/share/Rex_Omni/Compression/train/DOTA2/annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_length=max_length,
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
    type=GroundingSegmentTSVDataset,
    img_tsv_file="/home/share/Rex_Omni/Compression/train/SODA-A/images.tsv",
    ann_tsv_file="/home/share/Rex_Omni/Compression/train/SODA-A/annotations.tsv",
    ann_lineidx_file="/home/share/Rex_Omni/Compression/train/SODA-A/annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_length=max_length,
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

data_collator = dict(type=SegmentDataCollatorForSupervisedDataset)
