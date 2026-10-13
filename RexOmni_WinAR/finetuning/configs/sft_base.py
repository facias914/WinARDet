from dataset import (
    ConcatDataset,
    DataCollatorForSupervisedDataset,
    GroundingTSVDataset
)
from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import GROUNDING_SINGLE_REGION_STAGE_XYXY

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28
max_length=20000
model_name_or_path = "/home/u1120230285/wcx/Rex-Omni/Rex-Omni"

grounding_data_dota2 = dict(
    type=GroundingTSVDataset,
    img_tsv_file="/home/u1120230285/share/Rex_Omni/Compression/train/DOTA2/images.tsv",
    ann_tsv_file="/home/u1120230285/share/Rex_Omni/Compression/train/DOTA2/annotations.tsv",
    ann_lineidx_file="/home/u1120230285/share/Rex_Omni/Compression/train/DOTA2/annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_length=max_length,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="rexomni_grounding_data",
)

grounding_data_sodaa = dict(
    type=GroundingTSVDataset,
    img_tsv_file="/home/u1120230285/share/Rex_Omni/Compression/train/SODA-A/images.tsv",
    ann_tsv_file="/home/u1120230285/share/Rex_Omni/Compression/train/SODA-A/annotations.tsv",
    ann_lineidx_file="/home/u1120230285/share/Rex_Omni/Compression/train/SODA-A/annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_length=max_length,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="rexomni_grounding_data",
)


train_dataset = dict(
    type=ConcatDataset,
    datasets=[grounding_data_dota2, grounding_data_sodaa],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
