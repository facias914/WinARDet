from dataset import (
    ConcatDataset,
    DataCollatorForSupervisedDataset,
    PolygonTSVDataset,
)
from dataset.task_fns import PolygonTaskFn
from dataset.task_fns.task_prompts.polygon_task import POLYGON_SINGLE_REGION_STAGE_XYXYXYXY

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28
max_length=40000

model_name_or_path = "/home/u1120230285/wcx/Rex-Omni/Rex-Omni"

polygon_data_dota2 = dict(
    type=PolygonTSVDataset,
    img_tsv_file="/home/share/Rex_Omni/Compression/train/DOTA2_Poly/images.tsv",
    ann_tsv_file="/home/share/Rex_Omni/Compression/train/DOTA2_Poly/annotations.tsv",
    ann_lineidx_file="/home/share/Rex_Omni/Compression/train/DOTA2_Poly/annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_length=max_length,
    task_fn=dict(
        type=PolygonTaskFn,
        task_prompts=POLYGON_SINGLE_REGION_STAGE_XYXYXYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="rexomni_polygon_dota2",
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[polygon_data_dota2],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
