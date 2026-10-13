# WinARDet Training Guide

This guide covers environment installation, dataset preparation, configuration,
and supervised fine-tuning for **WinARDet**. 

## 1. Environment Setup

```bash
# Install the training package and its remaining dependencies.
cd finetuning
pip install -v -e .
```

## 3. Training Scripts

### 3.1 Script overview

| Script | Entry point | Configuration | Purpose |
|---|---|---|---|
| `scripts/sft_winar.sh` | `train_seg.py` | `configs/sft_winar.py` | Main WinAR horizontal-box training |
| `scripts/sft_winar_polygon.sh` | `train_seg.py` | `configs/sft_winar_polygon.py` | WinAR oriented-box training |
| `scripts/sft_base.sh` | `train.py` | `configs/sft_base.py` | RexOmni(FHAR) baseline |
| `scripts/sft_base_oversample.sh` | `train.py` | `configs/sft_base_oversample.py` | Exposure-matched RexOmni(FHAR*) baseline |
| `scripts/sft_base_polygon.sh` | `train.py` | `configs/sft_base_polygon.py` | Oriented-box RexOmni(FHAR*) baseline |

### 3.2 Train WinARDet

Before training, download the Rex-Omni base model from either
[ModelScope](https://modelscope.cn/models/IDEA-Research/Rex-Omni) or
[Hugging Face](https://huggingface.co/IDEA-Research/Rex-Omni). 

```bash
# Choice1: Download from ModelScope
pip install -U modelscope-hub
ms-hub download IDEA-Research/Rex-Omni \
  --repo-type model \
  --local-dir /path/to/Rex-Omni

# Choice2: Download from Hugging Face
pip install -U huggingface_hub
hf download IDEA-Research/Rex-Omni \
  --local-dir /path/to/Rex-Omni
```

Next, open the selected training config and set `model_name_or_path` to the
downloaded model directory. Update the three paths for every training dataset as
well. For example, in `configs/sft_winar.py`:

```python
model_name_or_path = "/path/to/Rex-Omni"

grounding_data_dota2 = dict(
    img_tsv_file="/path/to/WinAR_Data/train/DOTA2/images.tsv",
    ann_tsv_file="/path/to/WinAR_Data/train/DOTA2/annotations.tsv",
    ann_lineidx_file="/path/to/WinAR_Data/train/DOTA2/annotations.tsv.lineidx",
)
```

If multiple datasets are listed in the config, update all corresponding
`img_tsv_file`, `ann_tsv_file`, and `ann_lineidx_file` values. For oriented-box
training, make the same changes in `configs/sft_winar_polygon.py` using the
`DOTA2_Poly` or `SODA-A_Poly` training directory.

Finally, check the launch script and set `--config`, `--output_dir`, the visible
GPUs, and `--model_max_length` for the selected task. Then launch training from
the `finetuning/` directory:

```bash
cd WinARDet/finetuning

# Horizontal bounding boxes
bash scripts/sft_winar.sh

# Oriented bounding boxes
bash scripts/sft_winar_polygon.sh
```