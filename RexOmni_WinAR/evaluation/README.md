# WinARDet Evaluation Guide

This guide describes how to set up the inference environment, download the
released checkpoints, and evaluate WinARDet on dense-scene grounding and
oriented-box benchmarks. 

## 1. Environment Setup

The released evaluation scripts use three separately named FastEvaluate
backends. Build and install all three with the same Python interpreter used for
inference:

```bash
for backend in fastevaluate_all fastevaluate_dota fastevaluate_rsdense; do
  (
    cd "evaluation/${backend}"
    python setup.py build_ext --inplace
    python -m pip install -e .
  )
done
```

Verify that all extensions can be imported:

```bash
python - <<'PY'
import fastevaluate_all
import fastevaluate_dota
import fastevaluate_rsdense

print("All FastEvaluate backends were imported successfully.")
PY
```

The three versions implement the evaluation protocols used for different
benchmarks in the paper:

| Backend | Protocol | Dataset / script |
|---|---|---|
| `fastevaluate_all` | Retains every loaded prediction without an additional per-image `maxDets` truncation. Matching remains category-aware. | SKU110K; `eval_sku110k_multi*.sh` |
| `fastevaluate_dota` | Uses category-wise COCO-style matching with up to 1,000 predictions per category and image, following the DOTA2 evaluation setting. | DOTA2 & SODA-A; `eval_dota_multi*.sh` |
| `fastevaluate_rsdense` | Retains all predictions, pools TP/FP/FN across categories within each image, computes image-level precision/recall/F1, and then averages the metrics over images. | RSDense; `eval_rsdense_multi*.sh` |

The oriented-box benchmarks do not use FastEvaluate. The polygon scripts call
`evaluation/metrics/fast_rotated_iou_metric.py` directly.

## 2. Model Checkpoints

The released checkpoints are available from either
[ModelScope](https://modelscope.cn/models/facias914/WinAR_Model) or
[Hugging Face](https://huggingface.co/facias/WinARDet).

### 2.1 Option 1: ModelScope

Download the complete checkpoint repository:

```bash
pip install -U modelscope

modelscope download --model facias914/WinAR_Model \
  --local_dir ./checkpoints/WinAR_Model
```

To download only one checkpoint, use its repository path as an include pattern.
For example:

```bash
modelscope download --model facias914/WinAR_Model \
  --include 'dense/sft_seg_300_6-8-6_l6_e4_lr10_rescale/**' \
  --local_dir ./checkpoints/WinAR_Model
```

### 2.2 Option 2: Hugging Face

Download the complete checkpoint repository:

```bash
pip install -U huggingface_hub

hf download facias/WinARDet \
  --local-dir ./checkpoints/WinARDet
```

To download only one checkpoint, use its repository path as an include pattern.
For example:

```bash
hf download facias/WinARDet \
  --include 'compression/dense/sft_seg_300_6-8-6_l6_e4_lr10_rescale/**' \
  --local-dir ./checkpoints/WinARDet
```

### 2.3 Released models

`FHAR*` denotes the exposure-matched full-history baseline, while `WinAR`
denotes the compressed window auto-regressive model. Each link below opens the
directory for one independently loadable checkpoint. In the table, `L` is the
window length, `B` is the number of raw boundary tokens retained on each side,
and `N` is the number of latent tokens used to represent compressed history.

| Benchmark | Training setting | Difference | ModelScope | Hugging Face |
|---|---|---|---|---|
| DOTA2 & SODA-A | FHAR | Original full-history autoregression; retains every previous answer token and exposes each sample once per epoch. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=dense%2Fsft_base) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/dense/sft_base) |
| DOTA2 & SODA-A | FHAR* | Exposure-matched FHAR baseline; oversamples a length-`T` answer by approximately `ceil(T/300)` while retaining the complete raw history. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=dense%2Fsft_base_oversample300) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/dense/sft_base_oversample300) |
| DOTA2 & SODA-A | FHAR*-GRPO | Initializes from FHAR* and adds one epoch of GRPO post-training on joint DOTA2/SODA-A samples containing more than 200 objects. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=dense%2Fsft_base_oversample300_grpo%2Fmodel) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/dense/sft_base_oversample300_grpo/model) |
| DOTA2 & SODA-A | WinAR | Uses compressed window autoregression with `L=300`, `B=6` boundary tokens per side, and `N=8` latent history tokens. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=dense%2Fsft_seg_300_6-8-6_l6_e4_lr10_rescale) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/dense/sft_seg_300_6-8-6_l6_e4_lr10_rescale) |
| SKU110K | FHAR | Standard full-history baseline for dense retail scenes; retains all raw historical answer tokens. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=SKU110K%2Fsft_base) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/SKU110K/sft_base) |
| SKU110K | FHAR* | Exposure-matched full-history baseline with `ceil(T/300)` length-aware oversampling; no history compression. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=SKU110K%2Fsft_base_oversample300) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/SKU110K/sft_base_oversample300) |
| SKU110K | WinAR | Applies `L=300` compressed window autoregression to dense retail grounding, using `B=6` and `N=8`. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=SKU110K%2Fsft_seg_300_6-8-6_l6_e4_lr10_rescale) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/SKU110K/sft_seg_300_6-8-6_l6_e4_lr10_rescale) |
| DOTA2-Poly | FHAR* | Exposure-matched full-history baseline for eight-coordinate quadrilateral outputs, using `ceil(T/400)` oversampling. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=DOTA2_polygon%2Fsft_base_oversample400) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/DOTA2_polygon/sft_base_oversample400) |
| DOTA2-Poly | WinAR | Polygon WinAR with the larger `L=400`, `B=8`, and `N=12` configuration for quadrilateral generation. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=DOTA2_polygon%2Fsft_seg_400_8-12-8_l6_e4_lr10_rescale) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/DOTA2_polygon/sft_seg_400_8-12-8_l6_e4_lr10_rescale) |
| SODA-A-Poly | FHAR* | Exposure-matched full-history baseline for eight-coordinate quadrilateral outputs on the denser SODA-A scenes, using `ceil(T/400)` oversampling. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=SODA-A_polygon%2Fsft_base_oversample) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/SODA-A_polygon/sft_base_oversample) |
| SODA-A-Poly | WinAR | Polygon WinAR with `L=400`, `B=8`, and `N=12`, compressing completed windows during long quadrilateral generation. | [Download](https://modelscope.cn/models/facias914/WinAR_Model/files?path=SODA-A_polygon%2Fsft_seg_400_8-12-8_l6_e4_lr10_rescale) | [Download](https://huggingface.co/facias/WinARDet/tree/main/compression/SODA-A_polygon/sft_seg_400_8-12-8_l6_e4_lr10_rescale) |

## 3. Run Inference and Evaluation

### 3.1 Scripts

The script pairs are:

| Benchmark | FHAR / FHAR* | WinAR |
|---|---|---|
| DOTA2 & SODA-A | `evaluation/scripts/eval_dota_multi.sh` | `evaluation/scripts/eval_dota_multi_compressed.sh` |
| RSDense | `evaluation/scripts/eval_rsdense_multi.sh` | `evaluation/scripts/eval_rsdense_multi_compressed.sh` |
| SKU110K | `evaluation/scripts/eval_sku110k_multi.sh` | `evaluation/scripts/eval_sku110k_multi_compressed.sh` |
| DOTA2-Poly or SODA-A-Poly | `evaluation/scripts/eval_poly_multi.sh` | `evaluation/scripts/eval_poly_multi_compressed.sh` |


### 3.2 Launch evaluation

Before launching evaluation, update `MODEL_PATH`, `TEST_JSONL`, `IMAGE_ROOT`,
`COCO_JSON`, and `OUTPUT_DIR` in each shell script according to the selected
model and dataset. The polygon scripts do not use `COCO_JSON`.

Run commands from the WinARDet repository root so that all relative paths in
the scripts resolve correctly:

```bash
cd /path/to/WinARDet

# DOTA2 & SODA-A
bash evaluation/scripts/eval_dota_multi.sh
bash evaluation/scripts/eval_dota_multi_compressed.sh

# RSDense
bash evaluation/scripts/eval_rsdense_multi.sh
bash evaluation/scripts/eval_rsdense_multi_compressed.sh

# SKU110K
bash evaluation/scripts/eval_sku110k_multi.sh
bash evaluation/scripts/eval_sku110k_multi_compressed.sh

# DOTA2-Poly or SODA-A-Poly
bash evaluation/scripts/eval_poly_multi.sh
bash evaluation/scripts/eval_poly_multi_compressed.sh
```

Each horizontal-box script runs inference, converts the prediction JSONL to
FastEvaluate TSV format, and reports the evaluation metrics. Each polygon
script runs inference and writes the rotated-IoU evaluation result alongside
the prediction JSONL in `OUTPUT_DIR`.
