#!/bin/bash

# COCO Evaluation Pipeline - Multi-GPU Version
# This script runs the complete COCO evaluation pipeline using Multiple GPUs

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- 配置参数 ---
MODEL_PATH="/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/DOTA2_polygon/sft_base_oversample400"
TEST_JSONL="/home/share/Rex_Omni/Compression/test/DOTA2_Poly/annotations_rex.json"
IMAGE_ROOT="/home/share/dota_v2/DOTA_v2/split_ss_dota/trainval/images"
OUTPUT_DIR="/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/DOTA2_polygon/sft_base_oversample400/Results/Dota2_Polygon"

# 多卡配置
NUM_GPUS=8

# 模型推理参数
MAX_TOKENS=40000
MAX_MODEL_LEN=40000
MIN_PIXELS=$((16 * 28 * 28))
MAX_PIXELS=$((2560 * 28 * 28))

# --- 初始化 ---
mkdir -p "$OUTPUT_DIR"
OUTPUT_BOX_FORMAT="poly"
PRED_JSONL="$OUTPUT_DIR/answer.jsonl"
METRIC_JSON="$OUTPUT_DIR/poly_eval_results.json"

echo "🚀 Starting MULTI-GPU COCO Evaluation Pipeline"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "GPUs:  $NUM_GPUS"
echo "Output: $PRED_JSONL"
echo ""

# Step 1: 多卡并行推理
echo "📝 Step 1: Running Parallel Inference (Data Parallel)..."
# 注意：这里调用的是你刚才修改后的那个 Python 文件
python evaluation/inference_text_prompt_multi.py \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$TEST_JSONL" \
    --image_root_dir "$IMAGE_ROOT" \
    --save_path "$PRED_JSONL" \
    --num_gpus "$NUM_GPUS" \
    --max_new_tokens "$MAX_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --min_pixels "$MIN_PIXELS" \
    --max_pixels "$MAX_PIXELS" \
    --question "Detect [OBJ]. Output each rotated bounding box as polygon coordinates in [x0, y0, x1, y1, x2, y2, x3, y3] format." \
    --output_box_format "$OUTPUT_BOX_FORMAT" 


echo "✅ Compressed multi-GPU inference completed!"
echo ""

# Step 2: polygon 指标评估
echo "📊 Step 2: Running polygon IoU evaluation..."
"$PYTHON_BIN" "$EVAL_DIR/metrics/fast_rotated_iou_metric.py" \
    --data_path "$PRED_JSONL" \
    --output_path "$METRIC_JSON"

echo "✅ Polygon evaluation completed successfully!"

echo "🎉 All Done!"