#!/bin/bash

# COCO Evaluation Pipeline - Multi-GPU Version
# This script runs the complete COCO evaluation pipeline using Multiple GPUs

set -e  # Exit on any error

# --- 配置参数 ---
MODEL_PATH="/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/dense/sft_base_oversample300/"
TEST_JSONL="/home/u1120250365/share/Rex_Omni/Compression/test/DOTA2_SODA-A/annotations_rex.json"
IMAGE_ROOT="/home/u1120250365/share/Rex_Omni/Compression/test/DOTA2_SODA-A/images"
COCO_JSON="/home/u1120250365/share/Rex_Omni/Compression/test/DOTA2_SODA-A/annotations_coco.json"
OUTPUT_DIR="/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/dense/sft_base_oversample300/Results/DOTA2_SODA-A"

# 多卡配置
NUM_GPUS=8

# 模型推理参数
MAX_TOKENS=20000
MAX_MODEL_LEN=20000
MIN_PIXELS=$((16 * 28 * 28))
MAX_PIXELS=$((2560 * 28 * 28))

# --- 初始化 ---
mkdir -p "$OUTPUT_DIR"
PRED_JSONL="$OUTPUT_DIR/answer.jsonl"
FASTEVAL_TSV="$OUTPUT_DIR/fast_eval.tsv"

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
    --question "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format."

echo "✅ Multi-GPU Inference completed!"
echo ""

# Step 2: 转换格式 (保持不变)
echo "🔄 Step 2: Converting predictions to FastEval TSV format..."
python evaluation/utils/convert_coco_lvis_to_standard_format.py \
    --our_pred_jsonl "$PRED_JSONL" \
    --coco_json "$COCO_JSON" \
    --out_tsv "$FASTEVAL_TSV" \
    --positive_only

echo "✅ Format conversion completed!"
echo ""

# Step 3: 指标评估 (保持不变)
echo "📊 Step 3: Running FastEval evaluation..."
python evaluation/metrics/coco_lvis_metric_dota.py \
    --gt "$COCO_JSON" \
    --pred_tsv "$FASTEVAL_TSV" \
    --eval_type coco

echo "✅ Evaluation completed successfully!"
echo ""

# Step 4: box/s    valid box/s
# echo "⏱️ Step 4: Analyzing per-image box speed..."
# "$INFERENCE_PYTHON" "$EVAL_DIR/metrics/analyze_box_speed.py" \
#     --prediction_jsonl "$PRED_JSONL" \
#     2>&1 | tee -a "$INFERENCE_LOG"

# echo "✅ Speed analysis completed!"
# echo ""

echo "🎉 All Done!"