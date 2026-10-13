#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- 配置参数 ---
MODEL_PATH="/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/DOTA2_polygon/sft_seg_400_8-12-8_l6_e4_lr10_rescale"
TEST_JSONL="/home/u1120250365/share/Rex_Omni/Compression/test/DOTA2_Poly/annotations_rex.json"
IMAGE_ROOT="/home/u1120250365/share/dota_v2/DOTA_v2/split_ss_dota/val/images"
OUTPUT_DIR="/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/DOTA2_polygon/sft_seg_400_8-12-8_l6_e4_lr10_rescale/Results/DOTA2_Polygon"

# 多卡配置
GPU_IDS="0,1,2,3,4,5,6,7"
PROCS_PER_GPU=1

# 模型推理参数
MAX_TOKENS=40000
MIN_PIXELS=$((16 * 28 * 28))
MAX_PIXELS=$((2560 * 28 * 28))

# 压缩参数：需与训练配置/ckpt 对齐
MAX_TOKENS_PER_SEGMENT=400
NUM_MEMORY_TOKENS=12
COMPRESSED_PASSTHROUGH_TOKENS_PER_SIDE=8
COMPRESSOR_NUM_LAYERS=6
COMPRESSOR_NUM_HEADS=8

# 采样参数
TEMPERATURE=0.0
TOP_P=0.8
TOP_K=1
REPETITION_PENALTY=1.05

# --- 初始化 ---
if [ ! -d "$IMAGE_ROOT" ]; then
    echo "❌ IMAGE_ROOT not found: $IMAGE_ROOT"
    echo "Set IMAGE_ROOT to the directory containing DOTA2 poly images."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
PRED_JSONL="$OUTPUT_DIR/answer.jsonl"
METRIC_JSON="$OUTPUT_DIR/poly_eval_results.json"

TMP_PREFIX="${PRED_JSONL%.jsonl}_worker"

# Poly 推理/评测参数
OUTPUT_BOX_FORMAT="poly"
TASK_NAME_OVERRIDE="task4_detect_all_in_polygon"
POLY_QUESTION="Detect [OBJ]. Output each rotated bounding box as polygon coordinates in [x0, y0, x1, y1, x2, y2, x3, y3] format."

echo "🚀 Starting MULTI-GPU COMPRESSED DOTA2 Evaluation Pipeline"
echo "==========================================================="
echo "Model:       $MODEL_PATH"
echo "GPUs:        $GPU_IDS"
echo "Procs/GPU:   $PROCS_PER_GPU"
echo "Output:      $PRED_JSONL"
echo "Box format:  $OUTPUT_BOX_FORMAT"
echo "Segment:     $MAX_TOKENS_PER_SEGMENT tokens -> $NUM_MEMORY_TOKENS memory tokens"
echo "Python:      $PYTHON_BIN"
echo ""

# Step 1: 多卡并行压缩推理
echo "📝 Step 1: Running Parallel Segment-Compressed Inference..."
"$PYTHON_BIN" "$EVAL_DIR/inference_text_prompt_multi_compressed.py" \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$TEST_JSONL" \
    --image_root_dir "$IMAGE_ROOT" \
    --save_path "$PRED_JSONL" \
    --gpu_ids "$GPU_IDS" \
    --procs_per_gpu "$PROCS_PER_GPU" \
    --max_new_tokens "$MAX_TOKENS" \
    --min_pixels "$MIN_PIXELS" \
    --max_pixels "$MAX_PIXELS" \
    --max_tokens_per_segment "$MAX_TOKENS_PER_SEGMENT" \
    --num_memory_tokens "$NUM_MEMORY_TOKENS" \
    --compressor_passthrough_tokens_per_side "$COMPRESSED_PASSTHROUGH_TOKENS_PER_SIDE" \
    --compressor_num_layers "$COMPRESSOR_NUM_LAYERS" \
    --compressor_num_heads "$COMPRESSOR_NUM_HEADS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --output_box_format "$OUTPUT_BOX_FORMAT" \
    --task_name_override "$TASK_NAME_OVERRIDE" \
    --question "$POLY_QUESTION"

echo "✅ Compressed multi-GPU inference completed!"
echo ""

# Step 2: polygon 指标评估
echo "📊 Step 2: Running polygon IoU evaluation..."
"$PYTHON_BIN" "$EVAL_DIR/metrics/fast_rotated_iou_metric.py" \
    --data_path "$PRED_JSONL" \
    --output_path "$METRIC_JSON"

echo "✅ Polygon evaluation completed successfully!"

echo "🎉 All Done!"
