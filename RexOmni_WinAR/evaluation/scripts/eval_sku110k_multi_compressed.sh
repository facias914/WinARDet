#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- 配置参数 ---
MODEL_PATH="/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/SKU110K/sft_seg_300_6-8-6_l6_e4_lr10_rescale"
TEST_JSONL="/home/u1120250365/share/Rex_Omni/Compression/test/SKU110K/annotations_rex.json"
IMAGE_ROOT="/home/u1120250365/share/Rex_Omni/Compression/test/SKU110K/images"
COCO_JSON="/home/u1120250365/share/Rex_Omni/Compression/test/SKU110K/annotations_coco.json"
OUTPUT_DIR="/home/u1120250365/Rex-Omni/finetuning/work_dirs/compression/SKU110K/sft_seg_300_6-8-6_l6_e4_lr10_rescale/Results/SKU110K"

# 多卡配置
GPU_IDS="0,1,2,3,4,5,6,7"
PROCS_PER_GPU=1

# 模型推理参数
MAX_TOKENS=20000
MIN_PIXELS=$((16 * 28 * 28))
MAX_PIXELS=$((2560 * 28 * 28))

# 压缩参数：需与训练配置/ckpt 对齐
MAX_TOKENS_PER_SEGMENT=300
NUM_MEMORY_TOKENS=8
COMPRESSED_PASSTHROUGH_TOKENS_PER_SIDE=6
COMPRESSOR_NUM_LAYERS=6
COMPRESSOR_NUM_HEADS=8

# 采样参数
TEMPERATURE=0.0
TOP_P=0.8
TOP_K=1
REPETITION_PENALTY=1.05

# --- 初始化 ---
mkdir -p "$OUTPUT_DIR"
PRED_JSONL="$OUTPUT_DIR/answer.jsonl"
FASTEVAL_TSV="$OUTPUT_DIR/fast_eval.tsv"

TMP_PREFIX="${PRED_JSONL%.jsonl}_worker"

echo "🚀 Starting MULTI-GPU COMPRESSED DOTA2 Evaluation Pipeline"
echo "==========================================================="
echo "Model:       $MODEL_PATH"
echo "GPUs:        $GPU_IDS"
echo "Procs/GPU:   $PROCS_PER_GPU"
echo "Output:      $PRED_JSONL"
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
    --question "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format."

echo "✅ Compressed multi-GPU inference completed!"
echo ""

# Step 2: 转换格式
echo "🔄 Step 2: Converting predictions to FastEval TSV format..."
"$PYTHON_BIN" "$EVAL_DIR/utils/convert_coco_lvis_to_standard_format.py" \
    --our_pred_jsonl "$PRED_JSONL" \
    --coco_json "$COCO_JSON" \
    --out_tsv "$FASTEVAL_TSV" \
    --positive_only

echo "✅ Format conversion completed!"

# Step 3: 指标评估 (保持不变)
echo "📊 Step 3: Running FastEval evaluation..."
python evaluation/metrics/coco_lvis_metric_all.py \
    --gt "$COCO_JSON" \
    --pred_tsv "$FASTEVAL_TSV" \
    --eval_type coco

echo "✅ Evaluation completed successfully!"
echo ""


