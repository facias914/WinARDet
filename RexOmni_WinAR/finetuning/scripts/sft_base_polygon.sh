export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NNODES=1
GPUS_PER_NODE=8

export MASTER_ADDR=localhost
export MASTER_PORT=29500

run_name="rexomni_sft_polygon"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} train.py \
    --config configs/sft_base_polygon.py \
    --deepspeed scripts/zero2.json \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir work_dirs/sft_base_polygon \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 4e-5 \
    --mm_projector_lr 4e-5 \
    --vision_tower_lr 4e-6 \
    --optim adamw_torch \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 15000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb \