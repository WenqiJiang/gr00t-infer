set -x -e

# Redirect temp/shared memory files to NFS to avoid home directory quota limits
export TMPDIR=/home/scratch.wenqij_research/tmp
export TORCH_SHM_DIR=/home/scratch.wenqij_research/tmp/torch_shm
mkdir -p $TMPDIR $TORCH_SHM_DIR

export NUM_GPUS=${NUM_GPUS:-8}
EFFECTIVE_BATCH_SIZE=640
PER_DEVICE_BATCH_SIZE=80
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / NUM_GPUS))

LAUNCHER="uv run python"
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER="uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=29500"
fi

# Add --use_wandb below if you have WANDB_API_KEY set
$LAUNCHER gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/ \
    --embodiment_tag LIBERO_PANDA \
    --num_gpus $NUM_GPUS \
    --output_dir checkpoints/libero_object \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 2
