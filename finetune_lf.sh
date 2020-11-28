#!/bin/bash
#SBATCH --partition=2080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

export DATA_DIR=./data
export MAX_SEQ_LENGTH=256
export LEARNING_RATE=3e-5
export BATCH_SIZE=4
export ACCUMULATE_GRAD=4
export MAX_STEPS=2000
export WARMUP_STEPS=100
export SEED=42
export OUTPUT_DIR=./output

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python3 finetune_lf.py --data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_SEQ_LENGTH \
--learning_rate $LEARNING_RATE \
--max_steps $MAX_STEPS \
--warmup_steps $WARMUP_STEPS \
--train_batch_size $BATCH_SIZE \
--eval_batch_size $BATCH_SIZE \
--accumulate_grad_batches $ACCUMULATE_GRAD \
--seed $SEED \
--gpus 1