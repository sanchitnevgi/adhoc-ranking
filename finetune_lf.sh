#!/bin/bash
#SBATCH --partition=2080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

export DATA_DIR=./data
export MAX_LENGTH=256
export LEARNING_RATE=2e-5
export BATCH_SIZE=16
export ACCUMULATE_GRAD=2
export MAX_EPOCHS=5
export WARMUP_STEPS=10
export SEED=42
export OUTPUT_DIR=./output

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python3 finetune_lf.py --data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--max_epochs $MAX_EPOCHS \
--warmup_steps $WARMUP_STEPS \
--train_batch_size $BATCH_SIZE \
--eval_batch_size $BATCH_SIZE \
--accumulate_grad_batches $ACCUMULATE_GRAD \
--seed $SEED \
--gpus 0