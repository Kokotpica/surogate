#!/bin/bash
set -euo pipefail

MODEL="Qwen/Qwen3-0.6B"
TRAIN_FILES="data/gsm8kro/train.bin"
EVAL_FILE="data/gsm8kro/eval.bin"
RECIPE="bf16" # you can change this to other recipes like nvfp4, fp8-hybrid

./csrc/build/train --model $MODEL --train-file=$TRAIN_FILES --eval-file=$EVAL_FILE  \
    --gpus=1 --batch-size=2 --grad-accumulation=4 --seq-len=2048 \
    --lr=2e-4 --recipe=$RECIPE \
    --lora --lora-rank=32 --lora-alpha=64 --lora-target-modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --ckpt-interval=1000 --eval-num-steps=1000 --final-eval-num-steps=0 \
    --warmup=10 --eval-every-n-steps=0 --recompute-block \
    --save --out-dir=lora_bf16