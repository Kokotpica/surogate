# FP8 Hybrid Recipe

FP8 training with E4M3 format for forward pass and E5M2 for backward pass, using delayed scaling.

## Overview

This recipe implements NVIDIA TransformerEngine's "DelayedScaling" strategy with hybrid FP8 formats:
- **E4M3** (max=448): Used for forward pass activations and weights - higher precision
- **E5M2** (max=57344): Used for backward pass gradients - larger dynamic range

Delayed scaling uses scale factors computed from the previous iteration's abs-max values, providing more stable training than just-in-time scaling.

## Format

| Pass | Data Type | Max Value | Scaling |
|------|-----------|-----------|---------|
| Forward | FP8 E4M3 | 448 | Per-tensor delayed |
| Backward | FP8 E5M2 | 57344 | Per-tensor delayed |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fp8-amax-history` | 1024 | Length of amax history window for delayed scaling |

## Example

```bash
# Basic FP8 hybrid training
./build/train --model Qwen/Qwen3-4B \
    --train-file="data/train.bin" --eval-file="data/eval.bin" \
    --recipe=fp8-hybrid \
    --batch-size=2 --grad-accumulation=4 --seq-length=2048 \
    --lr=1e-5 --steps=1000 --recompute-block

# LoRA fine-tuning with FP8
./build/train --model Qwen/Qwen3-4B \
    --train-file="data/train.bin" --eval-file="data/eval.bin" \
    --recipe=fp8-hybrid \
    --lora --lora-rank=16 --lora-alpha=32 \
    --lora-target-modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --batch-size=2 --grad-accumulation=4 --seq-length=2048 \
    --lr=2e-4 --steps=1000 --recompute-block
```

## Hardware Requirements

- NVIDIA GPU with FP8 tensor cores (SM89+: Ada Lovelace, Hopper, Blackwell)
