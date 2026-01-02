# BF16 Recipe

Baseline recipe using BFloat16 for all matmul operations. No quantization is applied.

## Overview

This is the default recipe providing maximum numerical accuracy. Use this when:
- Memory and compute are not constrained
- You need a baseline for comparing quantized training
- Training smaller models where FP8/FP4 savings aren't significant

## Format

| Pass | Data Type | Scaling |
|------|-----------|---------|
| Forward | BF16 | None |
| Backward | BF16 | None |

## Parameters

No recipe-specific parameters.

## Example

```bash
./build/train --model Qwen/Qwen3-4B \
    --train-file="data/train.bin" --eval-file="data/eval.bin" \
    --recipe=bf16 \
    --batch-size=2 --grad-accumulation=4 --seq-length=2048 \
    --lr=1e-5 --steps=1000
```
