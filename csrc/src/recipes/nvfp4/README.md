# NVFP4 Recipe

FP4 E2M1 training with two-level block scaling.

## Overview

This recipe implements NVIDIA TransformerEngine's NVFP4BlockScaling strategy for extreme memory efficiency on Blackwell GPUs. FP4 uses only 4 bits per value, enabling 2x memory savings over FP8.

Key techniques for narrow-format stability:
1. **Two-level block scaling**: FP8 E4M3 scale per 16 values + FP32 global amax
2. **2D block quantization**: 16x16 blocks for weights (better accuracy)
3. **Stochastic rounding**: For gradients (avoids quantization bias)
4. **Random Hadamard Transform (RHT)**: Spreads outliers before quantization

## Format

| Tensor | Data Type | Scale Format | Block Size |
|--------|-----------|--------------|------------|
| Activations | FP4 E2M1 | FP8 E4M3 + FP32 | 16 |
| Weights | FP4 E2M1 | FP8 E4M3 + FP32 | 16x16 (2D) |
| Gradients | FP4 E2M1 | FP8 E4M3 + FP32 | 16 |

FP4 E2M1 representable values: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fp4-backend` | cudnn | Matmul backend: `cudnn` (default) or `cutlass` |
| `--no-fp4-hadamard` | false | Disable Random Hadamard Transform |
| `--no-fp4-stochastic-rounding` | false | Disable stochastic rounding for gradients |
| `--skip-quant-first-layers` | 0 | Skip quantization for first N layers (embedding) |
| `--skip-quant-last-layers` | 0 | Skip quantization for last N layers (lm_head) |

### Backend Selection

- **cudnn** (default): Uses cuDNN with F8_128x4 scale swizzling layout
- **cutlass**: Uses CUTLASS with Sm1xxBlkScaledConfig interleaved scale layout

Both backends implement the same quantization strategy; choose based on performance benchmarks for your workload.

### Stability Tips

- Use `--skip-quant-last-layers=2` if training is unstable (keeps lm_head layers in BF16)
- Use `--skip-quant-first-layers=1` to keep embedding layer in BF16
- Use `--master-dtype=fp32` for optimizer state precision
- RHT and stochastic rounding are recommended (enabled by default)

## Example

```bash
# Basic NVFP4 training (cuDNN backend)
./build/train --model Qwen/Qwen3-4B \
    --train-file="data/train.bin" --eval-file="data/eval.bin" \
    --recipe=nvfp4 --master-dtype=fp32 \
    --batch-size=2 --grad-accumulation=4 --seq-length=2048 \
    --lr=1e-5 --steps=1000 --recompute-block

# NVFP4 with CUTLASS backend
./build/train --model Qwen/Qwen3-4B \
    --train-file="data/train.bin" --eval-file="data/eval.bin" \
    --recipe=nvfp4 --fp4-backend=cutlass --master-dtype=fp32 \
    --batch-size=2 --grad-accumulation=4 --seq-length=2048 \
    --lr=1e-5 --steps=1000 --recompute-block

# With higher precision last layers for stability
./build/train --model Qwen/Qwen3-4B \
    --train-file="data/train.bin" --eval-file="data/eval.bin" \
    --recipe=nvfp4 --master-dtype=fp32 --skip-quant-last-layers=2 \
    --batch-size=2 --grad-accumulation=4 --seq-length=2048 \
    --lr=1e-5 --steps=1000 --recompute-block

# LoRA fine-tuning with NVFP4
./build/train --model Qwen/Qwen3-4B \
    --train-file="data/train.bin" --eval-file="data/eval.bin" \
    --recipe=nvfp4 --master-dtype=fp32 \
    --lora --lora-rank=16 --lora-alpha=32 \
    --lora-target-modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --batch-size=2 --grad-accumulation=4 --seq-length=2048 \
    --lr=2e-4 --steps=1000 --recompute-block
```

## Hardware Requirements

- NVIDIA Blackwell GPU (SM100+) with native FP4 tensor cores
- cuDNN 9.0+ (for cudnn backend) or CUTLASS 4.x (for cutlass backend)
