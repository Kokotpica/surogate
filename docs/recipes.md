# Precision Recipes

Surogate provides 3 out-of-the-box precision recipes for the 3 most common numerical formats used in training:

- **BF16 (bfloat16)**: default recipe providing maximum numerical accuracy and most memory usage.
- **FP8-Hybrid (float8)**: provides a balance between numerical accuracy and memory usage by using 8-bit floating point precision.
- **FP4 (nvfp4)**: provides maximum acceleration on Blackwell GPUs by using 4-bit floating point precision, at the cost of some numerical accuracy.

## BF16

This recipe uses `bfloat16` for all GEMM operations without any quantization. It is suitable when memory and compute resources are not constrained, or when training smaller models where the savings from lower precision formats are not significant.

Use this recipe when:

- Only bfloat16 is supported on your hardware
- Memory and compute are not constrained
- You need a baseline for comparing quantized training
- Training smaller models where FP8/FP4 savings aren't significant

### Forward/Backward Format

| Pass     | Data Type | Scaling |
| -------- | --------- | ------- |
| Forward  | bfloat16  | None    |
| Backward | bfloat16  | None    |

### Example

```yaml
recipe: bf16
```

## FP8-Hybrid

This recipe uses FP8 with E4M3 format for the forward pass and E5M2 format for the backward pass, employing delayed scaling for improved stability.

- **E4M3** (max=448): Used for forward pass activations and weights - higher precision
- **E5M2** (max=57344): Used for backward pass gradients - larger dynamic range

Delayed scaling uses scale factors computed from the previous iteration's abs-max values, providing more stable training than just-in-time scaling. The recipe maintains an amax history window and uses the maximum value from the history to compute scale factors.

The numerical accuracy is generally comparable to bfloat16, while providing significant memory savings and speedup on supported hardware with FP8 tensor cores (SM89+: Ada Lovelace, Hopper, Blackwell).

Use this recipe when:

- Your GPU supports FP8 tensor cores (SM89+: Ada Lovelace, Hopper, Blackwell)
- You accept a minor drop in numerical accuracy for significant memory and speed benefits
- Training large models

### Forward/Backward Format

| Pass     | Data Type | Max Value | Scaling            |
| -------- | --------- | --------- | ------------------ |
| Forward  | FP8 E4M3  | 448       | Per-tensor delayed |
| Backward | FP8 E5M2  | 57344     | Per-tensor delayed |

### Parameters

| Parameter                 | Default | Description                                                    |
| ------------------------- | ------- | -------------------------------------------------------------- |
| `fp8_amax_history`        | 1024    | Length of amax history window for delayed scaling              |
| `skip_quant_first_layers` | 0       | Number of first layers to skip quantization (keep in bfloat16) |
| `skip_quant_last_layers`  | 0       | Number of last layers to skip quantization (keep in bfloat16)  |

### Stability Tips

- Use `skip_quant_first_layers: 1` to keep embedding layer in BF16
- Use `skip_quant_last_layers: 2` if training is unstable (keeps lm_head layers in BF16)
- 
### Example

```yaml
recipe: fp8-hybrid
skip_quant_first_layers: 1
skip_quant_last_layers: 2
```

## FP4

This recipe uses NVIDIA's NVFP4 format for both forward and backward passes, employing two-level block scaling for improved stability. It uses FP8 E4M3 scales per 16 values and a global FP32 amax, along with 2D block quantization for weights, stochastic rounding for gradients, and optional Random Hadamard Transforms (RHT) to spread outliers before quantization.

It also includes the [Four Over Six (4/6)](https://arxiv.org/abs/2512.02010) technique (enabled by default), a modification to the NVFP4 quantization algorithm that evaluates two potential scale factors (max=4.0 vs max=6.0) for each block of values and selects the one with lower quantization error.

FP4 E2M1 representable values: ±{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

Use this recipe when:

- You are training on Blackwell GPUs with FP4 support (SM100+)

### Forward/Backward Format

| Tensor      | Data Type | Scale Format    | Block Size |
| ----------- | --------- | --------------- | ---------- |
| Activations | FP4 E2M1  | FP8 E4M3 + FP32 | 16         |
| Weights     | FP4 E2M1  | FP8 E4M3 + FP32 | 16x16 (2D) |
| Gradients   | FP4 E2M1  | FP8 E4M3 + FP32 | 16         |

### Parameters

| Parameter                    | Default | Description                                                        |
| ---------------------------- | ------- | ------------------------------------------------------------------ |
| `fp4_backend`                | cutlass | Matmul backend: `cutlass` (default) or `cudnn`                     |
| `no_fp4_hadamard`            | false   | Disable Random Hadamard Transform                                  |
| `no_fp4_stochastic_rounding` | false   | Disable stochastic rounding for gradients                          |
| `skip_quant_first_layers`    | 0       | Skip quantization for first N layers (keep in BF16 for stability)  |
| `skip_quant_last_layers`     | 0       | Skip quantization for last N layers (keep in BF16 for stability)   |

### Backend Selection

- **cutlass** (default): Uses CUTLASS with Sm1xxBlkScaledConfig interleaved scale layout. Supports alpha fusion in epilogue for direct BF16 output.
- **cudnn**: Uses cuDNN with F8_128x4 scale swizzling layout.

Both backends implement the same quantization strategy; choose based on performance benchmarks for your workload.

### Weight Caching (SM100+)

On Blackwell GPUs (SM100+), FP4 weight caching is **enabled by default** to eliminate per-forward weight quantization overhead. This is critical for datacenter GPUs (B200/B300) where FP4 GEMMs are extremely fast and weight quantization would otherwise become the bottleneck.

Weight caching uses additional GPU memory to store the pre-quantized weights but provides significant throughput improvements by eliminating the quantization overhead from the critical path.

**How it works:**

1. Weights are pre-quantized to FP4 format with CUTLASS-optimized layout during model initialization
2. The cached FP4 weights (packed data + FP8 block scales + global amax) are reused across forward passes
3. A separate transposed weight cache is maintained for the backward pass (dgrad), avoiding repeated transposition and re-quantization


**Requirements:**

- Blackwell GPU (SM100+)
- ZeRO-3/FSDP weight streaming disabled (weights must be static on device)
- Best suited for LoRA/QLoRA fine-tuning where base weights are frozen

### Stability Tips

- Use `skip_quant_first_layers: 1` to keep embedding layer in BF16
- Use `skip_quant_last_layers: 4` if training is unstable (keeps lm_head layers in BF16)
- Random Hadamard Transforms and stochastic rounding are recommended (enabled by default) for numerical stability

### Example

```yaml
recipe: nvfp4
skip_quant_first_layers: 1
skip_quant_last_layers: 4
```