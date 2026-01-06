# Mixed-Precision Training

Surogate is a versatible framework that supports mixed-precision training using a combination of numerical formats to optimize memory usage and computational speed while maintaining model accuracy.

The framework provides the following parameters to configure the precision of different components during training:

| Parameter          | Options          | Description                                                                                                                                                         |
| ------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **matmul_dtype**   | fp32, bf16, e4m3 | Data type for matrix multiplications. Defaults to model_dtype. e5m2/fp16/e2m1 not supported for forward pass. FP8 requires SM89+ (Ada/Hopper)                       |
| **gradient_dtype** | fp32, bf16, e5m2 | Data type for activation gradients and backward matmuls. Defaults to matmul_dtype. fp16/e4m3/e2m1 not supported. fp8-hybrid recipe forces e5m2                      |
| **master_dtype**   | fp32, bf16       | Master weight dtype for optimizer updates. Defaults to model_dtype. Only fp32 and bf16 are supported                                                                |
| **model_dtype**    | fp32, bf16       | Data type for non-matmul weights (RMSNorm, embeddings) and activations. Defaults to bf16. Other dtype params fall back to this. Only fp32/bf16 supported by kernels |
| **lora_dtype**     | fp32, bf16       | LoRA adapter master weights dtype for optimizer/export. Defaults to fp32. Work weights converted to model_dtype for compute. Only fp32↔bf16 conversion supported    |

## Matmul Dtype and Gradient Dtype

**Note on recipe behavior:** The `matmul_dtype` and `gradient_dtype` parameters are only respected when using the default `bf16` recipe. When using `fp8-hybrid` or `nvfp4` recipes, these parameters are overridden:

| Recipe       | Forward matmul | Backward matmul |
| ------------ | -------------- | --------------- |
| `bf16`       | matmul_dtype   | gradient_dtype  |
| `fp8-hybrid` | e4m3 (forced)  | e5m2 (forced)   |
| `nvfp4`      | e2m1 (forced)  | e2m1 (forced)   |

## Supported Matmul Dispatches

The following dtype combinations are supported for matrix multiplications:

| A (input/weight) | B (input/weight) | C (output) | Use Case                                        |
| ---------------- | ---------------- | ---------- | ----------------------------------------------- |
| fp32             | fp32             | fp32       | Full precision training                         |
| bf16             | bf16             | fp32       | Mixed precision (BF16 compute, FP32 accumulate) |
| bf16             | bf16             | bf16       | Pure BF16 training                              |
| e4m3             | e4m3             | fp32       | FP8 forward pass                                |
| e4m3             | e4m3             | bf16       | FP8 forward pass (BF16 output)                  |
| e4m3             | e5m2             | bf16       | FP8 backward pass (weight × gradient)           |

## Master Weights Dtype

The `master_dtype` parameter controls the precision of **master weights** - the authoritative copy of model weights used for:

1. **Optimizer updates**: The optimizer (8-bit AdamW) reads and writes master weights. Higher precision (fp32) reduces accumulation errors over many training steps.
2. **Checkpointing**: Master weights are saved to checkpoints and used for model export.
3. **Weight synchronization**: In multi-GPU training, master weights are the source of truth that gets converted to work weights.

**Work weights** vs **Master weights**:

- **Work weights**: Used for forward/backward passes. Can be lower precision (bf16, e4m3) for faster compute.
- **Master weights**: Used for optimizer updates. Stored in higher precision (fp32 or bf16) for numerical stability.

When `master_dtype` differs from `model_dtype` or `matmul_dtype`, separate storage is allocated:

- Master weights are updated by the optimizer
- Work weights are converted from master weights before each forward pass

This separation enables mixed-precision training where compute happens in low precision but weight updates accumulate in high precision, preventing gradient underflow and maintaining training stability.

**Example**: With `model_dtype=bf16` and `master_dtype=fp32`:

- Forward/backward use BF16 weights (faster, less memory per weight)
- Optimizer updates FP32 master weights (more precise accumulation)
- FP32 masters are converted to BF16 work weights each step

## Model Dtype

The `model_dtype` parameter is the **fundamental dtype** that controls the precision of model parameters and serves as the default for other dtype parameters. It affects:

1. **Non-matmul weights**: Normalization layer weights (RMSNorm, LayerNorm), embedding weights, and other non-linear projection weights are stored and computed in `model_dtype`.
2. **Default fallback**: When `matmul_dtype`, `gradient_dtype`, or `master_dtype` are not explicitly set, they default to `model_dtype`.
3. **Activation storage**: Intermediate activations during forward/backward passes use `model_dtype`.
4. **Non-block weights**: Embeddings, final layer norm, and LM head weights use `model_dtype`.

**Relationship between dtype parameters:**

```
model_dtype (fundamental)
    ├── matmul_dtype (defaults to model_dtype)
    │       └── gradient_dtype (defaults to matmul_dtype)
    └── master_dtype (defaults to model_dtype)
```

**Weight allocation by dtype:**

| Weight Type                 | Dtype Used   | Examples                      |
| --------------------------- | ------------ | ----------------------------- |
| Linear projections (matmul) | matmul_dtype | QKV, output proj, MLP up/down |
| Normalization weights       | model_dtype  | RMSNorm, LayerNorm            |
| Embeddings                  | model_dtype  | Token embeddings, LM head     |
| Master weights              | master_dtype | Optimizer state storage       |

**Practical implications:**

- Setting `model_dtype=bf16` (default) provides a good balance of speed and precision
- `model_dtype` determines memory footprint for non-matmul components
- When using FP8 recipes, `model_dtype` typically stays at bf16 while `matmul_dtype` uses e4m3

## LoRA Dtype

The `lora_dtype` parameter controls the precision of **LoRA adapter master weights**. Like the base model's master weights, LoRA uses a two-tier weight system:

1. **Master weights** (`lora_dtype`): Stored in the specified precision for optimizer updates and checkpoint export. Defaults to fp32 for numerical stability with the 8-bit AdamW optimizer.
2. **Work weights** (`model_dtype`): Converted from master weights before each forward pass. Used for the actual LoRA computation (forward/backward matmuls).

**Weight flow:**

```
lora_dtype (master)  →  convert  →  model_dtype (work)  →  forward/backward
     ↑                                                            │
     └────────────────── optimizer update ←───────────────────────┘
```

**Why fp32 is the default for LoRA:**

- LoRA adapters are small (low-rank), so fp32 storage overhead is minimal
- The 8-bit AdamW optimizer benefits from fp32 master weights for stable updates
- FP32 provides better precision for the optimizer's running statistics

**Supported configurations:**

| lora_dtype | model_dtype | Behavior                                       |
| ---------- | ----------- | ---------------------------------------------- |
| fp32       | bf16        | FP32 masters → BF16 work weights (recommended) |
| bf16       | bf16        | No conversion needed, same dtype               |
| fp32       | fp32        | No conversion needed, same dtype               |
| bf16       | fp32        | BF16 masters → FP32 work weights               |
