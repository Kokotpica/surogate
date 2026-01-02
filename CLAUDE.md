# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surogate is a high-performance LLM pre-training/fine-tuning framework with a C++/CUDA core and a Python CLI wrapper. The training engine is implemented in C++/CUDA and exposed to Python via nanobind bindings.

## Where to start (common tasks)

- **Run training from Python CLI**: `surogate/cli/main.py`, `surogate/train/*`
- **Training engine entry point**: `csrc/src/train.cpp`
- **Performance-sensitive code**: `csrc/src/kernels/`, `csrc/src/modules/`, `csrc/src/recipes/`
- **Bindings boundary**: `csrc/src/binding/` (keep APIs stable; avoid unnecessary churn)

When changing C++/CUDA code:
- prefer localized changes (kernels/modules/recipes) over broad refactors
- keep recipe defaults explicit and documented in the recipe README
- validate build + a minimal training run when possible

## Build Commands

```bash
# Build everything (C++ core + Python bindings)
make build

# Build only the train executable
make train

# Build Python wheel
make wheel

# Build wheel in development mode (faster iteration)
make wheel-dev

# Build and run unit tests
make test

# Clean build artifacts
make clean

# Full rebuild from scratch
make rebuild
```

### Build Options

```bash
# Debug build
make BUILD_TYPE=Debug build

# Without MPI support
make CMAKE_FLAGS='-DUSE_MPI=OFF' build
```

## Running the CLI

```bash
# Supervised Fine-Tuning
surogate sft --config path/to/config.yaml

# Tokenize datasets
surogate tokenize --config path/to/config.yaml
```

## Architecture

### C++ Core (`csrc/`)

- `src/train.cpp` - Main training executable entry point
- `src/training/` - Training loop, checkpointing, data loading
- `src/models/` - Model implementations (Qwen2.5, Qwen3, Llama)
- `src/kernels/` - CUDA kernels (attention, RMSNorm, RoPE, SwiGLU, etc.)
- `src/modules/` - Module implementations including QLoRA (FP8/FP4)
- `src/recipes/` - Mixed precision training recipes:
  - `bf16/` - Baseline BF16 training
  - `fp8_hybrid/` - FP8 E4M3 forward / E5M2 backward with delayed scaling
  - `nvfp4/` - FP4 E2M1 with two-level block scaling (Blackwell GPUs)
- `src/utilities/` - Allocators, tensor utilities, communication (NCCL)
- `src/binding/` - nanobind Python bindings

### Python Layer (`surogate/`)

- `cli/main.py` - CLI entry point, dispatches to subcommands
- `train/sft.py` - SFT training orchestration
- `train/trainer.py` - `SurogateTrainerWrapper` wraps C++ trainer
- `core/config/` - Configuration dataclasses (SFTConfig, ModelConfig, etc.)
- `core/datasets/` - Dataset loading and preprocessing
- `core/model/` - Model loading, chat templates, LoRA patching
- `utils/` - Logging, filesystem, HuggingFace utilities

### Build System

- CMake-based build with scikit-build-core for Python packaging
- Dependencies fetched via CMake FetchContent: CUTLASS, cuDNN-FE, CLI11, nlohmann/json, nanobind, fmt, Catch2
- Requires CUDA 12.x+, NCCL, cuDNN

## Mixed Precision Recipes

| Recipe | Format | GPU Requirement |
|--------|--------|-----------------|
| `bf16` | BF16 forward/backward | Any CUDA GPU |
| `fp8_hybrid` | FP8 E4M3 fwd / E5M2 bwd | SM89+ (Ada, Hopper, Blackwell) |
| `nvfp4` | FP4 E2M1 with block scaling | SM100+ (Blackwell only) |

## Configuration

Training is configured via YAML files. Example structure:

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
sequence_len: 2048
learning_rate: 2e-4
lora: true
lora_rank: 16
datasets:
  - path: "mlabonne/FineTome-100k"
    type: auto
```

## Key Configuration Options

- **Recomputation**: `recompute_block`, `recompute_att`, `recompute_ffn`, `recompute_qkv`, `recompute_swiglu`, `recompute_rmsnorm` - Trade compute for memory
- **Offloading**: `offload_optimizer`, `offload_grads`, `offload_residual`, `offload_master` - Offload tensors to host memory
- **ZeRO**: `zero_level` (1-3), `shard_weights`, `shard_gradients` - Distributed training optimizations
- **QLoRA**: `qlora_fp8` or `qlora_fp4` - Quantized base weights with LoRA adapters
