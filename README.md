<div align="center">
<a href="https://surogate.ai/">
<img width="64" alt="Surogate logo" src="./assets/img/logo-white.svg" />
</a>

<h1>Surogate</h1>
<h3>High-performance mixed-precision LLM pre-training & fine-tuning (C++/CUDA core, Python CLI/Wrapper, BF16, FP8 and NVFP4)</h3>

Runs on all NVIDIA GPUs from 
<br/>

<a href="https://surogate.ai">Home</a> ·
<a href="https://docs.surogate.ai">Docs</a> ·
<a href="https://github.com/invergent-ai/surogate/tree/master/examples">Examples</a> ·
<a href="./benchmarks/unsloth.md">Benchmarks</a>

[![GitHub stars](https://img.shields.io/github/stars/invergent-ai/surogate?style=social)](https://github.com/invergent-ai/surogate)
[![GitHub issues](https://img.shields.io/github/issues/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/invergent-ai/surogate)](https://github.com/invergent-ai/surogate/pulls)
[![Twitter Follow](https://img.shields.io/twitter/follow/invergentai?style=social)](https://twitter.com/invergentai)

<br/>
<b>If Surogate saves you time or GPUs, consider ⭐ starring ⭐ the repo.</b>
</div>

---

## What is Surogate?

Surogate is an **enterprise-grade LLM training framework** focused on **throughput, latency, and predictable scaling**.
It combines a **native C++/CUDA training engine** with a **low-overhead Python interface** and a **multi-threaded scheduler** for efficient multi-GPU execution, achieving the highest SOL (Speed-Of-Light) on current NVIDIA GPUs, supassing any existing training toolkit by a large margin.

If you care about:
- faster training iterations,
- consistent multi-GPU scaling,
- and production-minded training primitives (recipes, checkpointing, sharding/offload),

…Surogate is built for you.

> **Performance notes: Surogate targets near hardware-limited throughput.** See reproducible comparisons in [`./benchmarks`](./benchmarks/unsloth.md).

---

## Highlights
Surogate is built for developers and enterprises that need fast experimentation scalability and predictable outcomes — whether running on-premise, in private clouds, or inside turnkey systems such as the [DenseMAX Appliance](https://www.invergent.ai/densemax-appliance).

- **Pre-training + Fine-tuning**: full fine-tune, LoRA, QLoRA
- **Native multi-GPU** training
- **Mixed-precision recipes** tuned per architecture (BF16 / FP8 / NVFP4)
- **Optimized QLoRA** with native FP8/NVFP4 quantization for maximum performance !
- **Modern GPU support**: sm80, sm86, sm89, sm90, sm100, sm103, sm120
- **Designed for reliability**: deterministic configs, explicit recipes, and a clear C++ core
- **Pre-builtTaining recipes**: 
  - [**BF16**](./csrc/src/recipes/bf16/README.md): Baseline recipe using `bfloat16` for all GEMMs, designed for maximum numerical accuracy. No quantization is applied.
  - [**FP8**](./csrc/src/recipes/fp8_hybrid/README.md): Native `FP8` training delivering extreme performance with `E4M3` used for activations and weights and `E5M2` for gradients. Uses per-tensor delayed scaling to provide stable training.
  - [**NVFP4**](./csrc/src/recipes/nvfp4/README.md): Native CUTLASS `FP4 E2M1` training with two-level block scaling for extreme performance and memory efficiency on Blackwell GPUs (**sm100 and sm120**). Uses stochastic rounding and random Hadamard Transforms for numerical stability. **Supports NVIDIA RTX 5070, 5080, 5090 !!**
---

## Hardware / Requirements

- NVIDIA GPU + recent driver
- CUDA **12/13**, NCCL, cuDNN
- Linux

### Supported NVIDIA GPUs:
- `SM80`: A100, A30
- `SM86`: A2, A16, A10, A40, RTX3050, RTX3060, RTX 3070, RTX 3080, RTX 3090, A2000, A3000, A4000, A5000, A6000
- `SM89`: L4, L40, L40S, RTX 4050, RTX 4060, RTX 4070, RTX 4080, RTX 4090, RTX 2000 Ada, RTX 4000 SFF Ada, RTX 4000 Ada, RTX 4500 Ada, RTX 5000 Ada, RTX 6000 Ada
- `SM90`: H100, H200, GH200
- `SM100`: B200, GB200
- `SM103`: B300, GB300
- `SM120`: RTX PRO 6000/5000/4000/2500/2000 Blackwell,  RTX 5050,  RTX 5060,  RTX 5070,  RTX 5080,  RTX 5090
- `SM121`: DGX Spark (in progress)
  
---

## Install

### Option A: Install via script (recommended)
```bash
curl -LsSf https://surogate.ai/install.sh | sh
```

### Option B: Build from source (dev / contributors)
```bash
# ...clone repo...
make build
# or: make wheel / make wheel-dev
```

---

## Quickstart (SFT)

1) Create a config (example):

```yaml
model: Qwen/Qwen3-0.6B
output_dir: ./output

# training
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
sequence_len: 2048
learning_rate: 2e-4

# LoRA / QLoRA
lora: true
lora_rank: 16
# qlora_fp8: true  # optional, hardware-dependent
# qlora_fp4: true  # Blackwell+

datasets:
  - path: "mlabonne/FineTome-100k"
    type: auto
```

2) Run:
```bash
surogate sft --config path/to/config.yaml
```

3) Outputs:
- checkpoints/logs/artifacts are written under `output_dir`

---

## Documentation / Examples

- Docs: https://docs.surogate.ai
- Examples: https://github.com/invergent-ai/surogate/tree/master/examples
- Benchmarks: [`./benchmarks/unsloth.md`](./benchmarks/unsloth.md)

---

## Contributing

PRs and issues are welcome.
If you’re adding kernels/recipes or touching build/tooling, please keep changes minimal and include:
- a short description of the change,
- how to reproduce/validate locally (`make test` where applicable),
- and any GPU/arch assumptions.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).

