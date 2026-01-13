# Training Modes

Surogate supports three practical ways to adapt a model:

1) **Pretraining / Continued Pretraining (PT)**
2) **Full Fine-Tuning**
3) **LoRA / QLoRA (Adapter Fine-Tuning)**

They differ in *which parameters are updated*, *how much data you need*, and *how much compute/VRAM you’ll spend*.

---

## Quick decision guide

| Goal | Recommended mode |
| --- | --- |
| Train a base model from scratch on large text | **Pretraining** |
| Continue training a base model on more text (domain adaptation) | **Continued pretraining** |
| Maximize quality for a specific task/domain and you can afford it | **Full fine-tuning** |
| Fast adaptation, smaller GPU, cheaper runs, easy iteration | **LoRA** |
| Same as LoRA but you’re VRAM-limited | **QLoRA** |

---

## 1) Pretraining / Continued Pretraining (PT)

### How it works
**Pretraining** updates *all* (or nearly all) model weights by predicting the next token on large-scale raw text.

- **What updates?** Base model weights (full model training)
- **Typical data:** Raw text corpora (`type: text` datasets)
- **Typical run shape:** Lots of tokens, long runs, throughput-focused

In Surogate, you typically run PT with:

```bash
surogate pt path/to/config.yaml
```

### When to use it
- You’re training a new base model.
- You want **domain adaptation** via continued pretraining (e.g., legal/medical/finance corpora).
- You care about general capabilities and broad distribution learning.

### Tradeoffs
- Highest compute + longest wall-clock.
- Requires the most data.

---

## 2) Full Fine-Tuning (update all weights)

### How it works
**Full fine-tuning** starts from a pretrained checkpoint and updates the full parameter set on a task/domain dataset.

- **What updates?** Base model weights (and optionally embeddings / lm_head depending on your config)
- **Typical data:** Instruction or conversation datasets
- **Typical run shape:** Fewer tokens than PT, but still heavy VRAM/compute

In Surogate, full fine-tuning is usually done through the SFT workflow with LoRA disabled:

```yaml
lora: false
```

(and then run via `surogate sft ...`).

### When to use it
- You need the *best possible* task/domain performance.
- You can afford the VRAM/compute of updating all weights.
- You don’t need easy “adapter swapping” across many downstream tasks.

### Tradeoffs
- More expensive and heavier than LoRA.
- Harder to maintain multiple downstream variants (each run produces a full checkpoint).

---

## 3) LoRA / QLoRA (adapter fine-tuning)

### How LoRA works
**LoRA** freezes the base model weights and trains small low-rank adapter matrices inserted into selected linear layers.

- **What updates?** Only LoRA adapter parameters
- **Base weights:** Frozen (unchanged)
- **Typical data:** Instruction / conversation / task datasets

In config, you enable LoRA with:

```yaml
lora: true
lora_rank: 16
lora_alpha: 32
```

### When to use LoRA
- You want fast iteration and lower cost.
- You want multiple downstream specializations (adapters are easy to store, ship, and swap).
- You’re experimenting and want quick turnarounds.

### What is QLoRA?
**QLoRA** is LoRA plus quantization of the frozen base model weights to reduce VRAM.

- **What updates?** LoRA adapters (still)
- **Base weights:** Frozen and stored in a quantized format (FP8 / FP4 / NF4)
- **When to use it:** When VRAM is the bottleneck

Surogate supports:
- `qlora_fp8` (SM89+)
- `qlora_fp4` (SM100+ Blackwell)
- `qlora_bnb` (NF4 via BitsAndBytes, broad compatibility)

---

## Practical recommendations

- Start with **LoRA (bf16 recipe)** unless you have a strong reason not to.
- Use **QLoRA** when you can’t fit the base model + activations comfortably.
- Use **Full fine-tuning** when you want maximum quality and have budget.
- Use **(Continued) pretraining** for domain adaptation on large raw text.

---

## See also

- [Quickstart: Pretraining](quickstart-pretraining.md)
- [Quickstart: Supervised Fine-Tuning](quickstart-sft.md)
- [Configuration](../guides/configuration.md)
- [Precision & recipes](../guides/precision-and-recipes.md)
- [QLoRA](../guides/qlora.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
