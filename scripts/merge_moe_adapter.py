#!/usr/bin/env python3
"""
Merge MoE LoRA adapter into base model with memory-efficient CPU offloading.

This script handles Surogate's grouped expert LoRA format and converts it to
standard HuggingFace per-expert format compatible with vLLM.

Usage:
    python scripts/merge_moe_adapter.py \
        --base-model ./moe \
        --adapter ./output_moe/adapter \
        --output ./output_moe/merged \
        --max-shard-size 5GB
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM


def load_adapter_weights(adapter_path: str) -> Dict[str, torch.Tensor]:
    """Load LoRA adapter weights from safetensors."""
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")

    print(f"Loading adapter weights from {adapter_file}...")
    weights = load_file(adapter_file)
    print(f"Loaded {len(weights)} adapter tensors")
    return weights


def convert_grouped_to_per_expert(
    grouped_weights: Dict[str, torch.Tensor],
    num_experts: int
) -> Dict[str, torch.Tensor]:
    """
    Convert Surogate's grouped expert LoRA format to per-expert format.

    Input format:  base_model.model.model.layers.0.mlp.experts.grouped.gate_proj.lora_A  [rank, hidden]
                   base_model.model.model.layers.0.mlp.experts.grouped.gate_proj.lora_B  [num_experts*intermediate, rank]

    Output format: model.layers.0.mlp.experts.0.gate_proj.lora_A  [rank, hidden]
                   model.layers.0.mlp.experts.0.gate_proj.lora_B  [intermediate, rank]
                   model.layers.0.mlp.experts.1.gate_proj.lora_A  [rank, hidden]
                   ...
    """
    per_expert_weights = {}

    for key, tensor in grouped_weights.items():
        # Strip PEFT prefix if present (base_model.model.)
        clean_key = key
        if key.startswith("base_model.model."):
            clean_key = key[len("base_model.model."):]

        # Strip .weight suffix if present
        if clean_key.endswith(".weight"):
            clean_key = clean_key[:-len(".weight")]

        if "experts.grouped" not in clean_key:
            # Non-expert weights (attention layers) - keep as-is
            per_expert_weights[clean_key] = tensor
            continue

        # Parse grouped expert key
        # Example: model.layers.0.mlp.experts.grouped.gate_proj.lora_B
        parts = clean_key.split(".")
        grouped_idx = parts.index("grouped")
        layer_prefix = ".".join(parts[:grouped_idx])  # model.layers.0.mlp.experts
        projection = parts[grouped_idx + 1]  # gate_proj, up_proj, down_proj
        lora_type = parts[grouped_idx + 2]  # lora_A, lora_B

        # Check if tensor has expert dimension as first dim
        if tensor.dim() == 3 and tensor.shape[0] == num_experts:
            # Format: [num_experts, ...] → extract per-expert slices
            for expert_id in range(num_experts):
                expert_tensor = tensor[expert_id].clone()
                new_key = f"{layer_prefix}.{expert_id}.{projection}.{lora_type}"
                per_expert_weights[new_key] = expert_tensor
        else:
            # Unexpected format - keep as-is and warn
            print(f"Warning: Unexpected tensor shape for {clean_key}: {tensor.shape}")
            per_expert_weights[clean_key] = tensor

    print(f"Converted {len(grouped_weights)} grouped weights → {len(per_expert_weights)} per-expert weights")
    return per_expert_weights


def merge_lora_into_linear(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_alpha: float,
    lora_rank: int,
    scaling: float = None
) -> torch.Tensor:
    """Merge LoRA weights into base linear layer: W' = W + (B @ A) * scaling."""
    if scaling is None:
        scaling = lora_alpha / lora_rank

    # Compute LoRA delta: B @ A
    # lora_A: [rank, in_features]
    # lora_B: [out_features, rank]
    # delta: [out_features, in_features]
    delta = (lora_B @ lora_A) * scaling

    # Add to base weight
    merged = base_weight + delta
    return merged


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    max_shard_size: str = "5GB",
    cpu_offload: bool = True
):
    """
    Merge LoRA adapter into base model with CPU offloading for memory efficiency.
    """
    print("="*80)
    print("MoE LoRA Adapter Merge")
    print("="*80)

    # Load adapter config
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)

    lora_alpha = adapter_config["lora_alpha"]
    lora_rank = adapter_config["r"]
    target_modules = adapter_config["target_modules"]

    print(f"\nAdapter config:")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  Target modules: {target_modules}")

    # Load base model config
    base_config = AutoConfig.from_pretrained(base_model_path)
    num_experts = getattr(base_config, "num_experts", 8)
    print(f"\nBase model:")
    print(f"  Type: {base_config.model_type}")
    print(f"  Num experts: {num_experts}")
    print(f"  Num layers: {base_config.num_hidden_layers}")

    # Load adapter weights
    adapter_weights = load_adapter_weights(adapter_path)

    # Convert grouped to per-expert format
    print("\nConverting grouped expert format to per-expert format...")
    per_expert_weights = convert_grouped_to_per_expert(adapter_weights, num_experts)

    # Load base model (CPU offload if needed)
    print(f"\nLoading base model from {base_model_path}...")
    device_map = "cpu" if cpu_offload else "auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True
    )

    print("\nMerging LoRA weights into base model...")
    # Build LoRA pair mapping: {base_key: (lora_A_key, lora_B_key)}
    lora_pairs = {}
    for key in per_expert_weights:
        if key.endswith(".lora_A"):
            base_key = key.replace(".lora_A", "")
            lora_a_key = key
            lora_b_key = key.replace(".lora_A", ".lora_B")
            if lora_b_key in per_expert_weights:
                lora_pairs[base_key] = (lora_a_key, lora_b_key)

    print(f"Found {len(lora_pairs)} LoRA pairs to merge")

    # Merge each LoRA pair into base model
    merged_count = 0
    base_state_dict = base_model.state_dict()

    for base_key, (lora_a_key, lora_b_key) in lora_pairs.items():
        # Base model has .weight suffix, but our keys don't
        base_key_with_weight = base_key + ".weight"
        if base_key_with_weight not in base_state_dict:
            print(f"Warning: Base weight not found: {base_key_with_weight}")
            continue

        base_weight = base_state_dict[base_key_with_weight]
        lora_A = per_expert_weights[lora_a_key].to(base_weight.device)
        lora_B = per_expert_weights[lora_b_key].to(base_weight.device)

        merged_weight = merge_lora_into_linear(
            base_weight, lora_A, lora_B,
            lora_alpha, lora_rank
        )

        base_state_dict[base_key_with_weight] = merged_weight
        merged_count += 1

        if merged_count % 100 == 0:
            print(f"  Merged {merged_count}/{len(lora_pairs)} weights...")

    print(f"\n✓ Merged {merged_count} LoRA weights into base model")

    # Load merged weights back into model
    base_model.load_state_dict(base_state_dict)

    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    base_model.save_pretrained(
        output_path,
        max_shard_size=max_shard_size,
        safe_serialization=True
    )

    # Copy tokenizer files
    print("Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt"
    ]
    for filename in tokenizer_files:
        src = os.path.join(base_model_path, filename)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(output_path, filename))

    print("\n" + "="*80)
    print("✓ Merge complete!")
    print("="*80)
    print(f"\nMerged model saved to: {output_path}")
    print("\nYou can now load this model in vLLM:")
    print(f"  vllm serve {output_path} --dtype bfloat16")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Merge MoE LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, required=True,
                        help="Path to base model directory")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to adapter directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for merged model")
    parser.add_argument("--max-shard-size", type=str, default="5GB",
                        help="Max shard size (default: 5GB)")
    parser.add_argument("--no-cpu-offload", action="store_true",
                        help="Don't use CPU offloading (requires more GPU memory)")

    args = parser.parse_args()

    merge_adapter(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_path=args.output,
        max_shard_size=args.max_shard_size,
        cpu_offload=not args.no_cpu_offload
    )


if __name__ == "__main__":
    main()
