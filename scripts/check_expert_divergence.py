import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import os

"""
Comprehensive MoE health check script that analyzes:
1. Expert weight divergence - Are experts different from each other?
2. Router behavior - Is the router making confident, specialized routing decisions?

When you first upcycle a model, experts are clones (cosine similarity = 1.0000).
As training progresses, experts should diverge and the router should learn to use them effectively.

Weight Divergence Thresholds:
- Similarity > 0.9999: Experts are still clones, router has no signal
- Similarity < 0.9990: Tipping point - experts diverging, router can start learning
- Similarity < 0.990: Good divergence, experts are specializing

Router Confidence Thresholds (8 experts, random = 0.125):
- Confidence > 0.30: High certainty, router has strong opinions
- Confidence > 0.15: Emerging specialization
- Confidence < 0.15: Effectively random routing
"""
def check_expert_divergence(model_path, layer_idx=17):
    """Check weight divergence between experts."""
    print(f"\n{'='*60}")
    print("PART 1: EXPERT WEIGHT DIVERGENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Loading model from {model_path}...")

    # Path to the weights file
    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path, device="cpu")

    # Create the model shell using your config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.bfloat16)

    print(f"Analyzing Layer {layer_idx} (middle layer)...")
    
    # Surogate stores all experts in a single tensor: [num_experts, hidden_dim*2, input_dim]
    all_experts_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj.weight"

    if all_experts_key in state_dict:
        all_experts_weight = state_dict[all_experts_key]
        print(f"Found fused expert tensor with shape: {all_experts_weight.shape}")

        for expert_idx in range(8):
            # Extract this expert's fused weight: [6144, 1024]
            fused_weight = all_experts_weight[expert_idx]
            # Split into gate [3072, 1024] and up [3072, 1024]
            gate_w, up_w = torch.chunk(fused_weight, 2, dim=0)

            model.model.layers[layer_idx].mlp.experts[expert_idx].gate_proj.weight.data.copy_(gate_w)
            model.model.layers[layer_idx].mlp.experts[expert_idx].up_proj.weight.data.copy_(up_w)
    else:
        print(f"Error: Could not find key {all_experts_key}")
        print("Available keys with 'expert':", [k for k in state_dict.keys() if 'expert' in k.lower()][:5])

    # Compare Expert 0 and Expert 7
    e0_w = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.data
    e7_w = model.model.layers[layer_idx].mlp.experts[7].gate_proj.weight.data

    cos_sim = torch.nn.functional.cosine_similarity(
        e0_w.flatten().to(torch.float32), 
        e7_w.flatten().to(torch.float32), 
        dim=0
    ).item()

    print(f"\nCosine Similarity (Expert 0 vs Expert 7): {cos_sim:.10f}")

    if cos_sim > 0.9999999:
        print("Status: 🔴 Experts are EXACT CLONES.")
        print("Diagnosis: The router has zero signal to work with yet.")
        divergence_status = "clones"
    elif cos_sim > 0.999:
        print("Status: 🟡 Divergence detected!")
        print("Diagnosis: Training is successfully 'un-cloning' the experts.")
        divergence_status = "diverging"
    else:
        print("Status: 🟢 Specialized Experts!")
        divergence_status = "specialized"

    return model, config, state_dict, divergence_status


def check_router_behavior(model_path, model=None, config=None, state_dict=None, layer_idx=17):
    """Check router confidence and routing patterns."""
    print(f"\n{'='*60}")
    print("PART 2: ROUTER BEHAVIOR ANALYSIS")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model if not provided
    if model is None or config is None or state_dict is None:
        weights_path = os.path.join(model_path, "model.safetensors")
        state_dict = load_file(weights_path, device="cpu")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.bfloat16)

    # Move model to GPU
    model = model.to(device)

    # Map router/gate weights for all layers
    print("Loading router weights...")
    for i in range(config.num_hidden_layers):
        gate_key = f"model.layers.{i}.mlp.gate.weight"
        if gate_key in state_dict:
            model.model.layers[i].mlp.gate.weight.data.copy_(state_dict[gate_key].to(device))

    # Map expert weights for the target layer
    print(f"Loading expert weights for layer {layer_idx}...")
    all_experts_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj.weight"

    if all_experts_key in state_dict:
        all_experts_weight = state_dict[all_experts_key]
        num_experts = all_experts_weight.shape[0]

        for expert_idx in range(num_experts):
            fused_weight = all_experts_weight[expert_idx].to(device).to(torch.bfloat16)
            gate_w, up_w = torch.chunk(fused_weight, 2, dim=0)
            model.model.layers[layer_idx].mlp.experts[expert_idx].gate_proj.weight.data.copy_(gate_w)
            model.model.layers[layer_idx].mlp.experts[expert_idx].up_proj.weight.data.copy_(up_w)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Test cases with domain-specific content
    test_cases = {
        "Python Code": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Literature": "Through the shimmering veil of time, the wanderer glimpsed a forgotten empire.",
        "Math": "The quadratic formula is x = (-b ± sqrt(b² - 4ac)) / 2a",
        "Casual": "Hey, how's it going? I was thinking we could grab some coffee later."
    }

    print(f"\nTesting router behavior on diverse inputs (Layer {layer_idx}):\n")

    all_confidences = []
    for label, text in test_cases.items():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        # Get router logits for the target layer [Batch, Seq, Experts]
        logits = outputs.router_logits[layer_idx].to(torch.float32)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [Seq, Experts]

        # Confidence: Average probability assigned to the #1 choice
        top_probs, top_indices = torch.topk(probs, k=1, dim=-1)
        avg_confidence = top_probs.mean().item()
        all_confidences.append(avg_confidence)

        # Usage: Unique experts used in this sequence
        unique_experts = torch.unique(top_indices).tolist()

        print(f"[{label}]")
        print(f"  Router Confidence: {avg_confidence:.4f}")
        print(f"  Experts Used:      {unique_experts}")

    # Overall assessment
    overall_confidence = sum(all_confidences) / len(all_confidences)
    print(f"\n{'='*60}")
    print(f"Average Router Confidence: {overall_confidence:.4f}")
    print(f"Random baseline (8 experts): 0.125")

    if overall_confidence > 0.30:
        print("\nStatus: 🟢 HIGH CERTAINTY")
        print("Diagnosis: Router has strong opinions, making confident decisions.")
        router_status = "confident"
    elif overall_confidence > 0.15:
        print("\nStatus: 🟡 EMERGING SPECIALIZATION")
        print("Diagnosis: Router is beginning to differentiate between experts.")
        router_status = "emerging"
    else:
        print("\nStatus: 🔴 RANDOM ROUTING")
        print("Diagnosis: Router is effectively guessing (close to random baseline).")
        router_status = "random"

    return router_status


def check_moe_health(model_path, layer_idx=17):
    """Comprehensive MoE health check combining divergence and router analysis."""
    print(f"\n{'#'*60}")
    print(f"# MoE MODEL HEALTH CHECK")
    print(f"# Checkpoint: {model_path}")
    print(f"{'#'*60}")

    # Part 1: Check expert weight divergence
    model, config, state_dict, divergence_status = check_expert_divergence(model_path, layer_idx)

    # Part 2: Check router behavior
    router_status = check_router_behavior(model_path, model, config, state_dict, layer_idx)

    # Final summary
    print(f"\n{'='*60}")
    print("OVERALL HEALTH SUMMARY")
    print(f"{'='*60}")
    print(f"Expert Divergence:  {divergence_status}")
    print(f"Router Behavior:    {router_status}")

    # Combined diagnosis
    if divergence_status == "specialized" and router_status == "confident":
        print("\n🟢 EXCELLENT: Experts are specialized and router is confident.")
        print("   Your MoE model is training effectively!")
    elif divergence_status in ["diverging", "specialized"] and router_status == "emerging":
        print("\n🟡 GOOD PROGRESS: Experts are diverging and router is learning.")
        print("   Continue training - the model is on the right track.")
    elif divergence_status == "clones":
        print("\n🔴 EARLY STAGE: Experts are still clones.")
        print("   This is expected early in training. The high aux_loss will")
        print("   push experts to diverge. Be patient and monitor progress.")
    else:
        print(f"\n🟡 MIXED SIGNALS: Divergence={divergence_status}, Router={router_status}")
        print("   The model is learning but may need more training steps.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Update this to your latest checkpoint directory
    check_moe_health("./output_moe/step_00000200")