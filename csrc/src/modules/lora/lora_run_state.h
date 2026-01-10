// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_RUN_STATE_H
#define SUROGATE_SRC_MODULES_LORA_LORA_RUN_STATE_H

#include "utilities/tensor.h"

namespace modules {

/**
 * @brief Runtime state for LoRA execution
 */
struct LoRARunState {
    Tensor intermediate;   // (BT, rank) - first intermediate buffer
    Tensor intermediate2;  // (BT, rank) - second intermediate buffer for fused ops
    Tensor slice;
    Tensor norm_buffer;
    Tensor recompute_ln;   // (B, T, C) - buffer for recomputed ln1/ln2 activations
    Tensor recompute_rstd; // (B, T) - buffer for recomputed rstd (unused but required by kernel)
    int B = 0;
    int T = 0;

    // MoE expert LoRA state: pointers to current expert activations during hook execution.
    // These are set by the MoE block before calling expert hooks and read by the hook callback.
    // This avoids the need to pass activation pointers through the hook signature.
    struct MoEExpertContext {
        Tensor* expert_input = nullptr;   // (N, C) - input tokens routed to this expert
        Tensor* gate_up = nullptr;        // (N, 2*D) - gate_up projection output
        Tensor* activated = nullptr;      // (N, D) - activated value after SwiGLU
        Tensor* output = nullptr;         // (N, C) - down projection output
        int num_tokens = 0;               // N - number of tokens for this expert
    };
    MoEExpertContext moe_expert_ctx;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_RUN_STATE_H
