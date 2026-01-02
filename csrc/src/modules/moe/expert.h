// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_EXPERT_H
#define SUROGATE_SRC_MODULES_MOE_EXPERT_H

#include "modules/module_base.h"
#include "modules/primitives/linear.h"
#include "modules/primitives/swiglu.h"
#include "kernels/kernels.h"
#include "router.h"  // For RouterModule::RouterOutput

namespace modules {

/**
 * @brief Single Expert MLP module
 *
 * An expert is essentially an MLP (same structure as the dense FFN).
 * In MoE, multiple experts share the same structure but have independent weights.
 *
 * Structure: input -> up_proj -> activation -> down_proj -> output
 * With gated variants: input -> [gate_proj, up_proj] -> activation(gate) * up -> down_proj
 */
class ExpertModule : public ModuleBase<ExpertModule> {
public:
    /**
     * @brief Configuration for a single expert
     */
    struct Config {
        int hidden_size;            ///< Input/output dimension
        int intermediate_size;      ///< FFN intermediate dimension
        bool use_gated = true;      ///< Use gated activation (SwiGLU)
        float dropout = 0.0f;       ///< Dropout probability
    };

    /**
     * @brief Weight tensors for a single expert
     */
    struct Weights {
        Tensor gate_proj;           ///< (hidden_size, intermediate_size) - only if gated
        Tensor up_proj;             ///< (hidden_size, intermediate_size)
        Tensor down_proj;           ///< (intermediate_size, hidden_size)
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        Tensor input;               ///< Input to expert
        Tensor gate_up;             ///< Output of gate+up projection (before activation)
        Tensor activated;           ///< Output after activation
        Tensor output;              ///< Final output
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_gate_proj;
        Tensor d_up_proj;
        Tensor d_down_proj;
    };

    explicit ExpertModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass through expert MLP
     *
     * @param ctx Module context
     * @param w Expert weights
     * @param input Token representations routed to this expert (N, hidden_size)
     * @param acts Activation storage
     * @return Output (N, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass through expert MLP
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;
};

// ============================================================================
// Implementation
// ============================================================================

inline Tensor ExpertModule::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {

    const int N = input.Sizes[0];  // Number of tokens routed to this expert
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    acts.input = input;

    if (mConfig.use_gated) {
        // Gated activation (SwiGLU style)
        // gate_up has shape (N, 2*D) with [gate, up] concatenated

        // Combined projection: gate_up = input @ [gate_proj; up_proj]
        // For efficiency, gate and up projections are often fused
        matmul(
            acts.gate_up, w.gate_proj, input, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            2 * D, N, C, EMMTranspose::TN, false,
            ctx.stream
        );

        // SwiGLU activation: activated = swiglu(gate_up)
        swiglu_forward(acts.activated, acts.gate_up, nullptr, 1, N, D, ctx.stream);

    } else {
        // Simple activation (ReLU/GeLU)
        matmul(
            acts.gate_up, w.up_proj, input, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            D, N, C, EMMTranspose::TN, false,
            ctx.stream
        );

        // Apply activation (GeLU)
        // gelu_forward(acts.activated, acts.gate_up, N * D, ctx.stream);
        acts.activated = acts.gate_up;  // Placeholder
    }

    // Down projection: output = activated @ down_proj
    matmul(
        acts.output, w.down_proj, acts.activated, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, N, D, EMMTranspose::TN, false,
        ctx.stream
    );

    return acts.output;
}

inline Tensor ExpertModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_output, Gradients& grads, bool accumulate) {

    const int N = acts.input.Sizes[0];
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // Backward through down projection
    // d_activated = grad_output @ down_proj^T
    Tensor d_activated;
    d_activated.DType = grad_output.DType;
    matmul(
        d_activated, w.down_proj, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        D, N, C, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_down_proj = activated^T @ grad_output
    matmul(
        grads.d_down_proj, acts.activated, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        D, C, N, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    // Backward through activation
    Tensor d_gate_up;
    d_gate_up.DType = d_activated.DType;
    if (mConfig.use_gated) {
        // SwiGLU backward
        swiglu_backward(d_gate_up, d_activated, acts.gate_up, nullptr, 1, N, D, ctx.stream);
    } else {
        // GeLU backward
        // gelu_backward(d_gate_up, d_activated, acts.gate_up, N * D, ctx.stream);
        d_gate_up = d_activated;  // Placeholder
    }

    // Backward through gate/up projection
    // d_input = d_gate_up @ gate_proj^T (or up_proj^T if not gated)
    Tensor d_input;
    d_input.DType = acts.input.DType;
    matmul(
        d_input, w.gate_proj, d_gate_up, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, N, 2 * D, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_gate_proj = input^T @ d_gate_up
    matmul(
        grads.d_gate_proj, acts.input, d_gate_up, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, 2 * D, N, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    return d_input;
}

/**
 * @brief Expert group - collection of experts with shared dispatch logic
 *
 * Manages multiple experts and handles the scatter/gather operations
 * for routing tokens to experts and combining outputs.
 */
class ExpertGroupModule : public ModuleBase<ExpertGroupModule> {
public:
    /**
     * @brief Configuration for expert group
     */
    struct Config {
        int num_experts;            ///< Number of experts in the group
        int hidden_size;            ///< Input/output dimension
        int intermediate_size;      ///< FFN intermediate dimension per expert
        int top_k = 2;              ///< Number of experts per token
        int capacity_factor = 1;    ///< Capacity multiplier (tokens per expert = capacity_factor * tokens / num_experts * top_k)
        bool use_gated = true;      ///< Use gated activation
    };

    /**
     * @brief Weights for all experts
     */
    struct Weights {
        // Option 1: Separate weights per expert
        std::vector<ExpertModule::Weights> experts;

        // Option 2: Batched weights for efficient computation
        // Tensor gate_proj;  // (num_experts, hidden_size, intermediate_size)
        // Tensor up_proj;    // (num_experts, hidden_size, intermediate_size)
        // Tensor down_proj;  // (num_experts, intermediate_size, hidden_size)
    };

    /**
     * @brief Activations for all experts
     */
    struct Activations {
        std::vector<ExpertModule::Activations> expert_acts;

        // Dispatch/combine state
        Tensor dispatched_input;    ///< (num_experts, capacity, hidden_size)
        Tensor expert_outputs;      ///< (num_experts, capacity, hidden_size)
        Tensor combined_output;     ///< (B*T, hidden_size)
    };

    /**
     * @brief Gradients for all experts
     */
    struct Gradients {
        std::vector<ExpertModule::Gradients> experts;
    };

    explicit ExpertGroupModule(Config config);

    /**
     * @brief Forward pass: dispatch tokens to experts and combine outputs
     *
     * @param ctx Module context
     * @param w Expert weights
     * @param input Token representations (B*T, hidden_size)
     * @param routing Router output with dispatch information
     * @param acts Activation storage
     * @return Combined output (B*T, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input,
                        const RouterModule::RouterOutput& routing, Activations& acts);

    /**
     * @brief Backward pass through expert group
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         const RouterModule::RouterOutput& routing,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

private:
    Config mConfig;
    std::vector<ExpertModule> mExperts;

    // Helper functions for token dispatch/combine
    void dispatch_tokens(Tensor& input, const RouterModule::RouterOutput& routing,
                         Tensor& dispatched, cudaStream_t stream);
    void combine_outputs(Tensor& expert_outputs, const RouterModule::RouterOutput& routing,
                         Tensor& combined, cudaStream_t stream);
};

// ============================================================================
// ExpertGroupModule Implementation
// ============================================================================

inline ExpertGroupModule::ExpertGroupModule(Config config) : mConfig(config) {
    ExpertModule::Config expert_config;
    expert_config.hidden_size = config.hidden_size;
    expert_config.intermediate_size = config.intermediate_size;
    expert_config.use_gated = config.use_gated;

    mExperts.reserve(config.num_experts);
    for (int i = 0; i < config.num_experts; ++i) {
        mExperts.emplace_back(expert_config);
    }
}

inline Tensor ExpertGroupModule::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input,
    const RouterModule::RouterOutput& routing, Activations& acts) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int E = mConfig.num_experts;
    const int capacity = (BT * mConfig.top_k * mConfig.capacity_factor) / E;

    // Step 1: Dispatch tokens to experts based on routing
    // dispatched_input[e] contains tokens routed to expert e
    dispatch_tokens(input, routing, acts.dispatched_input, ctx.stream);

    // Step 2: Run each expert on its assigned tokens
    acts.expert_acts.resize(E);
    for (int e = 0; e < E; ++e) {
        // Get slice of dispatched input for this expert
        Tensor expert_input = acts.dispatched_input;
        expert_input.Data = acts.dispatched_input.Data +
                            e * capacity * C * get_dtype_size(expert_input.DType);
        expert_input.Sizes[0] = capacity;

        // Get expert output slice
        Tensor expert_output = acts.expert_outputs;
        expert_output.Data = acts.expert_outputs.Data +
                             e * capacity * C * get_dtype_size(expert_output.DType);
        expert_output.Sizes[0] = capacity;

        // Run expert forward
        mExperts[e].forward(ctx, w.experts[e], expert_input, acts.expert_acts[e]);
    }

    // Step 3: Combine expert outputs weighted by routing
    combine_outputs(acts.expert_outputs, routing, acts.combined_output, ctx.stream);

    return acts.combined_output;
}

inline Tensor ExpertGroupModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    const RouterModule::RouterOutput& routing,
    Tensor& grad_output, Gradients& grads, bool accumulate) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int E = mConfig.num_experts;
    const int capacity = (BT * mConfig.top_k * mConfig.capacity_factor) / E;

    // Step 1: Backward through combine (scatter gradient to experts)
    Tensor d_expert_outputs;
    // combine_outputs_backward(grad_output, routing, d_expert_outputs, ctx.stream);

    // Step 2: Backward through each expert
    Tensor d_dispatched;
    d_dispatched.DType = grad_output.DType;
    grads.experts.resize(E);

    for (int e = 0; e < E; ++e) {
        Tensor d_expert_output = d_expert_outputs;
        d_expert_output.Data = d_expert_outputs.Data +
                               e * capacity * C * get_dtype_size(d_expert_output.DType);
        d_expert_output.Sizes[0] = capacity;

        Tensor d_expert_input = d_dispatched;
        d_expert_input.Data = d_dispatched.Data +
                              e * capacity * C * get_dtype_size(d_expert_input.DType);

        mExperts[e].backward(ctx, w.experts[e], acts.expert_acts[e],
                             d_expert_output, grads.experts[e], accumulate);
    }

    // Step 3: Backward through dispatch (gather gradient from experts)
    Tensor d_input;
    // dispatch_tokens_backward(d_dispatched, routing, d_input, ctx.stream);

    return d_input;
}

inline void ExpertGroupModule::dispatch_tokens(
    Tensor& input, const RouterModule::RouterOutput& routing,
    Tensor& dispatched, cudaStream_t stream) {

    // Scatter tokens to their assigned experts
    // For each expert e:
    //   dispatched[e, :, :] = input[routing.token_indices[e], :]
    //
    // This requires a scatter operation that:
    // 1. Iterates over token_indices for each expert
    // 2. Copies tokens to the dispatched buffer
    // 3. Handles capacity overflow (drop tokens if needed)

    // Placeholder - requires specialized CUDA kernel
}

inline void ExpertGroupModule::combine_outputs(
    Tensor& expert_outputs, const RouterModule::RouterOutput& routing,
    Tensor& combined, cudaStream_t stream) {

    // Gather and combine expert outputs
    // For each token t:
    //   combined[t] = sum_k(routing_weights[t, k] * expert_outputs[expert_indices[t, k], position[t, k]])
    //
    // This requires a weighted gather operation

    // Placeholder - requires specialized CUDA kernel
}

/**
 * @brief Shared Expert - expert that processes all tokens (Nemotron/DeepSeek style)
 *
 * In some MoE architectures, a "shared expert" processes all tokens
 * in addition to the routed experts. This helps with representation quality.
 */
class SharedExpertModule : public ExpertModule {
public:
    struct Config : public ExpertModule::Config {
        float shared_expert_scale = 1.0f;  ///< Scale factor for shared expert output
    };

    explicit SharedExpertModule(Config config)
        : ExpertModule(static_cast<ExpertModule::Config&>(config))
        , mSharedConfig(config) {}

    /**
     * @brief Forward: run on all tokens (no routing)
     */
    Tensor forward_all(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        Tensor output = forward_impl(ctx, w, input, acts);

        // Scale output if configured
        if (mSharedConfig.shared_expert_scale != 1.0f) {
            // scale_tensor(output, mSharedConfig.shared_expert_scale, ctx.stream);
        }

        return output;
    }

private:
    Config mSharedConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_EXPERT_H
