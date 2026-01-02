// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_ROUTER_H
#define SUROGATE_SRC_MODULES_MOE_ROUTER_H

#include "modules/module_base.h"
#include "modules/primitives/linear.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Router module for Mixture of Experts
 *
 * Computes routing probabilities for each token to each expert using a linear
 * projection followed by softmax. Supports top-k routing where only the top-k
 * experts are activated per token.
 *
 * The router also computes auxiliary losses for load balancing:
 * - Load balancing loss: encourages uniform expert utilization
 * - Router z-loss: regularizes logits to prevent routing collapse
 *
 * Input: (B, T, hidden_size) token representations
 * Output: RouterOutput containing routing weights, indices, and auxiliary loss
 */
class RouterModule : public ModuleBase<RouterModule> {
public:
    /**
     * @brief Configuration for the router
     */
    struct Config {
        int hidden_size;            ///< Input hidden dimension
        int num_experts;            ///< Total number of experts
        int top_k = 2;              ///< Number of experts to route each token to
        float aux_loss_coef = 0.01f;  ///< Coefficient for load balancing auxiliary loss
        float z_loss_coef = 0.001f;   ///< Coefficient for router z-loss
        bool use_noisy_routing = false;  ///< Add noise during training for exploration
        float noise_std = 0.1f;     ///< Standard deviation of routing noise
        float capacity_factor = 1.25f;  ///< Expert capacity factor (tokens per expert)
        bool normalize_routing = true;  ///< Normalize routing weights to sum to 1
    };

    /**
     * @brief Weight tensors for router
     */
    struct Weights {
        Tensor gate;                ///< (hidden_size, num_experts) routing projection
    };

    /**
     * @brief Router output structure
     */
    struct RouterOutput {
        Tensor routing_weights;     ///< (B*T, top_k) normalized weights for selected experts
        Tensor expert_indices;      ///< (B*T, top_k) indices of selected experts (int32)
        Tensor expert_mask;         ///< (B*T, num_experts) binary mask of expert selection
        float aux_loss;             ///< Load balancing auxiliary loss
        float z_loss;               ///< Router z-loss for regularization

        // For expert dispatch
        Tensor token_indices;       ///< (num_experts, capacity) token indices per expert
        Tensor token_counts;        ///< (num_experts,) number of tokens per expert
        Tensor dispatch_mask;       ///< (B*T, num_experts) dispatch weights
        Tensor combine_weights;     ///< (B*T, num_experts) combine weights for output
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        Tensor input;               ///< (B*T, hidden_size) input to router
        Tensor logits;              ///< (B*T, num_experts) raw routing logits
        Tensor softmax_probs;       ///< (B*T, num_experts) softmax probabilities
        RouterOutput output;        ///< Full routing output
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_gate;              ///< (hidden_size, num_experts)
    };

    explicit RouterModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: compute routing decisions
     *
     * @param ctx Module context
     * @param w Router weights
     * @param input Hidden states (B*T, hidden_size)
     * @param acts Activation storage
     * @return RouterOutput with routing weights, indices, and auxiliary losses
     */
    RouterOutput forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass: compute router gradients
     *
     * @param ctx Module context
     * @param w Router weights
     * @param acts Saved activations
     * @param grad_routing_weights Gradient w.r.t. routing weights (B*T, top_k)
     * @param grads Gradient storage
     * @param accumulate If true, accumulate into existing gradients
     * @return Gradient w.r.t. input (B*T, hidden_size)
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_routing_weights, Gradients& grads, bool accumulate = false);

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int num_experts() const { return mConfig.num_experts; }
    [[nodiscard]] int top_k() const { return mConfig.top_k; }

private:
    Config mConfig;

    // Internal helpers
    void compute_aux_loss(Activations& acts, int B, int T);
    void top_k_routing(Tensor& probs, Tensor& weights, Tensor& indices, int BT, int k);
};

// ============================================================================
// Implementation
// ============================================================================

inline RouterModule::RouterOutput RouterModule::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int E = mConfig.num_experts;
    const int K = mConfig.top_k;

    // Save input for backward
    acts.input = input;

    // Compute routing logits: logits = input @ gate
    // gate is (hidden_size, num_experts), input is (BT, hidden_size)
    // logits is (BT, num_experts)
    matmul(
        acts.logits, w.gate, input, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        E, BT, C, EMMTranspose::TN, false,
        ctx.stream
    );

    // Add noise during training for exploration
    if (mConfig.use_noisy_routing && ctx.use_quantization) {  // use_quantization as training flag
        // Add Gaussian noise to logits
        // add_gaussian_noise(acts.logits, mConfig.noise_std, ctx.stream);
    }

    // Softmax to get routing probabilities
    // softmax(acts.softmax_probs, acts.logits, BT, E, ctx.stream);
    acts.softmax_probs = acts.logits;  // Placeholder - actual softmax needed

    // Top-k selection
    RouterOutput output;
    top_k_routing(acts.softmax_probs, output.routing_weights, output.expert_indices, BT, K);

    // Normalize routing weights if configured
    if (mConfig.normalize_routing) {
        // Normalize so selected weights sum to 1 per token
        // normalize_along_dim(output.routing_weights, 1, ctx.stream);
    }

    // Compute auxiliary losses
    compute_aux_loss(acts, ctx.B, ctx.T);
    output.aux_loss = acts.output.aux_loss;
    output.z_loss = acts.output.z_loss;

    // Compute dispatch/combine masks for expert execution
    // This involves:
    // 1. Creating per-expert token lists (respecting capacity)
    // 2. Computing dispatch mask for scattering tokens to experts
    // 3. Computing combine weights for gathering expert outputs

    acts.output = output;
    return output;
}

inline Tensor RouterModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_routing_weights, Gradients& grads, bool accumulate) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int E = mConfig.num_experts;

    // Gradient through top-k selection and softmax
    // d_logits = d_routing_weights * d_softmax / d_logits
    Tensor d_logits;
    d_logits.DType = acts.logits.DType;
    // softmax_backward(d_logits, grad_routing_weights, acts.softmax_probs, BT, E, ctx.stream);

    // Add auxiliary loss gradients
    // d_logits += aux_loss_coef * d_aux_loss / d_logits
    // d_logits += z_loss_coef * d_z_loss / d_logits

    // Gradient w.r.t. gate weights: d_gate = input^T @ d_logits
    matmul(
        grads.d_gate, acts.input, d_logits, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, E, BT, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    // Gradient w.r.t. input: d_input = d_logits @ gate^T
    Tensor d_input;
    d_input.DType = acts.input.DType;
    matmul(
        d_input, w.gate, d_logits, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, BT, E, EMMTranspose::NN, false,
        ctx.stream
    );

    return d_input;
}

inline void RouterModule::compute_aux_loss(Activations& acts, int B, int T) {
    const int BT = B * T;
    const int E = mConfig.num_experts;

    // Load balancing loss: encourages uniform expert utilization
    // aux_loss = E * sum_e(f_e * P_e)
    // where f_e = fraction of tokens routed to expert e
    //       P_e = mean routing probability to expert e

    // Router z-loss: prevents logits from growing too large
    // z_loss = (1/BT) * sum(log(sum(exp(logits))))^2

    acts.output.aux_loss = 0.0f;  // Placeholder
    acts.output.z_loss = 0.0f;    // Placeholder
}

inline void RouterModule::top_k_routing(Tensor& probs, Tensor& weights, Tensor& indices, int BT, int k) {
    // Select top-k experts for each token
    // This is typically done with a specialized CUDA kernel

    // For each token:
    // 1. Find the k experts with highest probability
    // 2. Store their indices in 'indices'
    // 3. Store their probabilities in 'weights'

    // Placeholder - actual implementation requires top-k kernel
}

/**
 * @brief Switch Router - simplified routing with top-1 selection
 *
 * A simplified router that routes each token to exactly one expert.
 * Used in Switch Transformer architecture.
 */
class SwitchRouterModule : public ModuleBase<SwitchRouterModule> {
public:
    struct Config {
        int hidden_size;
        int num_experts;
        int top_k = 1;              ///< Number of experts per token (typically 1 for Switch)
        float aux_loss_coef = 0.01f;  ///< Coefficient for load balancing auxiliary loss
        float capacity_factor = 1.0f;  ///< Expert capacity factor
        bool use_expert_choice = false;  ///< Expert chooses tokens instead of tokens choosing experts
    };

    // RouterOutput type for compatibility with MoETransformerBlock
    using RouterOutput = RouterModule::RouterOutput;

    struct Weights {
        Tensor gate;  ///< (hidden_size, num_experts)
    };

    struct Activations {
        Tensor input;
        Tensor logits;
        Tensor expert_index;     ///< (B*T,) single expert per token
        Tensor routing_weight;   ///< (B*T,) weight for selected expert
        RouterOutput output;     ///< Full routing output for compatibility
    };

    struct Gradients {
        Tensor d_gate;
    };

    explicit SwitchRouterModule(Config config) : mConfig(config) {}

    Activations forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

private:
    Config mConfig;
};

/**
 * @brief Expert Choice Router - experts select tokens
 *
 * Instead of tokens selecting experts (which can cause load imbalance),
 * each expert selects its top-k tokens. Guarantees perfect load balancing.
 * Used in some recent MoE architectures.
 */
class ExpertChoiceRouterModule : public ModuleBase<ExpertChoiceRouterModule> {
public:
    struct Config {
        int hidden_size;
        int num_experts;
        int tokens_per_expert;  ///< Fixed capacity per expert
        float aux_loss_coef = 0.0f;  ///< Usually 0 since load is balanced by design
    };

    struct Weights {
        Tensor gate;  ///< (hidden_size, num_experts)
    };

    struct Activations {
        Tensor input;
        Tensor logits;
        Tensor token_indices;    ///< (num_experts, tokens_per_expert) selected token indices
        Tensor routing_weights;  ///< (num_experts, tokens_per_expert) weights
    };

    struct Gradients {
        Tensor d_gate;
    };

    explicit ExpertChoiceRouterModule(Config config) : mConfig(config) {}

    Activations forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

private:
    Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_ROUTER_H
