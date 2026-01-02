// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_SWIGLU_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_SWIGLU_H

#include "modules/module_base.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief SwiGLU activation module: y = swish(gate) * up
 *
 * SwiGLU (Swish-Gated Linear Unit) is the activation function used in
 * LLaMA/Qwen style transformers. It takes a fused [gate, up] input and
 * computes element-wise: swish(gate) * up
 *
 * Input shape: (B*T, 2 * intermediate_size) - concatenation of gate and up
 * Output shape: (B*T, intermediate_size)
 *
 * The swish function is: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * This is a parameter-free activation (no learnable weights).
 */
class SwiGLUModule : public ModuleBase<SwiGLUModule> {
public:
    /**
     * @brief Configuration for SwiGLU
     */
    struct Config {
        int intermediate_size;      ///< Output dimension (input is 2x this)
    };

    /**
     * @brief No learnable weights for SwiGLU
     */
    struct Weights {};

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        Tensor input_cached;        ///< Fused [gate, up] input for backward
        QuantizableTensor output;   ///< Output (may be quantized for subsequent matmul)
    };

    /**
     * @brief No weight gradients for parameter-free activation
     */
    struct Gradients {};

    explicit SwiGLUModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: y = swish(gate) * up
     *
     * @param ctx Module context with CUDA resources
     * @param w Unused (no weights)
     * @param input Fused [gate, up] tensor (B*T, 2 * intermediate_size)
     * @param acts Activation storage for backward
     * @return Output tensor (B*T, intermediate_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass: compute gradient w.r.t. input
     *
     * @param ctx Module context with CUDA resources
     * @param w Unused (no weights)
     * @param acts Saved activations from forward
     * @param grad_output Gradient w.r.t. output (B*T, intermediate_size)
     * @param grads Unused (no weight gradients)
     * @param accumulate Unused (no accumulation needed for input gradient)
     * @return Gradient w.r.t. input (B*T, 2 * intermediate_size)
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int intermediate_size() const { return mConfig.intermediate_size; }

private:
    Config mConfig;
};

inline Tensor SwiGLUModule::forward_impl(ModuleContext& ctx, Weights&, Tensor& input, Activations& acts) {
    const int D = mConfig.intermediate_size;

    // Cache input for backward
    acts.input_cached = input;

    // Get abs_max pointer for output quantization
    float* abs_max_ptr = acts.output.Quant.has_value() ? acts.output.Quant->abs_max() : nullptr;

    swiglu_forward(
        acts.output.Value,
        input,
        abs_max_ptr,
        ctx.B, ctx.T, D,
        ctx.stream
    );

    return acts.output.Value;
}

inline Tensor SwiGLUModule::backward_impl(ModuleContext& ctx, Weights&, Activations& acts,
                                          Tensor& grad_output, Gradients&, bool) {
    const int D = mConfig.intermediate_size;

    // Allocate gradient input - caller provides the buffer
    Tensor grad_input;
    grad_input.DType = acts.input_cached.DType;
    // Caller must set grad_input.Data

    // Get abs_max for quantized gradient
    float* abs_max_ptr = nullptr;  // Could be set if quantizing d_input

    swiglu_backward(
        grad_input,
        grad_output,
        acts.input_cached,
        abs_max_ptr,
        ctx.B, ctx.T, D,
        ctx.stream
    );

    return grad_input;
}

/**
 * @brief GeGLU activation module: y = gelu(gate) * up
 *
 * GeGLU uses GELU instead of Swish. Used in some model variants.
 * Interface is identical to SwiGLU.
 *
 * Note: Not yet implemented - placeholder for future variants.
 */
class GeGLUModule : public ModuleBase<GeGLUModule> {
public:
    struct Config {
        int intermediate_size;
    };

    struct Weights {};
    struct Activations {
        Tensor input_cached;
        QuantizableTensor output;
    };
    struct Gradients {};

    explicit GeGLUModule(Config config) : mConfig(config) {}

    Tensor forward_impl(ModuleContext& ctx, Weights&, Tensor& input, Activations& acts);
    Tensor backward_impl(ModuleContext& ctx, Weights&, Activations& acts,
                         Tensor& grad_output, Gradients&, bool = false);

private:
    Config mConfig;
};

// GeGLU implementation would go here when kernel is available

} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_SWIGLU_H
