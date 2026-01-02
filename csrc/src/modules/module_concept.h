// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODULE_CONCEPT_H
#define SUROGATE_SRC_MODULES_MODULE_CONCEPT_H

#include <concepts>
#include <cuda_runtime.h>

#include "utilities/tensor.h"
#include "utilities/dtype.h"

// Forward declarations
struct cudaDeviceProp;
typedef struct cudnnContext* cudnnHandle_t;
typedef struct cublasLtContext* cublasLtHandle_t;

namespace modules {

/**
 * @brief Runtime context passed to module operations
 *
 * Contains handles, workspace, stream, and configuration that modules need
 * but shouldn't own. Lightweight value semantics - passed by reference.
 *
 * This is the primary interface between the model infrastructure and
 * individual modules. Modules receive this context during forward/backward
 * and use it to access CUDA resources.
 */
struct ModuleContext {
    cudaStream_t stream;
    cublasLtHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    Tensor* workspace;           // Pointer to avoid copy, managed by caller
    const cudaDeviceProp* device_prop;

    // Shape information for dynamic sizing
    int B;   // Batch size
    int T;   // Sequence length

    // Position IDs for RoPE (may be null if not needed)
    const int* position_ids;

    // Quantization configuration
    ETensorDType matmul_dtype;   // Dtype for matmul operations
    bool use_quantization;       // Whether to quantize activations for matmul

    // Helper to get workspace as raw bytes
    std::byte* workspace_ptr() const {
        return workspace ? workspace->Data : nullptr;
    }

    std::size_t workspace_size() const {
        return workspace ? workspace->bytes() : 0;
    }
};

/**
 * @brief Concept for a primitive module (Linear, RMSNorm, etc.)
 *
 * Modules must provide:
 * - forward() for inference/training forward pass
 * - backward() for gradient computation
 * - Weight type alias (struct containing weight tensors)
 * - Gradients type alias (struct for weight gradients)
 * - Activations type alias (saved state for backward pass)
 *
 * The concept uses static polymorphism via CRTP for zero-overhead dispatch.
 * All operations take a ModuleContext& for CUDA resource access.
 *
 * Template parameters allow compile-time customization of module behavior
 * without runtime overhead.
 */
template<typename T>
concept PrimitiveModule = requires(T module,
                                    typename T::Weights& weights,
                                    typename T::Activations& acts,
                                    typename T::Gradients& grads,
                                    ModuleContext& ctx,
                                    Tensor& input,
                                    Tensor& grad_output) {
    // Type requirements
    typename T::Weights;
    typename T::Activations;
    typename T::Gradients;
    typename T::Config;

    // Forward pass: input -> output, saving activations for backward
    { module.forward(ctx, weights, input, acts) } -> std::same_as<Tensor>;

    // Backward pass: grad_output -> grad_input, computing weight gradients
    { module.backward(ctx, weights, acts, grad_output, grads) } -> std::same_as<Tensor>;
};

/**
 * @brief Concept for composite modules that can contain other modules
 *
 * Composite modules (e.g., TransformerBlock) compose multiple primitives
 * and may support recomputation during backward pass.
 */
template<typename T>
concept CompositeModule = PrimitiveModule<T> && requires(T module,
                                                          typename T::Weights& weights,
                                                          typename T::Activations& acts,
                                                          ModuleContext& ctx,
                                                          Tensor& input) {
    // Optional recomputation support
    { module.recompute(ctx, weights, input, acts) } -> std::same_as<void>;
};

/**
 * @brief Tensor that may have a quantized representation
 *
 * Used for activations that need to be cached for backward pass.
 * If quantization is enabled, the Quant field holds the quantized
 * version with associated scale (stored in Quant->Stats).
 */
struct QuantizableTensor {
    Tensor Value;                       // Original high-precision value
    std::optional<Tensor> Quant;        // Optional quantized representation

    // Get the tensor to use for matmul (quantized if available)
    const Tensor& for_matmul() const {
        return Quant.has_value() ? *Quant : Value;
    }

    // Non-const version for matmul
    Tensor& for_matmul() {
        return Quant.has_value() ? *Quant : Value;
    }

    // Get scale pointer (null if not quantized)
    // Note: Tensor::scale() is non-const, so we need non-const here
    float* scale() {
        return Quant.has_value() ? Quant->scale() : nullptr;
    }

    // Get abs_max pointer (null if not quantized)
    float* abs_max() {
        return Quant.has_value() ? Quant->abs_max() : nullptr;
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODULE_CONCEPT_H
