// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_NVFP4_SCALED_SWIGLU_H
#define SUROGATE_SRC_RECIPES_NVFP4_SCALED_SWIGLU_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace recipes::nvfp4 {

/**
 * @brief Scaled SwiGLU forward pass for FP4 training stability.
 *
 * Computes:
 *   s = max(|gate|, dim=-1)     # per-row max of gate
 *   gate_norm = gate / s        # normalize gate to [-1, 1] range
 *   out = silu(up) * gate_norm  # scaled output
 *
 * The scale `s` must be saved and applied to down_proj output later:
 *   final_out = down_proj(out) * s
 *
 * This keeps intermediate values in FP4-friendly range.
 *
 * @param[out] out Output tensor of shape (B, T, C)
 * @param[out] scale Per-row scale tensor of shape (B*T,) - save for backward and down_proj
 * @param[in] inp Input tensor of shape (B, T, 2*C) with [up, gate] concatenated
 * @param B Batch size
 * @param T Sequence length
 * @param C Hidden dimension (output width, half of input's last dim)
 * @param stream CUDA stream
 */
void scaled_swiglu_forward(
    nv_bfloat16* out,
    float* scale,
    const nv_bfloat16* inp,
    int B, int T, int C,
    cudaStream_t stream);

/**
 * @brief Scaled SwiGLU backward pass.
 *
 * Given forward: out = silu(up) * (gate / s)
 *
 * Computes gradients:
 *   d_up = dout * (gate / s) * silu'(up)
 *   d_gate = dout * silu(up) / s
 *
 * Note: The scale s is treated as a constant (detached) in the backward pass,
 * matching the study/ folder implementation.
 *
 * @param[out] dinp Output gradients of shape (B, T, 2*C) with [d_up, d_gate]
 * @param[in] dout Upstream gradient of shape (B, T, C)
 * @param[in] inp Original input tensor of shape (B, T, 2*C)
 * @param[in] scale Per-row scale from forward pass of shape (B*T,)
 * @param B Batch size
 * @param T Sequence length
 * @param C Hidden dimension
 * @param stream CUDA stream
 */
void scaled_swiglu_backward(
    nv_bfloat16* dinp,
    const nv_bfloat16* dout,
    const nv_bfloat16* inp,
    const float* scale,
    int B, int T, int C,
    cudaStream_t stream);

/**
 * @brief Apply per-row scale to a tensor (in-place).
 *
 * Used to apply the saved scale from scaled_swiglu_forward to the down_proj output:
 *   data[bt, c] *= scale[bt]
 *
 * @param[in,out] data Tensor of shape (B, T, C) to scale in-place
 * @param[in] scale Per-row scale factors of shape (B*T,)
 * @param B Batch size
 * @param T Sequence length
 * @param C Hidden dimension
 * @param stream CUDA stream
 */
void scale_rows(
    nv_bfloat16* data,
    const float* scale,
    int B, int T, int C,
    cudaStream_t stream);

}  // namespace recipes::nvfp4

#endif  // SUROGATE_SRC_RECIPES_NVFP4_SCALED_SWIGLU_H
