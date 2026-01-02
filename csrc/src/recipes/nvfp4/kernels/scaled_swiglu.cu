// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file scaled_swiglu.cu
 * @brief Scaled SwiGLU activation for FP4 training stability.
 *
 * Implements the scaled SwiGLU activation from the study/ folder experiments:
 *   s = max(|gate|, dim=-1)     # per-row max of gate
 *   gate_norm = gate / s        # normalize gate to [-1, 1] range
 *   out = silu(up) * gate_norm  # scaled output
 *
 * The scale `s` is returned separately and must be applied to down_proj output.
 * This keeps intermediate values in FP4-friendly range for better numerical stability.
 *
 * Input: [up, gate] concatenated of shape (B, T, 2*C)
 * Output: out of shape (B, T, C), scale of shape (B, T, 1)
 */

#include <cassert>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "scaled_swiglu.h"
#include "utilities/utils.h"

namespace recipes::nvfp4 {

// ============================================================================
// Forward kernels
// ============================================================================

/**
 * @brief Compute per-row max of gate values.
 *
 * First pass: computes s = max(|gate[bt, :]|) for each row.
 * Uses warp shuffle reduction for efficiency.
 *
 * @param[out] scale Output scale tensor of shape (BT,)
 * @param[in] inp Input tensor of shape (BT, 2*C) with [up, gate] concatenated
 * @param BT Batch size * sequence length
 * @param C Hidden dimension (half of input width)
 */
__global__ void scaled_swiglu_compute_scale_kernel(
    float* scale,
    const nv_bfloat16* inp,
    int BT,
    int C)
{
    const int bt = blockIdx.x;
    if (bt >= BT) return;

    const nv_bfloat16* gate = inp + bt * 2 * C + C;  // gate is second half

    // Each thread processes multiple elements
    float thread_max = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = fabsf(__bfloat162float(gate[c]));
        thread_max = fmaxf(thread_max, val);
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    // Block reduction using shared memory
    __shared__ float warp_maxes[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        thread_max = (lane < num_warps) ? warp_maxes[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (lane == 0) {
            // Clamp to avoid division by zero
            scale[bt] = fmaxf(thread_max, 1e-12f);
        }
    }
}

/**
 * @brief Apply scaled SwiGLU using precomputed scales.
 *
 * Second pass: computes out = silu(up) * (gate / scale)
 *
 * @param[out] out Output tensor of shape (BT, C)
 * @param[in] inp Input tensor of shape (BT, 2*C) with [up, gate] concatenated
 * @param[in] scale Per-row scale tensor of shape (BT,)
 * @param BT Batch size * sequence length
 * @param C Hidden dimension
 */
__global__ void scaled_swiglu_apply_kernel(
    nv_bfloat16* out,
    const nv_bfloat16* inp,
    const float* scale,
    int BT,
    int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bt = idx / C;
    const int c = idx % C;

    if (bt >= BT) return;

    const nv_bfloat16* up = inp + bt * 2 * C;
    const nv_bfloat16* gate = up + C;

    float up_val = __bfloat162float(up[c]);
    float gate_val = __bfloat162float(gate[c]);
    float s = scale[bt];

    // silu(up) * (gate / s)
    float silu_up = up_val / (1.0f + expf(-up_val));
    float gate_norm = gate_val / s;
    float result = silu_up * gate_norm;

    out[bt * C + c] = __float2bfloat16(result);
}

/**
 * @brief Fused scaled SwiGLU forward kernel (single pass, less efficient for large C).
 *
 * Computes scale and applies in one kernel. Better for small C where launching
 * two kernels has overhead.
 *
 * @param[out] out Output tensor of shape (BT, C)
 * @param[out] scale Per-row scale tensor of shape (BT,)
 * @param[in] inp Input tensor of shape (BT, 2*C)
 * @param BT Batch size * sequence length
 * @param C Hidden dimension
 */
__global__ void scaled_swiglu_fused_kernel(
    nv_bfloat16* out,
    float* scale,
    const nv_bfloat16* inp,
    int BT,
    int C)
{
    const int bt = blockIdx.x;
    if (bt >= BT) return;

    const nv_bfloat16* up = inp + bt * 2 * C;
    const nv_bfloat16* gate = up + C;
    nv_bfloat16* out_row = out + bt * C;

    // Step 1: Compute max(|gate|) using shared memory
    extern __shared__ float smem[];
    float* shared_max = smem;

    float thread_max = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = fabsf(__bfloat162float(gate[c]));
        thread_max = fmaxf(thread_max, val);
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Final reduction
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        thread_max = (lane < num_warps) ? shared_max[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (lane == 0) {
            shared_max[0] = fmaxf(thread_max, 1e-12f);
        }
    }
    __syncthreads();

    float s = shared_max[0];

    // Store scale
    if (threadIdx.x == 0) {
        scale[bt] = s;
    }

    // Step 2: Apply scaled swiglu
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float up_val = __bfloat162float(up[c]);
        float gate_val = __bfloat162float(gate[c]);

        float silu_up = up_val / (1.0f + expf(-up_val));
        float gate_norm = gate_val / s;
        float result = silu_up * gate_norm;

        out_row[c] = __float2bfloat16(result);
    }
}

// ============================================================================
// Backward kernels
// ============================================================================

/**
 * @brief Scaled SwiGLU backward pass.
 *
 * Given: out = silu(up) * (gate / s)
 * where s = max(|gate|) (treated as constant in backward)
 *
 * Gradients:
 *   d_up = dout * (gate / s) * silu'(up)
 *        = dout * (gate / s) * sigmoid(up) * (1 + up * (1 - sigmoid(up)))
 *   d_gate = dout * silu(up) / s
 *
 * @param[out] dinp Output gradients of shape (BT, 2*C) with [d_up, d_gate]
 * @param[in] dout Upstream gradient of shape (BT, C)
 * @param[in] inp Original input of shape (BT, 2*C)
 * @param[in] scale Per-row scale from forward pass of shape (BT,)
 * @param BT Batch size * sequence length
 * @param C Hidden dimension
 */
__global__ void scaled_swiglu_backward_kernel(
    nv_bfloat16* dinp,
    const nv_bfloat16* dout,
    const nv_bfloat16* inp,
    const float* scale,
    int BT,
    int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bt = idx / C;
    const int c = idx % C;

    if (bt >= BT) return;

    const nv_bfloat16* up = inp + bt * 2 * C;
    const nv_bfloat16* gate = up + C;
    nv_bfloat16* d_up = dinp + bt * 2 * C;
    nv_bfloat16* d_gate = d_up + C;

    float up_val = __bfloat162float(up[c]);
    float gate_val = __bfloat162float(gate[c]);
    float dout_val = __bfloat162float(dout[bt * C + c]);
    float s = scale[bt];

    // sigmoid(up)
    float sig_up = 1.0f / (1.0f + expf(-up_val));
    // silu(up) = up * sigmoid(up)
    float silu_up = up_val * sig_up;
    // silu'(up) = sigmoid(up) * (1 + up * (1 - sigmoid(up)))
    float silu_prime = sig_up * (1.0f + up_val * (1.0f - sig_up));

    // gate_norm = gate / s
    float gate_norm = gate_val / s;

    // d_up = dout * gate_norm * silu'(up)
    float d_up_val = dout_val * gate_norm * silu_prime;
    // d_gate = dout * silu(up) / s
    float d_gate_val = dout_val * silu_up / s;

    d_up[c] = __float2bfloat16(d_up_val);
    d_gate[c] = __float2bfloat16(d_gate_val);
}

// ============================================================================
// Host API
// ============================================================================

void scaled_swiglu_forward(
    nv_bfloat16* out,
    float* scale,
    const nv_bfloat16* inp,
    int B, int T, int C,
    cudaStream_t stream)
{
    const int BT = B * T;

    // Use fused kernel for smaller C, two-pass for larger C
    if (C <= 2048) {
        const int block_size = 256;
        const int smem_size = ((block_size + 31) / 32) * sizeof(float);
        scaled_swiglu_fused_kernel<<<BT, block_size, smem_size, stream>>>(
            out, scale, inp, BT, C);
    } else {
        // Two-pass approach
        const int block_size = 256;

        // Pass 1: Compute scales
        scaled_swiglu_compute_scale_kernel<<<BT, block_size, 0, stream>>>(
            scale, inp, BT, C);

        // Pass 2: Apply scaled swiglu
        const int total_elements = BT * C;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        scaled_swiglu_apply_kernel<<<grid_size, block_size, 0, stream>>>(
            out, inp, scale, BT, C);
    }
    CUDA_CHECK(cudaGetLastError());
}

void scaled_swiglu_backward(
    nv_bfloat16* dinp,
    const nv_bfloat16* dout,
    const nv_bfloat16* inp,
    const float* scale,
    int B, int T, int C,
    cudaStream_t stream)
{
    const int BT = B * T;
    const int total_elements = BT * C;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    scaled_swiglu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        dinp, dout, inp, scale, BT, C);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Scale rows kernel (apply per-row scale after down_proj)
// ============================================================================

/**
 * @brief Multiply each row of a tensor by a per-row scale factor.
 *
 * Used to apply the scale saved from scaled_swiglu_forward to the down_proj output:
 *   output[bt, c] *= scale[bt]
 *
 * @param[in,out] data Tensor of shape (BT, C) to scale in-place
 * @param[in] scale Per-row scale factors of shape (BT,)
 * @param BT Number of rows (B * T)
 * @param C Number of columns
 */
__global__ void scale_rows_kernel(
    nv_bfloat16* data,
    const float* scale,
    int BT,
    int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bt = idx / C;
    const int c = idx % C;

    if (bt >= BT) return;

    float val = __bfloat162float(data[bt * C + c]);
    float s = scale[bt];
    data[bt * C + c] = __float2bfloat16(val * s);
}

void scale_rows(
    nv_bfloat16* data,
    const float* scale,
    int B, int T, int C,
    cudaStream_t stream)
{
    const int BT = B * T;
    const int total_elements = BT * C;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    scale_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        data, scale, BT, C);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace recipes::nvfp4
