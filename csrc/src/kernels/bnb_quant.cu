// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BitsAndBytes-style NF4 quantization kernels for QLoRA.
// Based on bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes)

/**
 * @file bnb_quant.cu
 * @brief CUDA kernels for BitsAndBytes-style NF4 blockwise quantization/dequantization.
 *
 * Provides GPU-accelerated blockwise NF4 quantization for memory-efficient QLoRA training:
 * - NF4 (Normal Float 4-bit) quantization with per-block absmax scaling
 * - Double quantization support (quantize absmax values to INT8)
 * - Works on any CUDA GPU (no SM89+ or SM100+ requirement)
 *
 * NF4 uses 16 asymmetric bins derived from a standard normal distribution N(0,1),
 * which better represents the weight distribution of neural networks compared to
 * uniform or FP4 quantization.
 */

#include "kernel_utils.cuh"
#include "utilities/tensor.h"
#include "utilities/utils.h"

#include <cuda_bf16.h>
#include <cub/cub.cuh>

// ============================================================================
// NF4 Lookup Tables
// ============================================================================

/**
 * @brief NF4 dequantization lookup table.
 *
 * 16 values derived from the normal distribution N(0,1) where each bin
 * has equal probability mass. Values are normalized to [-1, 1] range.
 * This distribution better matches neural network weight distributions.
 */
__device__ __constant__ float d_nf4_dequant_lut[16] = {
    -1.0f,                 // 0b0000
    -0.6961928009986877f,  // 0b0001
    -0.5250730514526367f,  // 0b0010
    -0.39491748809814453f, // 0b0011
    -0.28444138169288635f, // 0b0100
    -0.18477343022823334f, // 0b0101
    -0.09105003625154495f, // 0b0110
    0.0f,                  // 0b0111
    0.07958029955625534f,  // 0b1000
    0.16093020141124725f,  // 0b1001
    0.24611230194568634f,  // 0b1010
    0.33791524171829224f,  // 0b1011
    0.44070982933044434f,  // 0b1100
    0.5626170039176941f,   // 0b1101
    0.7229568362236023f,   // 0b1110
    1.0f                   // 0b1111
};

// Host-side copy of NF4 lookup table for initialization
static const float h_nf4_dequant_lut[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// ============================================================================
// NF4 Quantization/Dequantization Device Functions
// ============================================================================

/**
 * @brief Quantize a normalized value [-1, 1] to 4-bit NF4.
 *
 * Uses a binary decision tree for O(log n) quantization, matching the
 * bitsandbytes implementation exactly. Thresholds are midpoints between
 * adjacent NF4 codebook values.
 *
 * @param x Input value normalized to [-1, 1] range
 * @return 4-bit quantized value (0-15)
 */
__device__ __forceinline__ unsigned char dQuantizeNF4(float x) {
    // Binary decision tree for NF4 quantization
    // Thresholds are midpoints between adjacent codebook values
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)         // 1
            if (x > 0.6427869200706482f)     // 11
                if (x > 0.8614784181118011f) // 111
                    return 0b1111;
                else
                    return 0b1110;
            else if (x > 0.5016634166240692f) // 110
                return 0b1101;
            else
                return 0b1100;
        else if (x > 0.2035212516784668f) // 10
            if (x > 0.2920137718319893f)  // 101
                return 0b1011;
            else
                return 0b1010;
        else if (x > 0.1202552504837513f) // 100
            return 0b1001;
        else
            return 0b1000;
    else if (x > -0.33967943489551544f)     // 0
        if (x > -0.13791173323988914f)      // 01
            if (x > -0.045525018125772476f) // 011
                return 0b0111;
            else
                return 0b0110;
        else if (x > -0.23460740596055984f) // 010
            return 0b0101;
        else
            return 0b0100;
    else if (x > -0.6106329262256622f) // 00
        if (x > -0.4599952697753906f)  // 001
            return 0b0011;
        else
            return 0b0010;
    else if (x > -0.8480964004993439f) // 000
        return 0b0001;
    else
        return 0b0000;
}

/**
 * @brief Dequantize a 4-bit NF4 value to float.
 *
 * Simple lookup table dequantization.
 *
 * @param val 4-bit NF4 value (0-15)
 * @return Dequantized float value in [-1, 1]
 */
__device__ __forceinline__ float dDequantizeNF4(unsigned char val) {
    return d_nf4_dequant_lut[val & 0x0F];
}

// ============================================================================
// NF4 Blockwise Quantization Kernel
// ============================================================================

/**
 * @brief Simple NF4 blockwise quantization kernel.
 *
 * Quantizes BF16 weights to 4-bit NF4 with per-block absmax scaling.
 * Each block of `block_size` consecutive elements shares one FP32 scale.
 * Uses a simple approach without CUB for predictable memory layout.
 *
 * @param[out] out Output packed 4-bit array (n/2 bytes)
 * @param[out] absmax Output absmax scales (n/block_size floats)
 * @param[in] in Input BF16 array (n elements)
 * @param blocksize Number of elements per quantization block
 * @param n Total number of elements
 */
__global__ void kQuantizeBnBNF4Simple(
    unsigned char* __restrict__ out,
    float* __restrict__ absmax,
    const nv_bfloat16* __restrict__ in,
    const int blocksize,
    const long n)
{
    // Each thread handles one packed byte (2 elements)
    const long packed_n = (n + 1) / 2;
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= packed_n) return;

    // Element indices for this packed byte
    const long elem_idx = idx * 2;

    // Which absmax block do these elements belong to?
    const int blocksize_shift = 31 - __clz(blocksize);
    const int absmax_idx = elem_idx >> blocksize_shift;

    // Load the absmax for this block (computed in first pass)
    const float local_absmax = __ldg(&absmax[absmax_idx]);
    const float inv_absmax = (local_absmax > 0.0f) ? (1.0f / local_absmax) : 0.0f;

    // Load two elements
    float v0 = (elem_idx < n) ? ((float)in[elem_idx]) * inv_absmax : 0.0f;
    float v1 = (elem_idx + 1 < n) ? ((float)in[elem_idx + 1]) * inv_absmax : 0.0f;

    // Pack: high nibble = first value, low nibble = second value
    out[idx] = (dQuantizeNF4(v0) << 4) | dQuantizeNF4(v1);
}

/**
 * @brief Compute absmax for each block.
 *
 * First pass: compute the maximum absolute value for each block of elements.
 *
 * @param[out] absmax Output absmax scales (n/blocksize floats)
 * @param[in] in Input BF16 array (n elements)
 * @param blocksize Number of elements per quantization block
 * @param n Total number of elements
 */
__global__ void kComputeAbsmax(
    float* __restrict__ absmax,
    const nv_bfloat16* __restrict__ in,
    const int blocksize,
    const long n)
{
    const long num_blocks = (n + blocksize - 1) / blocksize;
    const long block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= num_blocks) return;

    const long start = block_idx * blocksize;
    const long end = min(start + blocksize, n);

    float local_max = 0.0f;
    for (long i = start; i < end; i++) {
        local_max = fmaxf(local_max, fabsf((float)in[i]));
    }

    absmax[block_idx] = local_max;
}

// ============================================================================
// NF4 Blockwise Dequantization Kernel
// ============================================================================

/**
 * @brief Simple NF4 blockwise dequantization kernel.
 *
 * Dequantizes packed 4-bit NF4 data back to BF16 using per-block absmax scales.
 * Each thread processes one packed byte at a time.
 *
 * @param[out] out Output BF16 array (n elements)
 * @param[in] in Input packed 4-bit array (n/2 bytes)
 * @param[in] absmax Per-block absmax scales
 * @param blocksize Quantization block size in elements (for computing absmax index)
 * @param n Total number of output elements
 */
__global__ void kDequantizeBnBNF4Simple(
    nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ in,
    const float* __restrict__ absmax,
    const int blocksize,
    const long n)
{
    const long packed_n = (n + 1) / 2;  // Number of packed bytes
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= packed_n) return;

    // Load packed byte
    unsigned char packed = in[idx];

    // Element indices for this packed byte
    const long elem_idx = idx * 2;

    // Compute absmax index based on element index
    const int blocksize_shift = 31 - __clz(blocksize);
    const int absmax_idx = elem_idx >> blocksize_shift;
    const float local_abs_max = __ldg(&absmax[absmax_idx]);

    // High nibble (bits 4-7) is first value
    if (elem_idx < n) {
        out[elem_idx] = (nv_bfloat16)(dDequantizeNF4(packed >> 4) * local_abs_max);
    }
    // Low nibble (bits 0-3) is second value
    if (elem_idx + 1 < n) {
        out[elem_idx + 1] = (nv_bfloat16)(dDequantizeNF4(packed & 0x0F) * local_abs_max);
    }
}

// ============================================================================
// Double Quantization Kernels (for absmax compression)
// ============================================================================

/**
 * @brief Quantize absmax values to INT8 for double quantization.
 *
 * Double quantization reduces memory overhead by quantizing the absmax
 * scaling factors themselves. This kernel:
 * 1. Groups absmax values into blocks of 256
 * 2. Computes per-group offset (mean) and scale (max after offset subtraction)
 * 3. Quantizes (absmax - offset) to INT8 using the scale
 *
 * @param[out] out_absmax_quant Output INT8 quantized absmax values
 * @param[out] out_absmax_scale Per-group FP32 scale for INT8 dequantization
 * @param[out] out_absmax_offset Per-group FP32 offset (subtracted before quantization)
 * @param[in] absmax Input FP32 absmax values
 * @param n Number of absmax values
 * @param group_size Number of absmax values per quantization group (default 256)
 */
__global__ void kQuantizeAbsmaxDouble(
    unsigned char* __restrict__ out_absmax_quant,
    float* __restrict__ out_absmax_scale,
    float* __restrict__ out_absmax_offset,
    const float* __restrict__ absmax,
    const int n,
    const int group_size = 256)
{
    const int group_idx = blockIdx.x;
    const int base_idx = group_idx * group_size;
    const int end_idx = min(base_idx + group_size, n);
    const int valid_items = end_idx - base_idx;

    // Shared memory for reduction
    __shared__ float s_sum;
    __shared__ float s_absmax;

    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        s_absmax = 0.0f;
    }
    __syncthreads();

    // Phase 1: Compute sum for offset (mean)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < valid_items; i += blockDim.x) {
        thread_sum += absmax[base_idx + i];
    }

    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&s_sum, thread_sum);
    }
    __syncthreads();

    // Compute offset (mean)
    float offset = s_sum / (float)valid_items;

    // Phase 2: Compute absmax of (value - offset)
    float thread_max = 0.0f;
    for (int i = threadIdx.x; i < valid_items; i += blockDim.x) {
        float val = absmax[base_idx + i] - offset;
        thread_max = fmaxf(thread_max, fabsf(val));
    }

    // Warp reduction for max
    for (int off = 16; off > 0; off >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, off));
    }
    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(&s_absmax), __float_as_uint(thread_max));
    }
    __syncthreads();

    float scale = s_absmax;
    float inv_scale = (scale > 0.0f) ? (127.0f / scale) : 0.0f;

    // Thread 0 stores the scale and offset
    if (threadIdx.x == 0) {
        out_absmax_scale[group_idx] = scale / 127.0f;  // Store dequant scale
        out_absmax_offset[group_idx] = offset;
    }

    // Phase 3: Quantize to INT8
    for (int i = threadIdx.x; i < valid_items; i += blockDim.x) {
        float val = absmax[base_idx + i] - offset;
        // Quantize to [-127, 127] range, stored as uint8 with offset 128
        int qval = __float2int_rn(val * inv_scale) + 128;
        qval = max(0, min(255, qval));
        out_absmax_quant[base_idx + i] = (unsigned char)qval;
    }
}

/**
 * @brief Dequantize INT8 absmax values back to FP32.
 *
 * @param[out] out_absmax Output FP32 absmax values
 * @param[in] in_absmax_quant Input INT8 quantized absmax values
 * @param[in] in_absmax_scale Per-group FP32 dequantization scale
 * @param[in] in_absmax_offset Per-group FP32 offset
 * @param n Number of absmax values
 * @param group_size Number of absmax values per group
 */
__global__ void kDequantizeAbsmaxDouble(
    float* __restrict__ out_absmax,
    const unsigned char* __restrict__ in_absmax_quant,
    const float* __restrict__ in_absmax_scale,
    const float* __restrict__ in_absmax_offset,
    const int n,
    const int group_size = 256)
{
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int group_idx = idx / group_size;
    const float scale = in_absmax_scale[group_idx];
    const float offset = in_absmax_offset[group_idx];

    // Dequantize: value = (qval - 128) * scale + offset
    float qval = (float)in_absmax_quant[idx] - 128.0f;
    out_absmax[idx] = qval * scale + offset;
}

// ============================================================================
// NF4 Dequantization with Double Quantization Support
// ============================================================================

/**
 * @brief Simple NF4 dequantization with inline absmax dequantization.
 *
 * When double quantization is used, this kernel handles both:
 * 1. Dequantizing INT8 absmax -> FP32 absmax
 * 2. Dequantizing NF4 data -> BF16 using the recovered absmax
 *
 * @param[out] out Output BF16 array (n elements)
 * @param[in] in Input packed 4-bit array (n/2 bytes)
 * @param[in] absmax_quant Quantized INT8 absmax values
 * @param[in] absmax_scale Per-group FP32 scale for absmax
 * @param[in] absmax_offset Per-group FP32 offset for absmax
 * @param blocksize Quantization block size in elements
 * @param absmax_group_size Group size for double quantization (typically 256)
 * @param n Total number of output elements
 */
__global__ void kDequantizeBnBNF4DoubleSimple(
    nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ in,
    const unsigned char* __restrict__ absmax_quant,
    const float* __restrict__ absmax_scale,
    const float* __restrict__ absmax_offset,
    const int blocksize,
    const int absmax_group_size,
    const long n)
{
    const long packed_n = (n + 1) / 2;  // Number of packed bytes
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= packed_n) return;

    // Load packed byte
    unsigned char packed = in[idx];

    // Element indices for this packed byte
    const long elem_idx = idx * 2;

    // Compute absmax index based on element index
    const int blocksize_shift = 31 - __clz(blocksize);
    const int absmax_idx = elem_idx >> blocksize_shift;

    // Dequantize absmax
    const int absmax_group = absmax_idx / absmax_group_size;
    const float scale = __ldg(&absmax_scale[absmax_group]);
    const float offset = __ldg(&absmax_offset[absmax_group]);
    const unsigned char qabsmax = __ldg(&absmax_quant[absmax_idx]);
    const float local_abs_max = ((float)qabsmax - 128.0f) * scale + offset;

    // High nibble (bits 4-7) is first value
    if (elem_idx < n) {
        out[elem_idx] = (nv_bfloat16)(dDequantizeNF4(packed >> 4) * local_abs_max);
    }
    // Low nibble (bits 0-3) is second value
    if (elem_idx + 1 < n) {
        out[elem_idx + 1] = (nv_bfloat16)(dDequantizeNF4(packed & 0x0F) * local_abs_max);
    }
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

/**
 * @brief Host launcher for NF4 blockwise quantization.
 *
 * Quantizes BF16 weights to packed 4-bit NF4 with per-block absmax scales.
 *
 * @param[out] out Output packed 4-bit array (M*K/2 bytes)
 * @param[out] absmax Output per-block absmax scales (M*K/blocksize floats)
 * @param[in] in Input BF16 array (M*K elements)
 * @param M Number of rows
 * @param K Number of columns
 * @param block_size Quantization block size (64, 128, 256, or 512)
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void quantize_bnb_nf4(
    unsigned char* out,
    float* absmax,
    const nv_bfloat16* in,
    int M, int K,
    int block_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const long n = (long)M * K;
    const long num_absmax_blocks = (n + block_size - 1) / block_size;
    const long packed_n = (n + 1) / 2;

    constexpr int THREADS = 256;

    // First pass: compute absmax for each block
    const int absmax_grid = (num_absmax_blocks + THREADS - 1) / THREADS;
    kComputeAbsmax<<<absmax_grid, THREADS, 0, stream>>>(absmax, in, block_size, n);
    CUDA_CHECK(cudaGetLastError());

    // Second pass: quantize using the computed absmax values
    const int quant_grid = (packed_n + THREADS - 1) / THREADS;
    kQuantizeBnBNF4Simple<<<quant_grid, THREADS, 0, stream>>>(out, absmax, in, block_size, n);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NF4 blockwise dequantization.
 *
 * Dequantizes packed 4-bit NF4 data back to BF16.
 *
 * @param[out] out Output BF16 array (M*K elements)
 * @param[in] in Input packed 4-bit array (M*K/2 bytes)
 * @param[in] absmax Per-block absmax scales
 * @param M Number of rows
 * @param K Number of columns
 * @param block_size Quantization block size
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void dequantize_bnb_nf4(
    nv_bfloat16* out,
    const unsigned char* in,
    const float* absmax,
    int M, int K,
    int block_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const long n = (long)M * K;
    const long packed_n = (n + 1) / 2;

    // Simple kernel: one thread per packed byte
    constexpr int THREADS = 256;
    const int num_blocks = (packed_n + THREADS - 1) / THREADS;

    kDequantizeBnBNF4Simple<<<num_blocks, THREADS, 0, stream>>>(
        out, in, absmax, block_size, n);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for double quantization of absmax values.
 *
 * Quantizes FP32 absmax values to INT8 for memory savings.
 *
 * @param[out] out_quant Output INT8 quantized absmax
 * @param[out] out_scale Per-group dequantization scale
 * @param[out] out_offset Per-group offset
 * @param[in] absmax Input FP32 absmax values
 * @param num_absmax Number of absmax values
 * @param group_size Values per quantization group (default 256)
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void quantize_absmax_double(
    unsigned char* out_quant,
    float* out_scale,
    float* out_offset,
    const float* absmax,
    int num_absmax,
    int group_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int num_groups = (num_absmax + group_size - 1) / group_size;
    const int threads = 256;

    kQuantizeAbsmaxDouble<<<num_groups, threads, 0, stream>>>(
        out_quant, out_scale, out_offset, absmax, num_absmax, group_size);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for dequantizing INT8 absmax back to FP32.
 *
 * @param[out] out_absmax Output FP32 absmax values
 * @param[in] in_quant Input INT8 quantized absmax
 * @param[in] in_scale Per-group dequantization scale
 * @param[in] in_offset Per-group offset
 * @param num_absmax Number of absmax values
 * @param group_size Values per quantization group
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void dequantize_absmax_double(
    float* out_absmax,
    const unsigned char* in_quant,
    const float* in_scale,
    const float* in_offset,
    int num_absmax,
    int group_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int threads = 256;
    const int num_blocks = (num_absmax + threads - 1) / threads;

    kDequantizeAbsmaxDouble<<<num_blocks, threads, 0, stream>>>(
        out_absmax, in_quant, in_scale, in_offset, num_absmax, group_size);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NF4 dequantization with double quantization.
 *
 * Dequantizes NF4 data when absmax values are also quantized (double quant).
 *
 * @param[out] out Output BF16 array (M*K elements)
 * @param[in] in Input packed 4-bit array (M*K/2 bytes)
 * @param[in] absmax_quant Quantized INT8 absmax values
 * @param[in] absmax_scale Per-group FP32 scale for absmax
 * @param[in] absmax_offset Per-group FP32 offset for absmax
 * @param M Number of rows
 * @param K Number of columns
 * @param block_size Quantization block size
 * @param absmax_group_size Group size for double quantization
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void dequantize_bnb_nf4_double(
    nv_bfloat16* out,
    const unsigned char* in,
    const unsigned char* absmax_quant,
    const float* absmax_scale,
    const float* absmax_offset,
    int M, int K,
    int block_size,
    int absmax_group_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const long n = (long)M * K;
    const long packed_n = (n + 1) / 2;

    // Simple kernel: one thread per packed byte
    constexpr int THREADS = 256;
    const int num_blocks = (packed_n + THREADS - 1) / THREADS;

    kDequantizeBnBNF4DoubleSimple<<<num_blocks, THREADS, 0, stream>>>(
        out, in, absmax_quant, absmax_scale, absmax_offset, block_size, absmax_group_size, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Tensor-based Wrapper Functions
// ============================================================================

/**
 * @brief Tensor-based wrapper for NF4 quantization.
 */
void quantize_bnb_nf4(
    Tensor& out,
    Tensor& absmax,
    const Tensor& in,
    int block_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_bnb_nf4: input must be BF16");
    }
    if (out.DType != ETensorDType::INT8) {
        throw std::runtime_error("quantize_bnb_nf4: output must be INT8 (packed NF4)");
    }
    if (absmax.DType != ETensorDType::FP32) {
        throw std::runtime_error("quantize_bnb_nf4: absmax must be FP32");
    }

    const int M = in.Sizes[0];
    const int K = (in.Rank == 2) ? in.Sizes[1] : 1;

    quantize_bnb_nf4(
        out.get<unsigned char>(),
        absmax.get<float>(),
        in.get<nv_bfloat16>(),
        M, K, block_size, dp, stream);
}

/**
 * @brief Tensor-based wrapper for NF4 dequantization.
 */
void dequantize_bnb_nf4(
    Tensor& out,
    const Tensor& in,
    const Tensor& absmax,
    int block_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (out.DType != ETensorDType::BF16) {
        throw std::runtime_error("dequantize_bnb_nf4: output must be BF16");
    }
    if (in.DType != ETensorDType::INT8) {
        throw std::runtime_error("dequantize_bnb_nf4: input must be INT8 (packed NF4)");
    }
    if (absmax.DType != ETensorDType::FP32) {
        throw std::runtime_error("dequantize_bnb_nf4: absmax must be FP32");
    }

    const int M = out.Sizes[0];
    const int K = (out.Rank == 2) ? out.Sizes[1] : 1;

    dequantize_bnb_nf4(
        out.get<nv_bfloat16>(),
        in.get<unsigned char>(),
        absmax.get<float>(),
        M, K, block_size, dp, stream);
}

/**
 * @brief Get the NF4 codebook values (host-side).
 *
 * Returns a pointer to the 16-element NF4 lookup table for host-side
 * operations or debugging.
 *
 * @return Pointer to static array of 16 floats
 */
const float* get_nf4_codebook() {
    return h_nf4_dequant_lut;
}
