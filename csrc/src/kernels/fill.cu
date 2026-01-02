// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file fill.cu
 * @brief CUDA kernels for filling arrays with constant values.
 *
 * Provides GPU-accelerated memory initialization with a specified value.
 */

#include <cassert>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

/**
 * @brief CUDA kernel for filling an array with a constant value.
 *
 * Each thread writes one element. Simple implementation without vectorization.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] dst Destination array to fill.
 * @param value Constant value to write to all elements.
 * @param count Number of elements to fill.
 *
 * @note TODO: This kernel could be optimized with vectorized stores.
 */
template<typename floatX>
__global__ void fill_kernel(floatX* dst, floatX value, std::size_t count) {
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count) return;
    // TODO vectorize
    dst[id] = value;
}

/**
 * @brief Template launcher for the fill kernel.
 *
 * Launches the fill_kernel with 256 threads per block.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] dst Destination array to fill.
 * @param value Constant value to write.
 * @param count Number of elements to fill.
 * @param stream CUDA stream for asynchronous execution.
 */
template<typename floatX>
void fill_imp(floatX* dst, floatX value, std::size_t count, cudaStream_t stream) {
    fill_kernel<<<div_ceil(count, static_cast<std::size_t>(256)), 256, 0, stream>>> (dst, value, count);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fills an FP32 array with a constant value.
 *
 * @param[out] dst Destination array in FP32.
 * @param value Constant value to write to all elements.
 * @param count Number of elements to fill.
 * @param stream CUDA stream for asynchronous execution.
 */
void fill_constant(float* dst, float value, std::size_t count, cudaStream_t stream) {
    fill_imp(dst, value, count, stream);
}

/**
 * @brief Fills a BF16 array with a constant value.
 *
 * @param[out] dst Destination array in BF16.
 * @param value Constant value to write to all elements.
 * @param count Number of elements to fill.
 * @param stream CUDA stream for asynchronous execution.
 */
void fill_constant(nv_bfloat16* dst, nv_bfloat16 value, std::size_t count, cudaStream_t stream) {
    fill_imp(dst, value, count, stream);
}
