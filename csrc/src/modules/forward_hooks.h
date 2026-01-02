// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_FORWARD_HOOKS_H
#define SUROGATE_SRC_MODULES_FORWARD_HOOKS_H

#include <functional>

#include <cuda_runtime.h>

namespace modules {

/**
 * @brief Hook points during the forward pass
 *
 * These correspond to specific locations in the transformer block forward
 * where additional computation can be injected (e.g., applying LoRA deltas).
 */
enum class ForwardHookPoint {
    AfterQKVProjection,      ///< After QKV matmul, before RoPE
    AfterAttnOutProjection,  ///< After attention out matmul, before residual+LN2
    AfterMLPUpProjection,    ///< After MLP up matmul, before SwiGLU
    AfterMLPDownProjection,  ///< After MLP down matmul
};

constexpr const char* hook_point_name(ForwardHookPoint point) {
    switch (point) {
        case ForwardHookPoint::AfterQKVProjection: return "AfterQKVProjection";
        case ForwardHookPoint::AfterAttnOutProjection: return "AfterAttnOutProjection";
        case ForwardHookPoint::AfterMLPUpProjection: return "AfterMLPUpProjection";
        case ForwardHookPoint::AfterMLPDownProjection: return "AfterMLPDownProjection";
        default: return "Unknown";
    }
}

using ForwardHook = std::function<void(int layer_idx, ForwardHookPoint point, cudaStream_t stream)>;

} // namespace modules

#endif // SUROGATE_SRC_MODULES_FORWARD_HOOKS_H

