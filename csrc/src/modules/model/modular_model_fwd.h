// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODEL_MODULAR_MODEL_FWD_H
#define SUROGATE_SRC_MODULES_MODEL_MODULAR_MODEL_FWD_H

#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../backward_hooks.h"
#include "../forward_hooks.h"
#include "../gradient_manager.h"
#include "../matmul_context.h"
#include "../model_config.h"
#include "../module_concept.h"
#include "../optimizer_state.h"
#include "../run_state.h"

#include "../weights/weight_manager_types.h"
#include "../weights/weight_manager.h"
#include "../weights/weight_manager_helpers.h"
#include "../weights/weight_manager_allocation.h"
#include "../weights/weight_manager_gather.h"
#include "../weights/weight_manager_optimizer.h"
#include "../weights/weight_manager_io.h"
#include "../weights/weight_manager_quantization.h"

#include "recipes/recipe.h"
#include "recipes/bf16/bf16_recipe.h"
#include "recipes/nvfp4/nvfp4_recipe.h"

#include "../optimizers/adamw_8bit.h"
#include "../optimizers/normuon.h"
#include "../optimizers/polar_express.h"
#include "../optimizers/optimizer_config.h"

#include "../primitives/attention.h"
#include "../primitives/embedding.h"
#include "../primitives/linear.h"
#include "../primitives/rmsnorm.h"
#include "../primitives/swiglu.h"
#include "../composite/transformer_block.h"

#include "../moe/moe_block.h"
#include "../moe/moe_types.h"

#include "kernels/kernels.h"
#include "training/runtime_options.h"
#include "training/model.h"
#include "training/dataloader.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/tensor_container.h"

namespace modules {

// Forward declarations
template<typename Block> class ModularWeightManager;
template<typename Block> class ModularGradientManager;
template<typename Block> class ModularRunState;

namespace detail {

/**
 * @brief Execute a callable either directly or under a CUDA graph capture/replay.
 *
 * Capture the callable on @p stream, then instantiate/update a cached cudaGraphExec and launch it.
 *
 * Note: If @p function depends on host-side decisions (e.g., optional hooks), the
 * caller must disable graphs in those cases to keep the graph topology stable.
 */
template<typename Function>
inline void trace_or_execute_cuda_graph(Function&& function, cudaStream_t stream,
                                        cudaGraphExec_t& instance, bool enabled) {
    if (!enabled) {
        function();
        return;
    }

    // Fast path: replay existing executable without re-capture.
    if (instance != nullptr) {
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        return;
    }

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
}

/**
 * @brief Execute a callable with stack checkpoint/restore for CUDA graph compatibility.
 *
 * When graphs are enabled and use temp_alloc() inside the captured function, we must
 * ensure the stack is in the same state before each graph replay. This overload:
 * - On first capture: saves a checkpoint after the function runs
 * - On replay: restores the stack to the checkpoint before launching the graph
 *
 * This ensures temp_alloc returns the same memory addresses that were captured in the graph.
 */
template<typename Function>
inline void trace_or_execute_cuda_graph_with_stack(Function&& function, cudaStream_t stream,
                                                    cudaGraphExec_t& instance, bool enabled,
                                                    DeviceMemoryStack& stack,
                                                    DeviceMemoryStack::Checkpoint& checkpoint) {
    if (!enabled) {
        function();
        return;
    }

    // Fast path: restore stack state and replay existing executable.
    if (instance != nullptr) {
        stack.restore(checkpoint);
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        return;
    }

    // Capture path: save checkpoint before capture so we know where to restore to.
    checkpoint = stack.checkpoint();

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
}

inline float* abs_max_ptr(Tensor& maybe_quant) {
    return maybe_quant.Data ? maybe_quant.abs_max() : nullptr;
}

/**
 * @brief Helper to execute recipe-driven forward matmul with full context setup
 *
 * This function handles the setup of MatmulContext for recipe->forward_matmul(),
 * including FP8 quant buffers and delayed scaling indices. Used by Phase 6
 * consolidation to reduce boilerplate in the forward loop.
 *
 * @param recipe The training recipe (must have handles_forward_matmul() == true)
 * @param out Output tensor
 * @param inp Input tensor
 * @param weight Weight tensor
 * @param bias Optional bias tensor (nullptr if no bias)
 * @param rs Run state
 * @param B Batch size
 * @param T Sequence length
 * @param C_in Input channels
 * @param C_out Output channels
 * @param layer_idx Current layer index
 * @param op Matmul operation type
 * @param inp_quant FP8 input quant buffer (for FP8 recipes)
 * @param cached_weight Optional cached FP8 weight
 * @param delayed_quantizer_idx Delayed scaling quantizer index (-1 for JIT)
 * @param stream CUDA stream
 * @param cached_fp4_data Optional cached FP4 packed weight data (CUTLASS layout)
 * @param cached_fp4_scales Optional cached FP4 block scales (FP8 E4M3, CUTLASS layout)
 * @param cached_fp4_amax Optional cached FP4 global amax pointer (device memory)
 */
template<typename Block>
inline void recipe_forward_matmul(
    const recipes::Recipe& recipe,
    Tensor& out, Tensor& inp, Tensor& weight, Tensor* bias,
    ModularRunState<Block>& rs,
    int B, int T, int C_in, int C_out,
    int layer_idx, MatmulOp op,
    Tensor* inp_quant,
    const Tensor* cached_weight,
    int delayed_quantizer_idx,
    cudaStream_t stream,
    const Tensor* cached_fp4_data = nullptr,
    const Tensor* cached_fp4_scales = nullptr,
    const float* cached_fp4_amax = nullptr,
    bool allow_quant = true)
{
    MatmulContext ctx;
    ctx.out = &out;
    ctx.inp = &inp;
    ctx.weight = &weight;
    ctx.bias = bias;
    ctx.B = B;
    ctx.T = T;
    ctx.C_in = C_in;
    ctx.C_out = C_out;
    ctx.run_state = &rs;
    ctx.stream = stream;
    ctx.layer_idx = layer_idx;
    ctx.op = op;
    ctx.inp_quant = inp_quant;
    ctx.cached_weight = allow_quant ? cached_weight : nullptr;
    ctx.delayed_quantizer_idx = allow_quant ? delayed_quantizer_idx : -1;
    ctx.cached_fp4_data = allow_quant ? cached_fp4_data : nullptr;
    ctx.cached_fp4_scales = allow_quant ? cached_fp4_scales : nullptr;
    ctx.cached_fp4_amax = allow_quant ? cached_fp4_amax : nullptr;
    ctx.allow_fp4 = allow_quant;
    ctx.allow_fp8 = allow_quant;

    recipe.forward_matmul(ctx);
}

/**
 * @brief Helper to execute recipe-driven backward matmul with full context setup
 *
 * @param recipe The training recipe
 * @param dinp Gradient w.r.t. input
 * @param dweight Gradient w.r.t. weight
 * @param dbias Optional gradient w.r.t. bias (nullptr if no bias)
 * @param dout Upstream gradient
 * @param inp Input activation from forward pass
 * @param weight Weight tensor
 * @param rs Run state
 * @param B Batch size
 * @param T Sequence length
 * @param C_in Input channels
 * @param C_out Output channels
 * @param layer_idx Current layer index
 * @param op Matmul operation type
 * @param accumulate Whether to accumulate into gradient buffers
 * @param skip_weight_grad Skip weight gradient (for LoRA-only)
 * @param inp_quant FP8 input quant buffer
 * @param dout_quant E5M2 gradient buffer
 * @param bias_buffer Scratch buffer for bias gradient
 * @param stream CUDA stream
 */
template<typename Block>
inline void recipe_backward_matmul(
    const recipes::Recipe& recipe,
    Tensor& dinp, Tensor& dweight, Tensor* dbias,
    Tensor& dout, Tensor& inp, Tensor& weight,
    ModularRunState<Block>& rs,
    int B, int T, int C_in, int C_out,
    int layer_idx, MatmulOp op,
    bool accumulate, bool skip_weight_grad,
    Tensor* inp_quant,
    Tensor* dout_quant,
    Tensor* bias_buffer,
    cudaStream_t stream)
{
    MatmulContext ctx;
    ctx.dinp = &dinp;
    ctx.dweight = &dweight;
    ctx.dbias = dbias;
    ctx.dout = &dout;
    ctx.inp = &inp;
    ctx.weight = &weight;
    ctx.B = B;
    ctx.T = T;
    ctx.C_in = C_in;
    ctx.C_out = C_out;
    ctx.run_state = &rs;
    ctx.stream = stream;
    ctx.layer_idx = layer_idx;
    ctx.op = op;
    ctx.accumulate = accumulate;
    ctx.skip_weight_grad = skip_weight_grad;
    ctx.inp_quant = inp_quant;
    ctx.dout_quant = dout_quant;
    ctx.bias_buffer = bias_buffer;

    recipe.backward_matmul(ctx);
}

} // namespace detail

/**
 * @brief Empty tensor container for unused ITensorContainer returns
 */
class EmptyTensorContainer : public ITensorContainer {
public:
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>&) override {}
};

namespace detail {

/**
 * @brief Container exposing 8-bit AdamW first moment (state1 + absmax1) for checkpointing
 */
class AdamW8BitMomentumContainer final : public ITensorContainer {
public:
    AdamW8BitMomentumContainer(Tensor* state1, Tensor* absmax1)
        : mState1(state1), mAbsmax1(absmax1) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState1 || !mState1->Data) return;
        callback("adamw8bit.state1", TensorShard(*mState1));
        if (mAbsmax1 && mAbsmax1->Data) {
            callback("adamw8bit.absmax1", TensorShard(*mAbsmax1));
        }
    }

    void update_pointers(Tensor* state1, Tensor* absmax1) {
        mState1 = state1;
        mAbsmax1 = absmax1;
    }

private:
    Tensor* mState1;
    Tensor* mAbsmax1;
};

/**
 * @brief Container exposing 8-bit AdamW second moment (state2 + absmax2) for checkpointing
 */
class AdamW8BitVarianceContainer final : public ITensorContainer {
public:
    AdamW8BitVarianceContainer(Tensor* state2, Tensor* absmax2)
        : mState2(state2), mAbsmax2(absmax2) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState2 || !mState2->Data) return;
        callback("adamw8bit.state2", TensorShard(*mState2));
        if (mAbsmax2 && mAbsmax2->Data) {
            callback("adamw8bit.absmax2", TensorShard(*mAbsmax2));
        }
    }

    void update_pointers(Tensor* state2, Tensor* absmax2) {
        mState2 = state2;
        mAbsmax2 = absmax2;
    }

private:
    Tensor* mState2;
    Tensor* mAbsmax2;
};

/**
 * @brief Container exposing NorMuon momentum state for checkpointing
 *
 * Includes: AdamW state for 1D params + NorMuon momentum for 2D params
 */
class NorMuonMomentumContainer final : public ITensorContainer {
public:
    NorMuonMomentumContainer(Tensor* adamw_state1, Tensor* adamw_absmax1,
                              Tensor* normuon_momentum, Tensor* normuon_absmax)
        : mAdamwState1(adamw_state1), mAdamwAbsmax1(adamw_absmax1),
          mNormuonMomentum(normuon_momentum), mNormuonAbsmax(normuon_absmax) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        // AdamW momentum for 1D parameters
        if (mAdamwState1 && mAdamwState1->Data) {
            callback("normuon.adamw_state1", TensorShard(*mAdamwState1));
            if (mAdamwAbsmax1 && mAdamwAbsmax1->Data) {
                callback("normuon.adamw_absmax1", TensorShard(*mAdamwAbsmax1));
            }
        }
        // NorMuon momentum for 2D parameters
        if (mNormuonMomentum && mNormuonMomentum->Data) {
            callback("normuon.momentum_state", TensorShard(*mNormuonMomentum));
            if (mNormuonAbsmax && mNormuonAbsmax->Data) {
                callback("normuon.momentum_absmax", TensorShard(*mNormuonAbsmax));
            }
        }
    }

    void update_pointers(Tensor* adamw_state1, Tensor* adamw_absmax1,
                        Tensor* normuon_momentum, Tensor* normuon_absmax) {
        mAdamwState1 = adamw_state1;
        mAdamwAbsmax1 = adamw_absmax1;
        mNormuonMomentum = normuon_momentum;
        mNormuonAbsmax = normuon_absmax;
    }

private:
    Tensor* mAdamwState1;
    Tensor* mAdamwAbsmax1;
    Tensor* mNormuonMomentum;
    Tensor* mNormuonAbsmax;
};

/**
 * @brief Container exposing NorMuon variance state for checkpointing
 *
 * Includes: AdamW variance for 1D params + NorMuon variance buffers for 2D params
 */
class NorMuonVarianceContainer final : public ITensorContainer {
public:
    NorMuonVarianceContainer(Tensor* adamw_state2, Tensor* adamw_absmax2,
                              std::vector<Tensor>* variance_buffers)
        : mAdamwState2(adamw_state2), mAdamwAbsmax2(adamw_absmax2),
          mVarianceBuffers(variance_buffers) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        // AdamW variance for 1D parameters
        if (mAdamwState2 && mAdamwState2->Data) {
            callback("normuon.adamw_state2", TensorShard(*mAdamwState2));
            if (mAdamwAbsmax2 && mAdamwAbsmax2->Data) {
                callback("normuon.adamw_absmax2", TensorShard(*mAdamwAbsmax2));
            }
        }
        // NorMuon variance buffers for 2D parameters (one per weight)
        if (mVarianceBuffers) {
            for (size_t i = 0; i < mVarianceBuffers->size(); ++i) {
                auto& buf = (*mVarianceBuffers)[i];
                if (buf.Data) {
                    callback(fmt::format("normuon.variance_{}", i), TensorShard(buf));
                }
            }
        }
    }

    void update_pointers(Tensor* adamw_state2, Tensor* adamw_absmax2,
                        std::vector<Tensor>* variance_buffers) {
        mAdamwState2 = adamw_state2;
        mAdamwAbsmax2 = adamw_absmax2;
        mVarianceBuffers = variance_buffers;
    }

private:
    Tensor* mAdamwState2;
    Tensor* mAdamwAbsmax2;
    std::vector<Tensor>* mVarianceBuffers;
};

} // namespace detail

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODEL_MODULAR_MODEL_FWD_H
