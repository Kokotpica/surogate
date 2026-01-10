// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_FORWARD_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_FORWARD_H

#include "lora_model_core.h"
#include "lora_model_utils.h"
#include "lora_utils.h"

namespace modules {

template<typename Block>
void ModularLoRAModel<Block>::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!lora_enabled()) {
        mBaseModel->forward(inputs, position_ids, comm, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    if (qlora_enabled() && micro_step == 0) {
        if (mFP8WeightProvider) mFP8WeightProvider->invalidate_cache();
        if (mFP4WeightProvider) mFP4WeightProvider->invalidate_cache();
        if (mBnBWeightProvider) mBnBWeightProvider->invalidate_cache();
    }

    auto hook = [this](int layer_idx, cudaStream_t stream, ForwardHookPoint point) {
        const auto& cfg = mBaseModel->config();
        auto& rs = mBaseModel->run_state();
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig.rank;
        const float scaling = mLoRAConfig.scaling();

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case ForwardHookPoint::AfterQKVProjection: {
                if (lora_block.attention.q.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
        }
    };

    mBaseModel->forward_with_hook(inputs, position_ids, comm, micro_step, hook);
}

template<typename Block>
float ModularLoRAModel<Block>::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!lora_enabled()) {
        return mBaseModel->validate(inputs, position_ids, targets, comm, micro_step);
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto full_hook = [this](int layer_idx, cudaStream_t stream, ForwardHookPoint point) {
        const auto& cfg = mBaseModel->config();
        auto& rs = mBaseModel->run_state();
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig.rank;
        const float scaling = mLoRAConfig.scaling();

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case ForwardHookPoint::AfterQKVProjection: {
                if (lora_block.attention.q.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
        }
    };

    return mBaseModel->validate_with_hook(inputs, position_ids, targets, comm, micro_step, full_hook);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_FORWARD_H
