// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_WEIGHTS_H
#define SUROGATE_SRC_MODULES_LORA_LORA_WEIGHTS_H

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "lora_config.h"
#include "modules/model_config.h"
#include "kernels/kernels.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"

namespace modules {

/**
 * @brief LoRA weights for a single linear layer: W' = W + scaling * B @ A
 *
 * A is (rank, in_features) - initialized with Kaiming uniform
 * B is (out_features, rank) - initialized with zeros
 */
template<typename TTensor>
struct LoRALayerWeights {
    TTensor A;  ///< (rank, in_features)
    TTensor B;  ///< (out_features, rank)

    [[nodiscard]] bool has_value() const { return A.Data != nullptr; }
};

/**
 * @brief LoRA weights for attention projections
 */
template<typename TTensor>
struct LoRAAttentionWeights {
    std::optional<LoRALayerWeights<TTensor>> q;  ///< Query projection
    std::optional<LoRALayerWeights<TTensor>> k;  ///< Key projection
    std::optional<LoRALayerWeights<TTensor>> v;  ///< Value projection
    std::optional<LoRALayerWeights<TTensor>> o;  ///< Output projection
};

/**
 * @brief LoRA weights for MLP projections
 */
template<typename TTensor>
struct LoRAMLPWeights {
    std::optional<LoRALayerWeights<TTensor>> gate;  ///< Gate projection
    std::optional<LoRALayerWeights<TTensor>> up;    ///< Up projection
    std::optional<LoRALayerWeights<TTensor>> down;  ///< Down projection
};

/**
 * @brief LoRA weights for a single MoE expert
 *
 * Each expert has its own independent LoRA adapters for gate, up, and down projections.
 * This enables per-expert fine-tuning in MoE models.
 */
template<typename TTensor>
struct LoRAExpertWeights {
    std::optional<LoRALayerWeights<TTensor>> gate;  ///< Gate projection LoRA
    std::optional<LoRALayerWeights<TTensor>> up;    ///< Up projection LoRA
    std::optional<LoRALayerWeights<TTensor>> down;  ///< Down projection LoRA

    [[nodiscard]] bool has_any() const {
        return (gate.has_value() && gate->has_value()) ||
               (up.has_value() && up->has_value()) ||
               (down.has_value() && down->has_value());
    }
};

/**
 * @brief LoRA weights for all experts in a MoE block
 *
 * Manages per-expert LoRA adapters for MoE transformer blocks.
 * Each expert can have independent LoRA weights.
 */
template<typename TTensor>
struct LoRAMoEWeights {
    std::vector<LoRAExpertWeights<TTensor>> experts;  ///< Per-expert LoRA weights

    [[nodiscard]] bool has_any() const {
        for (const auto& expert : experts) {
            if (expert.has_any()) return true;
        }
        return false;
    }

    [[nodiscard]] int num_experts() const {
        return static_cast<int>(experts.size());
    }
};

/**
 * @brief LoRA weights for a transformer block
 */
template<typename TTensor>
struct LoRABlockWeights {
    LoRAAttentionWeights<TTensor> attention;
    LoRAMLPWeights<TTensor> mlp;       ///< For dense models
    LoRAMoEWeights<TTensor> moe;       ///< For MoE models (per-expert LoRA)
};

/**
 * @brief Complete LoRA adapter weights
 */
template<typename TTensor>
struct LoRAWeightsSet {
    std::vector<LoRABlockWeights<TTensor>> blocks;
    ModularLoRAConfig config;
};

/**
 * @brief Modular LoRA weights manager
 *
 * Manages LoRA adapter weights for all layers, supporting:
 * - Sharded storage for multi-GPU
 * - Random initialization (Kaiming for A, zeros for B)
 * - Import/export from safetensors
 */
class ModularLoRAWeightsManager : public ITensorContainer {
public:
    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        ModularLoRAConfig lora_config;
        ETensorDType work_dtype = ETensorDType::BF16;  // compute dtype (typically base model dtype)
        int shard_idx = 0;
        int num_shards = 1;
        bool is_moe = false;  ///< True for MoE models

        // MoE-specific configuration (only used when is_moe = true)
        int num_experts = 0;              ///< Number of experts per layer
        int moe_intermediate_size = 0;    ///< Per-expert intermediate size (0 = use intermediate_size)

        [[nodiscard]] int effective_moe_intermediate() const {
            return moe_intermediate_size > 0 ? moe_intermediate_size : intermediate_size;
        }
    };

    ModularLoRAWeightsManager(const Config& config, TensorAllocator& allocator);
    ~ModularLoRAWeightsManager() = default;

    /**
     * @brief Initialize LoRA weights randomly
     *
     * A matrices: Kaiming uniform initialization
     * B matrices: Zeros (so initial output is zero)
     */
    void random_init(int seed, NCCLCommunicator& comm);

    /**
     * @brief Import LoRA adapter from safetensors file
     */
    void import_from_file(const std::string& file_name, NCCLCommunicator& comm);

    /**
     * @brief Export LoRA adapter to safetensors file
     */
    void export_to_file(const std::string& file_name, NCCLCommunicator& comm) const;

    /**
     * @brief Get block weights for forward/backward pass
     */
    LoRABlockWeights<Tensor>& get_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get master block weights for optimizer
     */
    LoRABlockWeights<TensorShard>& get_master_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get LoRA scaling factor
     */
    [[nodiscard]] float scaling() const { return mConfig.lora_config.scaling(); }

    /**
     * @brief Check if LoRA is enabled
     */
    [[nodiscard]] bool enabled() const { return mConfig.lora_config.enabled(); }

    /**
     * @brief Get the LoRA configuration
     */
    [[nodiscard]] const ModularLoRAConfig& lora_config() const { return mConfig.lora_config; }

    /**
     * @brief Get number of trainable parameters
     */
    [[nodiscard]] std::size_t num_parameters() const;

    // ITensorContainer interface
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

private:
    Config mConfig;
    TensorAllocator* mAllocator;

    // Master weights (sharded for multi-GPU)
    LoRAWeightsSet<TensorShard> mMaster;

    // Working weights (full precision for compute)
    LoRAWeightsSet<Tensor> mWork;

    void allocate_layer_weights(LoRALayerWeights<TensorShard>& shard,
                                 LoRALayerWeights<Tensor>& work,
                                 int in_features, int out_features,
                                 const std::string& name);
    void allocate_block_weights(int layer_idx);
    void allocate_expert_weights(LoRAExpertWeights<TensorShard>& master_expert,
                                  LoRAExpertWeights<Tensor>& work_expert,
                                  int layer_idx, int expert_idx);
};

/**
 * @brief Modular LoRA gradients manager
 *
 * Manages gradient storage for LoRA adapter training.
 */
class ModularLoRAGradsManager {
public:
    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        ModularLoRAConfig lora_config;
        ETensorDType grad_dtype;
        int shard_idx = 0;
        int num_shards = 1;
        bool is_moe = false;  ///< True for MoE models

        // MoE-specific configuration (only used when is_moe = true)
        int num_experts = 0;              ///< Number of experts per layer
        int moe_intermediate_size = 0;    ///< Per-expert intermediate size (0 = use intermediate_size)

        [[nodiscard]] int effective_moe_intermediate() const {
            return moe_intermediate_size > 0 ? moe_intermediate_size : intermediate_size;
        }
    };

    ModularLoRAGradsManager(const Config& config, const std::shared_ptr<TensorAllocator>& allocator);
    ~ModularLoRAGradsManager();

    /**
     * @brief Start a micro-step (for gradient accumulation)
     */
    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);

    /**
     * @brief End a micro-step
     */
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm);

    /**
     * @brief Get full gradients for backward pass
     */
    LoRABlockWeights<Tensor>& get_block_full(int layer_idx, cudaStream_t stream,
                                              NCCLCommunicator& comm, bool& accumulate);

    /**
     * @brief Get sharded gradients for optimizer
     */
    LoRABlockWeights<TensorShard>& get_block_shard(int layer_idx, cudaStream_t stream);

    /**
     * @brief Notify gradient computation complete for a block
     */
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm);

    [[nodiscard]] bool is_first_micro_step() const { return mIsFirstMicroStep; }
    [[nodiscard]] bool is_last_micro_step() const { return mIsLastMicroStep; }

private:
    Config mConfig;
    std::shared_ptr<TensorAllocator> mAllocator;

    // Full gradients (for backward computation)
    LoRAWeightsSet<Tensor> mFullGrads;

    // Sharded gradients (after all-reduce, for optimizer)
    LoRAWeightsSet<TensorShard> mShardedGrads;

    bool mIsFirstMicroStep = true;
    bool mIsLastMicroStep = false;

    void allocate_gradients();
    void reduce_gradients(cudaStream_t stream, NCCLCommunicator& comm);
};

/**
 * @brief Modular LoRA optimizer state manager
 *
 * Manages Adam optimizer state (m and v) for LoRA parameters.
 */
class ModularLoRAOptimizerState {
public:
    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        ModularLoRAConfig lora_config;
        ETensorDType m_dtype;
        ETensorDType v_dtype;
        int shard_idx = 0;
        int num_shards = 1;
        bool offload_m = false;
        bool offload_v = false;
        bool use_zero_copy = false;
        EAllocationType offload_alloc = EAllocationType::PINNED;
    };

    ModularLoRAOptimizerState(const Config& config, cudaStream_t stream,
                               NCCLCommunicator& comm, TensorAllocator& allocator);
    ~ModularLoRAOptimizerState();

    /**
     * @brief Get block momentum for optimizer
     */
    LoRABlockWeights<TensorShard>& get_block_m(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get block variance for optimizer
     */
    LoRABlockWeights<TensorShard>& get_block_v(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get FP8 scales for block momentum (m)
     */
    LoRABlockWeights<TensorShard>& get_block_m_scales(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get FP8 scales for block variance (v)
     */
    LoRABlockWeights<TensorShard>& get_block_v_scales(int layer_idx, cudaStream_t stream);

    /**
     * @brief Tensor containers for checkpointing (names match PEFT adapter tensors)
     */
    ITensorContainer& full_m();
    ITensorContainer& full_v();
    ITensorContainer& full_m_scales();
    ITensorContainer& full_v_scales();

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] Tensor& staging_m() { return mStagingM; }
    [[nodiscard]] Tensor& staging_v() { return mStagingV; }
    [[nodiscard]] Tensor& staging_m_scales() { return mStagingMScales; }
    [[nodiscard]] Tensor& staging_v_scales() { return mStagingVScales; }

private:
    Config mConfig;
    TensorAllocator* mAllocator;

    LoRAWeightsSet<TensorShard> mMomentum;   // First moment (m)
    LoRAWeightsSet<TensorShard> mVariance;   // Second moment (v)
    LoRAWeightsSet<TensorShard> mMomentumScales;   // FP8 scales for m (FP32)
    LoRAWeightsSet<TensorShard> mVarianceScales;   // FP8 scales for v (FP32)

    class StateContainer final : public ITensorContainer {
    public:
        explicit StateContainer(LoRAWeightsSet<TensorShard>* set) : mSet(set) {}
        void set(LoRAWeightsSet<TensorShard>* set) { mSet = set; }
        void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;
    private:
        LoRAWeightsSet<TensorShard>* mSet = nullptr;
    };

    StateContainer mMomentumContainer{&mMomentum};
    StateContainer mVarianceContainer{&mVariance};
    StateContainer mMomentumScalesContainer{&mMomentumScales};
    StateContainer mVarianceScalesContainer{&mVarianceScales};

    // Device staging buffers used when optimizer state is offloaded to host.
    // These are reused across all tensors and rely on stream ordering for correctness.
    Tensor mStagingM;
    Tensor mStagingV;
    Tensor mStagingMScales;
    Tensor mStagingVScales;

    void allocate_state();
};

// Helper functions

/**
 * @brief Calculate number of LoRA parameters
 */
std::size_t lora_num_parameters(const ModelConfig& model_config, const ModularLoRAConfig& lora_config);

/**
 * @brief Calculate bytes required for LoRA adapter
 */
std::size_t lora_bytes(const ModelConfig& model_config, const ModularLoRAConfig& lora_config);

} // namespace modules

// ============================================================================
// Implementation (header-only for now)
// ============================================================================

namespace modules {

inline ModularLoRAWeightsManager::ModularLoRAWeightsManager(const Config& config, TensorAllocator& allocator)
    : mConfig(config), mAllocator(&allocator) {
    mMaster.config = config.lora_config;
    mWork.config = config.lora_config;

    if (!enabled()) {
        return;
    }

    auto ctx = mAllocator->with_context("Modular_LoRA_Weights");
    mMaster.blocks.resize(config.num_layers);
    mWork.blocks.resize(config.num_layers);
    for (int l = 0; l < config.num_layers; ++l) {
        allocate_block_weights(l);
    }
}

inline void ModularLoRAWeightsManager::allocate_layer_weights(
    LoRALayerWeights<TensorShard>& shard,
    LoRALayerWeights<Tensor>& work,
    int in_features,
    int out_features,
    const std::string& name) {

    const int r = mConfig.lora_config.rank;
    const ETensorDType master_dtype = mConfig.lora_config.dtype;
    const ETensorDType work_dtype = mConfig.work_dtype;

    // Data-parallel LoRA: replicate weights on all ranks (no sharding yet).
    shard.A = TensorShard(mAllocator->allocate(master_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_features}));
    shard.B = mAllocator->allocate_shard(master_dtype, /*shard_idx=*/0, /*num_shards=*/1, (name + "_B").c_str(), {out_features, r});

    work.A = mAllocator->allocate(work_dtype, (name + "_A_work").c_str(), EAllocationType::ON_DEVICE, {r, in_features});
    work.B = mAllocator->allocate(work_dtype, (name + "_B_work").c_str(), EAllocationType::ON_DEVICE, {out_features, r});
}

inline void ModularLoRAWeightsManager::allocate_block_weights(int layer_idx) {
    if (!enabled()) return;

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int Hq = mConfig.num_query_heads;
    const int Hkv = mConfig.num_kv_heads;
    const int Hs = mConfig.head_size;
    const int q_out = Hq * Hs;
    const int kv_out = Hkv * Hs;

    auto& master = mMaster.blocks[layer_idx];
    auto& work = mWork.blocks[layer_idx];

    const std::string prefix = fmt::format("lora_layer_{}", layer_idx);

    if (mConfig.lora_config.applies_to_q()) {
        master.attention.q.emplace();
        work.attention.q.emplace();
        allocate_layer_weights(*master.attention.q, *work.attention.q, /*in=*/C, /*out=*/q_out, prefix + "_q");
    }
    if (mConfig.lora_config.applies_to_k()) {
        master.attention.k.emplace();
        work.attention.k.emplace();
        allocate_layer_weights(*master.attention.k, *work.attention.k, /*in=*/C, /*out=*/kv_out, prefix + "_k");
    }
    if (mConfig.lora_config.applies_to_v()) {
        master.attention.v.emplace();
        work.attention.v.emplace();
        allocate_layer_weights(*master.attention.v, *work.attention.v, /*in=*/C, /*out=*/kv_out, prefix + "_v");
    }
    if (mConfig.lora_config.applies_to_o()) {
        master.attention.o.emplace();
        work.attention.o.emplace();
        allocate_layer_weights(*master.attention.o, *work.attention.o, /*in=*/q_out, /*out=*/C, prefix + "_o");
    }

    // MLP LoRA: For dense models, use standard MLP LoRA. For MoE models, use per-expert LoRA.
    if (mConfig.is_moe && mConfig.num_experts > 0) {
        // Allocate per-expert LoRA weights for MoE models
        const bool has_mlp_lora = mConfig.lora_config.applies_to_gate() ||
                                   mConfig.lora_config.applies_to_up() ||
                                   mConfig.lora_config.applies_to_down();
        if (has_mlp_lora) {
            master.moe.experts.resize(mConfig.num_experts);
            work.moe.experts.resize(mConfig.num_experts);
            for (int e = 0; e < mConfig.num_experts; ++e) {
                allocate_expert_weights(master.moe.experts[e], work.moe.experts[e], layer_idx, e);
            }
        }
    } else {
        // Dense model: standard MLP LoRA
        if (mConfig.lora_config.applies_to_gate()) {
            master.mlp.gate.emplace();
            work.mlp.gate.emplace();
            allocate_layer_weights(*master.mlp.gate, *work.mlp.gate, /*in=*/C, /*out=*/D, prefix + "_gate");
        }
        if (mConfig.lora_config.applies_to_up()) {
            master.mlp.up.emplace();
            work.mlp.up.emplace();
            allocate_layer_weights(*master.mlp.up, *work.mlp.up, /*in=*/C, /*out=*/D, prefix + "_up");
        }
        if (mConfig.lora_config.applies_to_down()) {
            master.mlp.down.emplace();
            work.mlp.down.emplace();
            allocate_layer_weights(*master.mlp.down, *work.mlp.down, /*in=*/D, /*out=*/C, prefix + "_down");
        }
    }
}

inline void ModularLoRAWeightsManager::allocate_expert_weights(
    LoRAExpertWeights<TensorShard>& master_expert,
    LoRAExpertWeights<Tensor>& work_expert,
    int layer_idx, int expert_idx) {

    const int C = mConfig.hidden_size;
    const int D = mConfig.effective_moe_intermediate();
    const std::string prefix = fmt::format("lora_layer_{}_expert_{}", layer_idx, expert_idx);

    if (mConfig.lora_config.applies_to_gate()) {
        master_expert.gate.emplace();
        work_expert.gate.emplace();
        allocate_layer_weights(*master_expert.gate, *work_expert.gate, /*in=*/C, /*out=*/D, prefix + "_gate");
    }
    if (mConfig.lora_config.applies_to_up()) {
        master_expert.up.emplace();
        work_expert.up.emplace();
        allocate_layer_weights(*master_expert.up, *work_expert.up, /*in=*/C, /*out=*/D, prefix + "_up");
    }
    if (mConfig.lora_config.applies_to_down()) {
        master_expert.down.emplace();
        work_expert.down.emplace();
        allocate_layer_weights(*master_expert.down, *work_expert.down, /*in=*/D, /*out=*/C, prefix + "_down");
    }
}

inline void ModularLoRAWeightsManager::random_init(int seed, NCCLCommunicator& comm) {
    if (!enabled()) return;

    auto init_layer = [&](std::optional<LoRALayerWeights<TensorShard>>& layer,
                          int in_features,
                          unsigned long long subsequence) {
        if (!layer.has_value()) return;
        // Match legacy init: std consistent with kaiming_uniform_(a=sqrt(5)) => bound = 1/sqrt(fan_in)
        float std_a = 1.0f / std::sqrt(3.0f * static_cast<float>(in_features));
        fill_normal(layer->A, layer->A.nelem(), 0.0f, std_a, seed, subsequence, nullptr);
        fill_zero(layer->B, nullptr);
    };

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int D_moe = mConfig.effective_moe_intermediate();
    const int q_out = mConfig.num_query_heads * mConfig.head_size;
    const int E = mConfig.num_experts;

    for (int l = 0; l < mConfig.num_layers; ++l) {
        auto& b = mMaster.blocks[l];
        unsigned long long base = static_cast<unsigned long long>(l) * 32ULL;
        init_layer(b.attention.q, C, base + 0);
        init_layer(b.attention.k, C, base + 1);
        init_layer(b.attention.v, C, base + 2);
        init_layer(b.attention.o, q_out, base + 3);

        // Dense MLP LoRA
        init_layer(b.mlp.gate, C, base + 4);
        init_layer(b.mlp.up, C, base + 5);
        init_layer(b.mlp.down, D, base + 6);

        // MoE expert LoRA
        for (int e = 0; e < (int)b.moe.experts.size(); ++e) {
            auto& expert = b.moe.experts[e];
            // Use separate subsequence space for each expert to avoid correlation
            unsigned long long expert_base = base + 8ULL + static_cast<unsigned long long>(e) * 4ULL;
            init_layer(expert.gate, C, expert_base + 0);
            init_layer(expert.up, C, expert_base + 1);
            init_layer(expert.down, D_moe, expert_base + 2);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();
}

inline void ModularLoRAWeightsManager::import_from_file(const std::string& file_name, NCCLCommunicator& comm) {
    if (!enabled()) return;
    load_safetensors(file_name, *this, /*allow_cast=*/true);
    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();
}

inline void ModularLoRAWeightsManager::export_to_file(const std::string& file_name, NCCLCommunicator& comm) const {
    if (!enabled()) return;
    if (comm.rank() == 0) {
        write_safetensors(file_name, const_cast<ModularLoRAWeightsManager&>(*this));
    }
    comm.barrier();
}

inline LoRABlockWeights<Tensor>& ModularLoRAWeightsManager::get_block(int layer_idx, cudaStream_t stream) {
    auto& work = mWork.blocks[layer_idx];
    if (!enabled()) return work;

    auto& master = mMaster.blocks[layer_idx];

    auto sync_tensor = [&](Tensor& dst_t, const TensorShard& src_t, const char* name) {
        if (!dst_t.Data || !src_t.Data) return;
        if (dst_t.nelem() != src_t.nelem()) {
            throw std::logic_error(fmt::format("ModularLoRAWeightsManager::get_block: {} nelem mismatch (dst={}, src={})",
                                               name, dst_t.nelem(), src_t.nelem()));
        }

        if (dst_t.DType == src_t.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst_t.Data, src_t.Data, dst_t.bytes(), cudaMemcpyDeviceToDevice, stream));
            return;
        }

        if (dst_t.DType == ETensorDType::BF16 && src_t.DType == ETensorDType::FP32) {
            convert_dtype(dst_t.get<nv_bfloat16>(), src_t.get<float>(), dst_t.nelem(), stream);
            return;
        }
        if (dst_t.DType == ETensorDType::FP32 && src_t.DType == ETensorDType::BF16) {
            convert_dtype(dst_t.get<float>(), src_t.get<nv_bfloat16>(), dst_t.nelem(), stream);
            return;
        }

        throw std::logic_error(fmt::format(
            "ModularLoRAWeightsManager::get_block: unsupported dtype cast for {} (src={}, dst={})",
            name, dtype_to_str(src_t.DType), dtype_to_str(dst_t.DType)));
    };

    auto sync_layer = [&](std::optional<LoRALayerWeights<Tensor>>& dst,
                          const std::optional<LoRALayerWeights<TensorShard>>& src,
                          const char* layer_name) {
        if (!dst.has_value() || !src.has_value()) return;
        sync_tensor(dst->A, src->A, (std::string(layer_name) + ".A").c_str());
        sync_tensor(dst->B, src->B, (std::string(layer_name) + ".B").c_str());
    };

    sync_layer(work.attention.q, master.attention.q, "q_proj");
    sync_layer(work.attention.k, master.attention.k, "k_proj");
    sync_layer(work.attention.v, master.attention.v, "v_proj");
    sync_layer(work.attention.o, master.attention.o, "o_proj");

    // Dense MLP LoRA
    sync_layer(work.mlp.gate, master.mlp.gate, "gate_proj");
    sync_layer(work.mlp.up, master.mlp.up, "up_proj");
    sync_layer(work.mlp.down, master.mlp.down, "down_proj");

    // MoE expert LoRA
    for (int e = 0; e < (int)master.moe.experts.size(); ++e) {
        auto& master_expert = master.moe.experts[e];
        auto& work_expert = work.moe.experts[e];
        std::string expert_prefix = fmt::format("expert_{}", e);
        sync_layer(work_expert.gate, master_expert.gate, (expert_prefix + "_gate").c_str());
        sync_layer(work_expert.up, master_expert.up, (expert_prefix + "_up").c_str());
        sync_layer(work_expert.down, master_expert.down, (expert_prefix + "_down").c_str());
    }

    return work;
}

inline LoRABlockWeights<TensorShard>& ModularLoRAWeightsManager::get_master_block(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mMaster.blocks[layer_idx];
}

inline std::size_t ModularLoRAWeightsManager::num_parameters() const {
    if (!enabled()) return 0;

    const std::size_t r = static_cast<std::size_t>(mConfig.lora_config.rank);
    const std::size_t C = static_cast<std::size_t>(mConfig.hidden_size);
    const std::size_t D = static_cast<std::size_t>(mConfig.intermediate_size);
    const std::size_t D_moe = static_cast<std::size_t>(mConfig.effective_moe_intermediate());
    const std::size_t Hq = static_cast<std::size_t>(mConfig.num_query_heads);
    const std::size_t Hkv = static_cast<std::size_t>(mConfig.num_kv_heads);
    const std::size_t Hs = static_cast<std::size_t>(mConfig.head_size);
    const std::size_t q_out = Hq * Hs;
    const std::size_t kv_out = Hkv * Hs;
    const std::size_t E = static_cast<std::size_t>(mConfig.num_experts);

    std::size_t per_layer = 0;

    // Attention LoRA parameters
    if (mConfig.lora_config.applies_to_q()) per_layer += r * C + q_out * r;
    if (mConfig.lora_config.applies_to_k()) per_layer += r * C + kv_out * r;
    if (mConfig.lora_config.applies_to_v()) per_layer += r * C + kv_out * r;
    if (mConfig.lora_config.applies_to_o()) per_layer += r * q_out + C * r;

    // MLP LoRA parameters (dense or MoE)
    if (mConfig.is_moe && E > 0) {
        // Per-expert LoRA for MoE models
        std::size_t per_expert = 0;
        if (mConfig.lora_config.applies_to_gate()) per_expert += r * C + D_moe * r;
        if (mConfig.lora_config.applies_to_up()) per_expert += r * C + D_moe * r;
        if (mConfig.lora_config.applies_to_down()) per_expert += r * D_moe + C * r;
        per_layer += per_expert * E;
    } else {
        // Dense MLP LoRA
        if (mConfig.lora_config.applies_to_gate()) per_layer += r * C + D * r;
        if (mConfig.lora_config.applies_to_up()) per_layer += r * C + D * r;
        if (mConfig.lora_config.applies_to_down()) per_layer += r * D + C * r;
    }

    return per_layer * static_cast<std::size_t>(mConfig.num_layers);
}

inline void ModularLoRAWeightsManager::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {
    if (!enabled()) return;

    for (int l = 0; l < (int)mMaster.blocks.size(); ++l) {
        std::string prefix = fmt::format("base_model.model.model.layers.{}", l);
        auto& block = mMaster.blocks[l];

        if (block.attention.q.has_value()) {
            callback(prefix + ".self_attn.q_proj.lora_A.weight", block.attention.q->A);
            callback(prefix + ".self_attn.q_proj.lora_B.weight", block.attention.q->B);
        }
        if (block.attention.k.has_value()) {
            callback(prefix + ".self_attn.k_proj.lora_A.weight", block.attention.k->A);
            callback(prefix + ".self_attn.k_proj.lora_B.weight", block.attention.k->B);
        }
        if (block.attention.v.has_value()) {
            callback(prefix + ".self_attn.v_proj.lora_A.weight", block.attention.v->A);
            callback(prefix + ".self_attn.v_proj.lora_B.weight", block.attention.v->B);
        }
        if (block.attention.o.has_value()) {
            callback(prefix + ".self_attn.o_proj.lora_A.weight", block.attention.o->A);
            callback(prefix + ".self_attn.o_proj.lora_B.weight", block.attention.o->B);
        }

        // Dense MLP LoRA
        if (block.mlp.gate.has_value()) {
            callback(prefix + ".mlp.gate_proj.lora_A.weight", block.mlp.gate->A);
            callback(prefix + ".mlp.gate_proj.lora_B.weight", block.mlp.gate->B);
        }
        if (block.mlp.up.has_value()) {
            callback(prefix + ".mlp.up_proj.lora_A.weight", block.mlp.up->A);
            callback(prefix + ".mlp.up_proj.lora_B.weight", block.mlp.up->B);
        }
        if (block.mlp.down.has_value()) {
            callback(prefix + ".mlp.down_proj.lora_A.weight", block.mlp.down->A);
            callback(prefix + ".mlp.down_proj.lora_B.weight", block.mlp.down->B);
        }

        // MoE expert LoRA (HuggingFace naming convention: .mlp.experts.{e}.{proj})
        for (int e = 0; e < (int)block.moe.experts.size(); ++e) {
            auto& expert = block.moe.experts[e];
            std::string expert_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);

            if (expert.gate.has_value()) {
                callback(expert_prefix + ".gate_proj.lora_A.weight", expert.gate->A);
                callback(expert_prefix + ".gate_proj.lora_B.weight", expert.gate->B);
            }
            if (expert.up.has_value()) {
                callback(expert_prefix + ".up_proj.lora_A.weight", expert.up->A);
                callback(expert_prefix + ".up_proj.lora_B.weight", expert.up->B);
            }
            if (expert.down.has_value()) {
                callback(expert_prefix + ".down_proj.lora_A.weight", expert.down->A);
                callback(expert_prefix + ".down_proj.lora_B.weight", expert.down->B);
            }
        }
    }
}

inline ModularLoRAGradsManager::ModularLoRAGradsManager(const Config& config, const std::shared_ptr<TensorAllocator>& allocator)
    : mConfig(config), mAllocator(allocator) {
    mFullGrads.config = config.lora_config;
    mShardedGrads.config = config.lora_config;

    if (!config.lora_config.enabled()) return;
    allocate_gradients();
}

inline ModularLoRAGradsManager::~ModularLoRAGradsManager() = default;

inline void ModularLoRAGradsManager::allocate_gradients() {
    auto ctx = mAllocator->with_context("Modular_LoRA_Grads");
    mFullGrads.blocks.resize(mConfig.num_layers);
    mShardedGrads.blocks.resize(mConfig.num_layers);

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int D_moe = mConfig.effective_moe_intermediate();
    const int q_out = mConfig.num_query_heads * mConfig.head_size;
    const int kv_out = mConfig.num_kv_heads * mConfig.head_size;
    const int r = mConfig.lora_config.rank;
    const int E = mConfig.num_experts;

    auto alloc_full = [&](int in_f, int out_f, const std::string& name) -> LoRALayerWeights<Tensor> {
        LoRALayerWeights<Tensor> w;
        w.A = mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_f});
        w.B = mAllocator->allocate(mConfig.grad_dtype, (name + "_B").c_str(), EAllocationType::ON_DEVICE, {out_f, r});
        return w;
    };
    auto alloc_shard = [&](int in_f, int out_f, const std::string& name) -> LoRALayerWeights<TensorShard> {
        LoRALayerWeights<TensorShard> w;
        w.A = TensorShard(mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_f}));
        w.B = mAllocator->allocate_shard(mConfig.grad_dtype, /*shard_idx=*/0, /*num_shards=*/1, (name + "_B").c_str(), {out_f, r});
        return w;
    };

    for (int l = 0; l < mConfig.num_layers; ++l) {
        std::string prefix = fmt::format("lora_grad_layer_{}", l);
        auto& full = mFullGrads.blocks[l];
        auto& shard = mShardedGrads.blocks[l];

        if (mConfig.lora_config.applies_to_q()) {
            full.attention.q = alloc_full(C, q_out, prefix + "_q");
            shard.attention.q = alloc_shard(C, q_out, prefix + "_q_shard");
        }
        if (mConfig.lora_config.applies_to_k()) {
            full.attention.k = alloc_full(C, kv_out, prefix + "_k");
            shard.attention.k = alloc_shard(C, kv_out, prefix + "_k_shard");
        }
        if (mConfig.lora_config.applies_to_v()) {
            full.attention.v = alloc_full(C, kv_out, prefix + "_v");
            shard.attention.v = alloc_shard(C, kv_out, prefix + "_v_shard");
        }
        if (mConfig.lora_config.applies_to_o()) {
            full.attention.o = alloc_full(q_out, C, prefix + "_o");
            shard.attention.o = alloc_shard(q_out, C, prefix + "_o_shard");
        }

        // MLP LoRA gradients: For MoE models, allocate per-expert gradients
        if (mConfig.is_moe && E > 0) {
            const bool has_mlp_lora = mConfig.lora_config.applies_to_gate() ||
                                       mConfig.lora_config.applies_to_up() ||
                                       mConfig.lora_config.applies_to_down();
            if (has_mlp_lora) {
                full.moe.experts.resize(E);
                shard.moe.experts.resize(E);
                for (int e = 0; e < E; ++e) {
                    std::string expert_prefix = fmt::format("{}_expert_{}", prefix, e);
                    auto& full_expert = full.moe.experts[e];
                    auto& shard_expert = shard.moe.experts[e];

                    if (mConfig.lora_config.applies_to_gate()) {
                        full_expert.gate = alloc_full(C, D_moe, expert_prefix + "_gate");
                        shard_expert.gate = alloc_shard(C, D_moe, expert_prefix + "_gate_shard");
                    }
                    if (mConfig.lora_config.applies_to_up()) {
                        full_expert.up = alloc_full(C, D_moe, expert_prefix + "_up");
                        shard_expert.up = alloc_shard(C, D_moe, expert_prefix + "_up_shard");
                    }
                    if (mConfig.lora_config.applies_to_down()) {
                        full_expert.down = alloc_full(D_moe, C, expert_prefix + "_down");
                        shard_expert.down = alloc_shard(D_moe, C, expert_prefix + "_down_shard");
                    }
                }
            }
        } else {
            // Dense MLP LoRA gradients
            if (mConfig.lora_config.applies_to_gate()) {
                full.mlp.gate = alloc_full(C, D, prefix + "_gate");
                shard.mlp.gate = alloc_shard(C, D, prefix + "_gate_shard");
            }
            if (mConfig.lora_config.applies_to_up()) {
                full.mlp.up = alloc_full(C, D, prefix + "_up");
                shard.mlp.up = alloc_shard(C, D, prefix + "_up_shard");
            }
            if (mConfig.lora_config.applies_to_down()) {
                full.mlp.down = alloc_full(D, C, prefix + "_down");
                shard.mlp.down = alloc_shard(D, C, prefix + "_down_shard");
            }
        }
    }
}

inline void ModularLoRAGradsManager::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    mIsFirstMicroStep = (micro_step == 0);
    mIsLastMicroStep = (micro_step == total_steps - 1);

    if (!mConfig.lora_config.enabled()) return;

    if (mIsFirstMicroStep) {
        for (auto& block : mFullGrads.blocks) {
            auto zero_layer = [stream](auto& opt_layer) {
                if (!opt_layer.has_value()) return;
                if (opt_layer->A.Data) fill_zero(opt_layer->A, stream);
                if (opt_layer->B.Data) fill_zero(opt_layer->B, stream);
            };
            zero_layer(block.attention.q);
            zero_layer(block.attention.k);
            zero_layer(block.attention.v);
            zero_layer(block.attention.o);
            zero_layer(block.mlp.gate);
            zero_layer(block.mlp.up);
            zero_layer(block.mlp.down);

            // MoE expert LoRA gradients
            for (auto& expert : block.moe.experts) {
                zero_layer(expert.gate);
                zero_layer(expert.up);
                zero_layer(expert.down);
            }
        }
    }
}

inline void ModularLoRAGradsManager::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    if (!mConfig.lora_config.enabled()) return;
    if (mIsLastMicroStep) {
        reduce_gradients(stream, comm);
    }
}

inline LoRABlockWeights<Tensor>& ModularLoRAGradsManager::get_block_full(
    int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    (void)stream;
    (void)comm;
    accumulate = !mIsFirstMicroStep;
    return mFullGrads.blocks[layer_idx];
}

inline LoRABlockWeights<TensorShard>& ModularLoRAGradsManager::get_block_shard(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mShardedGrads.blocks[layer_idx];
}

inline void ModularLoRAGradsManager::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    (void)layer_idx;
    (void)stream;
    (void)comm;
    // No-op for now (reduction batched in end_micro_step).
}

inline void ModularLoRAGradsManager::reduce_gradients(cudaStream_t stream, NCCLCommunicator& comm) {
    if (comm.world_size() == 1) return;

    auto all_reduce_layer = [&](std::optional<LoRALayerWeights<Tensor>>& layer) {
        if (!layer.has_value()) return;
        if (layer->A.Data) comm.all_reduce_avg(layer->A, stream);
        if (layer->B.Data) comm.all_reduce_avg(layer->B, stream);
    };

    for (auto& block : mFullGrads.blocks) {
        all_reduce_layer(block.attention.q);
        all_reduce_layer(block.attention.k);
        all_reduce_layer(block.attention.v);
        all_reduce_layer(block.attention.o);
        all_reduce_layer(block.mlp.gate);
        all_reduce_layer(block.mlp.up);
        all_reduce_layer(block.mlp.down);

        // MoE expert LoRA gradients
        for (auto& expert : block.moe.experts) {
            all_reduce_layer(expert.gate);
            all_reduce_layer(expert.up);
            all_reduce_layer(expert.down);
        }
    }
}

inline ModularLoRAOptimizerState::ModularLoRAOptimizerState(const Config& config, cudaStream_t stream,
                                                            NCCLCommunicator& comm, TensorAllocator& allocator)
    : mConfig(config), mAllocator(&allocator) {
    mMomentum.config = config.lora_config;
    mVariance.config = config.lora_config;
    mMomentumScales.config = config.lora_config;
    mVarianceScales.config = config.lora_config;

    if (!config.lora_config.enabled()) return;

    allocate_state();

    auto zero_layer = [stream](auto& opt_layer) {
        if (!opt_layer.has_value()) return;
        if (opt_layer->A.Data) {
            if (opt_layer->A.Device < 0) {
                std::memset(opt_layer->A.Data, 0, opt_layer->A.bytes());
            } else {
                fill_zero(opt_layer->A, stream);
            }
        }
        if (opt_layer->B.Data) {
            if (opt_layer->B.Device < 0) {
                std::memset(opt_layer->B.Data, 0, opt_layer->B.bytes());
            } else {
                fill_zero(opt_layer->B, stream);
            }
        }
    };

    for (auto& block : mMomentum.blocks) {
        zero_layer(block.attention.q);
        zero_layer(block.attention.k);
        zero_layer(block.attention.v);
        zero_layer(block.attention.o);
        zero_layer(block.mlp.gate);
        zero_layer(block.mlp.up);
        zero_layer(block.mlp.down);
    }
    for (auto& block : mVariance.blocks) {
        zero_layer(block.attention.q);
        zero_layer(block.attention.k);
        zero_layer(block.attention.v);
        zero_layer(block.attention.o);
        zero_layer(block.mlp.gate);
        zero_layer(block.mlp.up);
        zero_layer(block.mlp.down);
    }

    // Staging buffers are always device-resident.
    // Make sure any device-side zeroing has completed before proceeding.
    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();
}

inline ModularLoRAOptimizerState::~ModularLoRAOptimizerState() = default;

inline void ModularLoRAOptimizerState::allocate_state() {
    auto ctx = mAllocator->with_context("Modular_LoRA_OptState");
    mMomentum.blocks.resize(mConfig.num_layers);
    mVariance.blocks.resize(mConfig.num_layers);
    mMomentumScales.blocks.resize(mConfig.num_layers);
    mVarianceScales.blocks.resize(mConfig.num_layers);

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int q_out = mConfig.num_query_heads * mConfig.head_size;
    const int kv_out = mConfig.num_kv_heads * mConfig.head_size;
    const int r = mConfig.lora_config.rank;

    const EAllocationType kind_m = mConfig.offload_m ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;
    const EAllocationType kind_v = mConfig.offload_v ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;

    auto alloc_state = [&](ETensorDType dtype, EAllocationType kind, int in_f, int out_f, const std::string& name) -> LoRALayerWeights<TensorShard> {
        LoRALayerWeights<TensorShard> w;
        w.A = TensorShard(mAllocator->allocate(dtype, (name + "_A").c_str(), kind, {r, in_f}));
        w.B = mAllocator->allocate_shard(dtype, /*shard_idx=*/0, /*num_shards=*/1, (name + "_B").c_str(), {out_f, r}, kind);
        return w;
    };

    auto alloc_scales = [&](EAllocationType kind, int in_f, int out_f, const std::string& name) -> LoRALayerWeights<TensorShard> {
        LoRALayerWeights<TensorShard> w;
        const long a_elems = static_cast<long>(r) * static_cast<long>(in_f);
        const long b_elems = static_cast<long>(out_f) * static_cast<long>(r);
        const long a_blocks = div_ceil(a_elems, 128L);
        const long b_blocks = div_ceil(b_elems, 128L);
        w.A = TensorShard(mAllocator->allocate(ETensorDType::FP32, (name + "_A").c_str(), kind, {a_blocks}));
        w.B = TensorShard(mAllocator->allocate(ETensorDType::FP32, (name + "_B").c_str(), kind, {b_blocks}));
        return w;
    };

    for (int l = 0; l < mConfig.num_layers; ++l) {
        std::string prefix = fmt::format("lora_opt_layer_{}", l);
        auto& m = mMomentum.blocks[l];
        auto& v = mVariance.blocks[l];

        if (mConfig.lora_config.applies_to_q()) {
            m.attention.q = alloc_state(mConfig.m_dtype, kind_m, C, q_out, prefix + "_q_m");
            v.attention.q = alloc_state(mConfig.v_dtype, kind_v, C, q_out, prefix + "_q_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].attention.q = alloc_scales(kind_m, C, q_out, prefix + "_q_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].attention.q = alloc_scales(kind_v, C, q_out, prefix + "_q_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_k()) {
            m.attention.k = alloc_state(mConfig.m_dtype, kind_m, C, kv_out, prefix + "_k_m");
            v.attention.k = alloc_state(mConfig.v_dtype, kind_v, C, kv_out, prefix + "_k_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].attention.k = alloc_scales(kind_m, C, kv_out, prefix + "_k_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].attention.k = alloc_scales(kind_v, C, kv_out, prefix + "_k_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_v()) {
            m.attention.v = alloc_state(mConfig.m_dtype, kind_m, C, kv_out, prefix + "_v_m");
            v.attention.v = alloc_state(mConfig.v_dtype, kind_v, C, kv_out, prefix + "_v_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].attention.v = alloc_scales(kind_m, C, kv_out, prefix + "_v_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].attention.v = alloc_scales(kind_v, C, kv_out, prefix + "_v_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_o()) {
            m.attention.o = alloc_state(mConfig.m_dtype, kind_m, q_out, C, prefix + "_o_m");
            v.attention.o = alloc_state(mConfig.v_dtype, kind_v, q_out, C, prefix + "_o_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].attention.o = alloc_scales(kind_m, q_out, C, prefix + "_o_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].attention.o = alloc_scales(kind_v, q_out, C, prefix + "_o_v_scales");
            }
        }

        if (mConfig.lora_config.applies_to_gate()) {
            m.mlp.gate = alloc_state(mConfig.m_dtype, kind_m, C, D, prefix + "_gate_m");
            v.mlp.gate = alloc_state(mConfig.v_dtype, kind_v, C, D, prefix + "_gate_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].mlp.gate = alloc_scales(kind_m, C, D, prefix + "_gate_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].mlp.gate = alloc_scales(kind_v, C, D, prefix + "_gate_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_up()) {
            m.mlp.up = alloc_state(mConfig.m_dtype, kind_m, C, D, prefix + "_up_m");
            v.mlp.up = alloc_state(mConfig.v_dtype, kind_v, C, D, prefix + "_up_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].mlp.up = alloc_scales(kind_m, C, D, prefix + "_up_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].mlp.up = alloc_scales(kind_v, C, D, prefix + "_up_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_down()) {
            m.mlp.down = alloc_state(mConfig.m_dtype, kind_m, D, C, prefix + "_down_m");
            v.mlp.down = alloc_state(mConfig.v_dtype, kind_v, D, C, prefix + "_down_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].mlp.down = alloc_scales(kind_m, D, C, prefix + "_down_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].mlp.down = alloc_scales(kind_v, D, C, prefix + "_down_v_scales");
            }
        }
    }

    // Allocate device staging buffers when host offload is enabled.
    // Sized for the largest LoRA moment tensor (A or B) across modules.
    const int max_features = std::max({C, D, q_out, kv_out});
    const long max_elems = static_cast<long>(r) * static_cast<long>(max_features);
    const long max_scale_elems = div_ceil(max_elems, 128L);
    if (mConfig.offload_m && !mStagingM.Data) {
        mStagingM = mAllocator->allocate(mConfig.m_dtype, "lora_opt_m_stage", EAllocationType::ON_DEVICE, {max_elems});
    }
    if (mConfig.offload_v && !mStagingV.Data) {
        mStagingV = mAllocator->allocate(mConfig.v_dtype, "lora_opt_v_stage", EAllocationType::ON_DEVICE, {max_elems});
    }
    if (mConfig.offload_m && is_fp8_dtype(mConfig.m_dtype) && !mStagingMScales.Data) {
        mStagingMScales = mAllocator->allocate(ETensorDType::FP32, "lora_opt_m_scales_stage", EAllocationType::ON_DEVICE, {max_scale_elems});
    }
    if (mConfig.offload_v && is_fp8_dtype(mConfig.v_dtype) && !mStagingVScales.Data) {
        mStagingVScales = mAllocator->allocate(ETensorDType::FP32, "lora_opt_v_scales_stage", EAllocationType::ON_DEVICE, {max_scale_elems});
    }
}

inline LoRABlockWeights<TensorShard>& ModularLoRAOptimizerState::get_block_m(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mMomentum.blocks[layer_idx];
}

inline LoRABlockWeights<TensorShard>& ModularLoRAOptimizerState::get_block_v(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mVariance.blocks[layer_idx];
}

inline LoRABlockWeights<TensorShard>& ModularLoRAOptimizerState::get_block_m_scales(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mMomentumScales.blocks[layer_idx];
}

inline LoRABlockWeights<TensorShard>& ModularLoRAOptimizerState::get_block_v_scales(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mVarianceScales.blocks[layer_idx];
}

inline ITensorContainer& ModularLoRAOptimizerState::full_m() {
    return mMomentumContainer;
}

inline ITensorContainer& ModularLoRAOptimizerState::full_v() {
    return mVarianceContainer;
}

inline ITensorContainer& ModularLoRAOptimizerState::full_m_scales() {
    return mMomentumScalesContainer;
}

inline ITensorContainer& ModularLoRAOptimizerState::full_v_scales() {
    return mVarianceScalesContainer;
}

inline void ModularLoRAOptimizerState::StateContainer::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {
    if (!mSet) return;

    for (int l = 0; l < (int)mSet->blocks.size(); ++l) {
        std::string prefix = fmt::format("base_model.model.model.layers.{}", l);
        auto& block = mSet->blocks[l];

        if (block.attention.q.has_value()) {
            callback(prefix + ".self_attn.q_proj.lora_A.weight", block.attention.q->A);
            callback(prefix + ".self_attn.q_proj.lora_B.weight", block.attention.q->B);
        }
        if (block.attention.k.has_value()) {
            callback(prefix + ".self_attn.k_proj.lora_A.weight", block.attention.k->A);
            callback(prefix + ".self_attn.k_proj.lora_B.weight", block.attention.k->B);
        }
        if (block.attention.v.has_value()) {
            callback(prefix + ".self_attn.v_proj.lora_A.weight", block.attention.v->A);
            callback(prefix + ".self_attn.v_proj.lora_B.weight", block.attention.v->B);
        }
        if (block.attention.o.has_value()) {
            callback(prefix + ".self_attn.o_proj.lora_A.weight", block.attention.o->A);
            callback(prefix + ".self_attn.o_proj.lora_B.weight", block.attention.o->B);
        }

        // Dense MLP LoRA
        if (block.mlp.gate.has_value()) {
            callback(prefix + ".mlp.gate_proj.lora_A.weight", block.mlp.gate->A);
            callback(prefix + ".mlp.gate_proj.lora_B.weight", block.mlp.gate->B);
        }
        if (block.mlp.up.has_value()) {
            callback(prefix + ".mlp.up_proj.lora_A.weight", block.mlp.up->A);
            callback(prefix + ".mlp.up_proj.lora_B.weight", block.mlp.up->B);
        }
        if (block.mlp.down.has_value()) {
            callback(prefix + ".mlp.down_proj.lora_A.weight", block.mlp.down->A);
            callback(prefix + ".mlp.down_proj.lora_B.weight", block.mlp.down->B);
        }

        // MoE expert LoRA
        for (int e = 0; e < (int)block.moe.experts.size(); ++e) {
            auto& expert = block.moe.experts[e];
            std::string expert_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);

            if (expert.gate.has_value()) {
                callback(expert_prefix + ".gate_proj.lora_A.weight", expert.gate->A);
                callback(expert_prefix + ".gate_proj.lora_B.weight", expert.gate->B);
            }
            if (expert.up.has_value()) {
                callback(expert_prefix + ".up_proj.lora_A.weight", expert.up->A);
                callback(expert_prefix + ".up_proj.lora_B.weight", expert.up->B);
            }
            if (expert.down.has_value()) {
                callback(expert_prefix + ".down_proj.lora_A.weight", expert.down->A);
                callback(expert_prefix + ".down_proj.lora_B.weight", expert.down->B);
            }
        }
    }
}

inline std::size_t lora_num_parameters(const ModelConfig& model_config, const ModularLoRAConfig& lora_config) {
    if (!lora_config.enabled()) return 0;

    const std::size_t r = static_cast<std::size_t>(lora_config.rank);
    const std::size_t C = static_cast<std::size_t>(model_config.HiddenSize);
    const std::size_t D = static_cast<std::size_t>(model_config.IntermediateSize);
    const std::size_t Hq = static_cast<std::size_t>(model_config.NumQueryHeads);
    const std::size_t Hkv = static_cast<std::size_t>(model_config.NumKeyValHeads);
    const std::size_t Hs = static_cast<std::size_t>(model_config.head_size());
    const std::size_t q_out = Hq * Hs;
    const std::size_t kv_out = Hkv * Hs;

    std::size_t per_layer = 0;
    if (lora_config.applies_to_q()) per_layer += r * C + q_out * r;
    if (lora_config.applies_to_k()) per_layer += r * C + kv_out * r;
    if (lora_config.applies_to_v()) per_layer += r * C + kv_out * r;
    if (lora_config.applies_to_o()) per_layer += r * q_out + C * r;
    if (lora_config.applies_to_gate()) per_layer += r * C + D * r;
    if (lora_config.applies_to_up()) per_layer += r * C + D * r;
    if (lora_config.applies_to_down()) per_layer += r * D + C * r;

    return per_layer * static_cast<std::size_t>(model_config.NumLayers);
}

inline std::size_t lora_bytes(const ModelConfig& model_config, const ModularLoRAConfig& lora_config) {
    return lora_num_parameters(model_config, lora_config) * get_dtype_size(lora_config.dtype);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_WEIGHTS_H
