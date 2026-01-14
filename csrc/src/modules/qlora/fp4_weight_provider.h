// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP4 weight provider: on-the-fly dequantization of FP4 weights to BF16

#ifndef SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H
#define SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H

#include <memory>
#include <vector>

#include "fp4_weights.h"
#include "fp4_block_quantized_tensor.h"
#include "moe_weights.h"
#include "qlora_config.h"
#include "kernels/kernels.h"
#include "modules/composite/transformer_block.h"
#include "modules/lora/lora_config.h"
#include "modules/moe/moe_types.h"
#include "modules/weights/weight_manager_types.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/utils.h"

namespace modules {

/**
 * @brief Provides dequantized weights for FP4 QLoRA training
 *
 * This class wraps FP4WeightsManager and provides on-the-fly dequantization
 * of FP4 base weights to BF16 for use in the forward pass.
 *
 * Key design:
 * - Quantized weights (FP4 + two-level block scales) are stored permanently
 * - Dequantization buffers are allocated once and reused
 * - get_block() dequantizes the requested layer's weights to BF16
 * - Compatible with the existing ModularWeightManager interface patterns
 *
 * FP4 uses two-level block scaling:
 * - Level 1: FP8 E4M3 scale per 16 consecutive values
 * - Level 2: FP32 global per-tensor scale (amax)
 *
 * @tparam Block The transformer block type (e.g., DenseTransformerBlock<>)
 */
template<typename Block>
class FP4WeightProvider {
public:
    using BlockWeights = typename Block::Weights;
    using BlockConfig = typename Block::Config;

    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        int vocab_size;
        QLoRAConfig qlora_config;
        ModularLoRAConfig lora_config;
        ETensorDType model_dtype = ETensorDType::BF16;
        bool use_qk_norm = false;
        bool tied_embeddings = true;
        int shard_idx = 0;
        int num_shards = 1;

        /// Enable selective expert dequantization for MoE models.
        /// When enabled, only the experts selected by the router are dequantized,
        /// reducing memory usage from O(num_experts) to O(top_k) for dequant buffers.
        bool selective_expert_dequant = true;

        /// Offload MoE expert FP4 weights to CPU pinned memory.
        /// When enabled, expert weights are stored in CPU memory and streamed to GPU
        /// on-demand when selected by the router. Saves ~10GB for 128-expert models.
        /// Implies selective_expert_dequant = true.
        bool offload_experts = false;
    };

    FP4WeightProvider(const Config& config, TensorAllocator& allocator,
                      const cudaDeviceProp& device_props);
    ~FP4WeightProvider() = default;

    /**
     * @brief Import and quantize base model weights from file
     */
    void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                             cudaStream_t stream);

    /**
     * @brief Get dequantized block weights
     *
     * Dequantizes the FP4 weights for the specified layer and returns
     * a BlockWeights struct with BF16 tensors ready for matmul.
     *
     * Uses caching to avoid redundant dequantization within the same step
     * (forward and backward access the same layer).
     *
     * @param layer_idx Layer index
     * @param stream CUDA stream for dequantization
     * @return Reference to BlockWeights with dequantized tensors
     */
    BlockWeights& get_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Release block weights (no-op, kept for interface compat)
     */
    void release_block(int layer_idx, cudaStream_t stream) {
        (void)layer_idx;
        (void)stream;
    }

    /**
     * @brief Signal the start of a new training step
     */
    void new_step() {
        ++mStepVersion;
    }

    /**
     * @brief Legacy alias for new_step()
     */
    void invalidate_cache() {
        new_step();
    }

    /**
     * @brief Get embeddings (not quantized)
     */
    Tensor& get_embeddings(cudaStream_t stream) {
        (void)stream;
        return mFP4Weights->get_embeddings().embedding;
    }

    /**
     * @brief Get final norm weight (not quantized)
     */
    Tensor& get_final_norm(cudaStream_t stream);

    /**
     * @brief Get LM head (not quantized, may be tied to embeddings)
     */
    Tensor& get_lm_head(cudaStream_t stream) {
        (void)stream;
        return mFP4Weights->get_embeddings().lm_head;
    }

    /**
     * @brief Access the underlying FP4WeightsManager
     */
    FP4WeightsManager& fp4_weights() { return *mFP4Weights; }
    const FP4WeightsManager& fp4_weights() const { return *mFP4Weights; }

    /**
     * @brief Get QLoRA config
     */
    const QLoRAConfig& qlora_config() const { return mConfig.qlora_config; }

    // =========================================================================
    // MoE Support
    // =========================================================================

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] bool is_moe() const {
        return mFP4Weights && mFP4Weights->is_moe();
    }

    /**
     * @brief Get number of experts (0 for dense models)
     */
    [[nodiscard]] int num_experts() const {
        return mFP4Weights ? mFP4Weights->num_experts() : 0;
    }

    /**
     * @brief Get router gate weights for MoE (BF16, no dequant needed)
     *
     * Router gate is small and kept in BF16.
     *
     * @param layer_idx Layer index
     * @param stream CUDA stream (unused, kept for interface consistency)
     * @return Reference to router gate tensor (num_experts, hidden_size)
     */
    Tensor& get_router_gate(int layer_idx, cudaStream_t stream);

    /**
     * @brief Dequantize only the selected experts (selective dequantization)
     *
     * This method dequantizes only the experts that were selected by the router,
     * significantly reducing memory usage for MoE models with many experts.
     *
     * The dequantized weights are placed in a compact buffer indexed by
     * selection_info.expert_to_compact mapping.
     *
     * @param layer_idx Layer index
     * @param selection_info Information about which experts were selected
     * @param stream CUDA stream for dequantization
     */
    void dequantize_selected_experts(int layer_idx, const SelectiveExpertInfo& selection_info,
                                     cudaStream_t stream);

    /**
     * @brief Check if selective expert dequantization is enabled
     */
    [[nodiscard]] bool use_selective_dequant() const {
        return mConfig.selective_expert_dequant && mConfig.qlora_config.is_moe();
    }

    /**
     * @brief Get the current selection info (for backward pass)
     *
     * Returns the selection info cached from the last dequantize_selected_experts call.
     * This is used in backward to match the compact indexing used in forward.
     */
    [[nodiscard]] const SelectiveExpertInfo& get_current_selection() const {
        return mCurrentSelection;
    }

    /**
     * @brief Get the maximum number of active experts for buffer sizing
     */
    [[nodiscard]] int max_active_experts() const {
        return mNumMoEExperts;
    }

    /**
     * @brief Get memory stats
     */
    std::size_t quantized_weights_bytes() const {
        return mFP4Weights->quantized_weights_bytes();
    }

    float memory_savings_ratio() const {
        return mFP4Weights->memory_savings_ratio();
    }

private:
    /**
     * @brief Get attention and expert weights for MoE blocks
     *
     * Dequantizes attention weights (QKV + output) and ALL expert weights
     * into batched tensors for efficient forward pass.
     *
     * @param layer_idx Layer index
     * @param stream CUDA stream for dequantization
     */
    void get_moe_attention_weights(int layer_idx, cudaStream_t stream);

    Config mConfig;
    TensorAllocator* mAllocator;
    cudaDeviceProp mDeviceProps;  // Store by value to avoid dangling pointer

    // The underlying FP4 weights manager (owns quantized weights)
    std::unique_ptr<FP4WeightsManager> mFP4Weights;

    // Dequantization buffers (BF16, single layer, reused)
    Tensor mDequantQKV;
    Tensor mDequantOut;
    Tensor mDequantGateUp;
    Tensor mDequantDown;

    // Final norm weight (not quantized)
    Tensor mFinalNormWeight;

    // Cached dequantized block weights
    BlockWeights mDequantBlock;

    // Zero-overhead forward/backward cache via step versioning
    int mCurrentLayer = -1;
    uint64_t mStepVersion = 0;
    uint64_t mBufferVersion = 0;

    // =========================================================================
    // MoE-specific members for batched expert dequantization
    // =========================================================================

    /// Batched expert dequantization buffers (all experts, for forward pass)
    /// Shape: (num_experts, 2 * moe_intermediate, hidden_size)
    Tensor mBatchedExpertGateUp;
    /// Shape: (num_experts, hidden_size, moe_intermediate)
    Tensor mBatchedExpertDown;

    /// Number of experts in MoE model
    int mNumMoEExperts = 0;

    /// Current selection info for selective dequantization caching
    SelectiveExpertInfo mCurrentSelection;

    /// Layer index for which the current expert selection is valid
    /// Used to avoid reusing cached experts from a different layer
    int mCurrentExpertLayer = -1;

    /// Number of experts currently in the compact buffers
    int mNumActiveExperts = 0;

    // =========================================================================
    // Expert offloading support
    // =========================================================================

    /// GPU staging buffer for streaming FP4 expert data from CPU
    /// Size: max(expert_gate_up_bytes, expert_down_bytes)
    Tensor mExpertFP4Staging;

    /// GPU staging buffer for FP8 block scales (for offloaded experts)
    Tensor mExpertScalesStaging;

    /// Whether expert weights are offloaded to CPU
    bool mExpertsOffloaded = false;

    void allocate_dequant_buffers();
    void allocate_moe_expert_buffers();
    void allocate_offload_staging_buffers();
    void setup_block_weights_structure();
    void dequantize_fp4_weight(const FP4BlockQuantizedWeight& src, Tensor& dst, cudaStream_t stream);

    /// Stream FP4 expert weight from CPU to GPU staging buffer, then dequantize
    void stream_and_dequantize_expert(const FP4BlockQuantizedWeight& src, Tensor& dst,
                                      cudaStream_t stream);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
FP4WeightProvider<Block>::FP4WeightProvider(
    const Config& config, TensorAllocator& allocator, const cudaDeviceProp& device_props)
    : mConfig(config)
    , mAllocator(&allocator)
    , mDeviceProps(device_props)  // Copy by value
    , mExpertsOffloaded(config.offload_experts && config.qlora_config.is_moe())
{
    // If offload_experts is enabled, force selective_expert_dequant to true
    if (mExpertsOffloaded && !mConfig.selective_expert_dequant) {
        mConfig.selective_expert_dequant = true;
    }

    // Create FP4 weights manager
    FP4WeightsManager::Config fp4_config{
        .num_layers = config.num_layers,
        .hidden_size = config.hidden_size,
        .intermediate_size = config.intermediate_size,
        .num_query_heads = config.num_query_heads,
        .num_kv_heads = config.num_kv_heads,
        .head_size = config.head_size,
        .vocab_size = config.vocab_size,
        .qlora_config = config.qlora_config,
        .use_qk_norm = config.use_qk_norm,
        .tied_embeddings = config.tied_embeddings,
        .shard_idx = config.shard_idx,
        .num_shards = config.num_shards,
        .offload_experts = config.offload_experts
    };
    mFP4Weights = std::make_unique<FP4WeightsManager>(fp4_config, allocator, device_props);

    // Allocate dequantization buffers
    allocate_dequant_buffers();

    // Allocate MoE expert buffers if needed
    if (mFP4Weights->is_moe()) {
        allocate_moe_expert_buffers();

        // Allocate staging buffers for CPU -> GPU streaming if offloading is enabled
        if (mExpertsOffloaded) {
            allocate_offload_staging_buffers();
        }
    }

    // Set up the block weights structure with pointers to dequant buffers
    setup_block_weights_structure();
}

template<typename Block>
void FP4WeightProvider<Block>::allocate_dequant_buffers() {
    auto ctx = mAllocator->with_context("FP4_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;

    // Allocate BF16 dequantization buffers - single layer, reused across all layers
    mDequantQKV = mAllocator->allocate(ETensorDType::BF16, "dequant_qkv",
                                        EAllocationType::ON_DEVICE,
                                        {(long)qkv_out, (long)hidden});

    mDequantOut = mAllocator->allocate(ETensorDType::BF16, "dequant_out",
                                        EAllocationType::ON_DEVICE,
                                        {(long)hidden, (long)(num_q_heads * head_size)});

    mDequantGateUp = mAllocator->allocate(ETensorDType::BF16, "dequant_gate_up",
                                           EAllocationType::ON_DEVICE,
                                           {(long)(2 * intermediate), (long)hidden});

    mDequantDown = mAllocator->allocate(ETensorDType::BF16, "dequant_down",
                                         EAllocationType::ON_DEVICE,
                                         {(long)hidden, (long)intermediate});

    // Final norm weight (not quantized)
    mFinalNormWeight = mAllocator->allocate(ETensorDType::BF16, "final_norm",
                                             EAllocationType::ON_DEVICE,
                                             {(long)hidden});
}

template<typename Block>
void FP4WeightProvider<Block>::setup_block_weights_structure() {
    // Point the dequant block's weight tensors to our buffers
    // Note: We only set up the main projection weights here.
    // Layer norm weights are set per-layer since they're small and stored in BF16.

    // Set up attention weights
    mDequantBlock.attention.qkv_weight = mDequantQKV;
    mDequantBlock.attention.out_weight = mDequantOut;

    // Set up MLP weights (only for dense blocks - MoE blocks have experts instead)
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        mDequantBlock.mlp_up_weight = mDequantGateUp;
        mDequantBlock.mlp_down_weight = mDequantDown;
    }

    // Set up MoE expert weights (batched layout)
    if constexpr (has_moe_weights<BlockWeights>::value) {
        if (mConfig.qlora_config.is_moe()) {
            mDequantBlock.experts.use_batched = true;
            mDequantBlock.experts.gate_up_proj = mBatchedExpertGateUp;
            mDequantBlock.experts.down_proj = mBatchedExpertDown;
        }
    }
}

template<typename Block>
void FP4WeightProvider<Block>::import_and_quantize(
    const std::string& file_name, NCCLCommunicator& comm, cudaStream_t stream) {

    // Import and quantize base model weights to FP4
    mFP4Weights->import_and_quantize(file_name, comm, stream);

    // Load final norm weight from file (not quantized)
    SafeTensorsReader reader(file_name);
    const std::vector<std::string> final_norm_names = {
        "model.norm.weight",
        "transformer.ln_f.weight",
        "model.final_layernorm.weight"
    };
    for (const auto& name : final_norm_names) {
        bool found = false;
        for (const auto& entry : reader.entries()) {
            if (entry.name() == name) {
                entry.read_tensor(mFinalNormWeight, /*allow_cast=*/true);
                found = true;
                break;
            }
        }
        if (found) break;
    }
}

template<typename Block>
typename FP4WeightProvider<Block>::BlockWeights& FP4WeightProvider<Block>::get_block(
    int layer_idx, cudaStream_t stream) {

    // For MoE models, use the MoE-specific path that only handles attention weights
    // MoE models don't have dense MLP weights - they have per-expert weights instead
    if (is_moe()) {
        get_moe_attention_weights(layer_idx, stream);
        return mDequantBlock;
    }

    // Dense model path
    const auto& fp4_block = mFP4Weights->get_fp4_block(layer_idx);

    // Check if we already have this layer dequantized in the current step
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Cache miss: need to dequantize FP4 → BF16
        // Use the FP4 block dequantization kernel which handles two-level scaling:
        // FP4 data * FP8 block scale * global amax scale → BF16

        // QKV projection
        float qkv_scale = fp4_block.qkv_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantQKV.get<nv_bfloat16>(),
            fp4_block.qkv_proj.data.get<uint8_t>(),
            fp4_block.qkv_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            qkv_scale,
            fp4_block.qkv_proj.M, fp4_block.qkv_proj.K,
            mDeviceProps, stream);

        // Output projection
        float out_scale = fp4_block.out_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantOut.get<nv_bfloat16>(),
            fp4_block.out_proj.data.get<uint8_t>(),
            fp4_block.out_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            out_scale,
            fp4_block.out_proj.M, fp4_block.out_proj.K,
            mDeviceProps, stream);

        // Gate+Up projection
        float gate_up_scale = fp4_block.gate_up_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantGateUp.get<nv_bfloat16>(),
            fp4_block.gate_up_proj.data.get<uint8_t>(),
            fp4_block.gate_up_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            gate_up_scale,
            fp4_block.gate_up_proj.M, fp4_block.gate_up_proj.K,
            mDeviceProps, stream);

        // Down projection
        float down_scale = fp4_block.down_proj.global_decode_scale_rowwise();
        dequantize_fp4_block(
            mDequantDown.get<nv_bfloat16>(),
            fp4_block.down_proj.data.get<uint8_t>(),
            fp4_block.down_proj.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            down_scale,
            fp4_block.down_proj.M, fp4_block.down_proj.K,
            mDeviceProps, stream);

        // Synchronize to ensure dequantization completes before returning.
        // This fixes an intermittent hang that occurs when the dequant kernels
        // don't complete in time before subsequent matmul operations.
        // TODO: Investigate root cause - all operations are on the same stream
        // so this sync shouldn't be necessary in theory.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Update cache metadata
        mCurrentLayer = layer_idx;
        mBufferVersion = mStepVersion;
    }

    // Update layer norm pointers (these are just references, not cached data)
    mDequantBlock.ln1.weight = fp4_block.ln1_weight;
    mDequantBlock.ln2.weight = fp4_block.ln2_weight;

    // Copy QK-norm weights if present (for models like Qwen3)
    if constexpr (requires { mDequantBlock.attention.q_norm_weight; mDequantBlock.attention.k_norm_weight; }) {
        if (fp4_block.q_norm_weight.has_value() && fp4_block.k_norm_weight.has_value()) {
            mDequantBlock.attention.q_norm_weight = fp4_block.q_norm_weight;
            mDequantBlock.attention.k_norm_weight = fp4_block.k_norm_weight;
        }
    }

    return mDequantBlock;
}

template<typename Block>
Tensor& FP4WeightProvider<Block>::get_final_norm(cudaStream_t stream) {
    (void)stream;
    return mFinalNormWeight;
}

// ============================================================================
// MoE Support Implementation
// ============================================================================

template<typename Block>
void FP4WeightProvider<Block>::allocate_moe_expert_buffers() {
    auto ctx = mAllocator->with_context("FP4_MoE_DequantBuf");

    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
    const int num_experts = mConfig.qlora_config.num_experts;

    mNumMoEExperts = num_experts;

    // Allocate batched expert buffers (all experts for forward pass)
    // gate_up_proj: (num_experts, 2 * moe_intermediate, hidden_size)
    // down_proj: (num_experts, hidden_size, moe_intermediate)
    mBatchedExpertGateUp = mAllocator->allocate(ETensorDType::BF16,
        "batched_expert_gate_up",
        EAllocationType::ON_DEVICE,
        {(long)num_experts, (long)(2 * moe_inter), (long)hidden});

    mBatchedExpertDown = mAllocator->allocate(ETensorDType::BF16,
        "batched_expert_down",
        EAllocationType::ON_DEVICE,
        {(long)num_experts, (long)hidden, (long)moe_inter});
}

template<typename Block>
Tensor& FP4WeightProvider<Block>::get_router_gate(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mFP4Weights->get_moe_block(layer_idx).router_gate;
}

template<typename Block>
void FP4WeightProvider<Block>::get_moe_attention_weights(int layer_idx, cudaStream_t stream) {
    const auto& qblock = mFP4Weights->get_moe_block(layer_idx);

    // Check cache for this layer's weights
    const bool cache_hit = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    if (!cache_hit) {
        // Dequantize attention weights
        dequantize_fp4_weight(qblock.qkv_proj, mDequantQKV, stream);
        dequantize_fp4_weight(qblock.out_proj, mDequantOut, stream);

        // When selective_expert_dequant is enabled AND LoRA is enabled, skip dequantizing
        // all experts here. The LoRA forward hook will call dequantize_selected_experts()
        // later with only the router-selected experts. This saves memory.
        //
        // IMPORTANT: When LoRA is disabled (lora: false), we MUST dequantize all experts here
        // because the LoRA hook won't run and dequantize_selected_experts() won't be called.
        const bool use_selective_path = mConfig.selective_expert_dequant && mConfig.lora_config.enabled();

        if (!use_selective_path) {
            // Dequantize ALL expert weights into batched buffers
            const int hidden = mConfig.hidden_size;
            const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                                  mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;

            for (int e = 0; e < mNumMoEExperts; ++e) {
                const auto& expert_weights = qblock.experts[e];

                // Create slice views into the batched buffers for this expert
                Tensor gate_up_slice = Tensor::from_pointer(
                    static_cast<std::byte*>(mBatchedExpertGateUp.Data) +
                        static_cast<size_t>(e) * (2 * moe_inter) * hidden * sizeof(nv_bfloat16),
                    mBatchedExpertGateUp.Device,
                    ETensorDType::BF16,
                    std::array<long, 2>{2 * moe_inter, hidden}
                );

                Tensor down_slice = Tensor::from_pointer(
                    static_cast<std::byte*>(mBatchedExpertDown.Data) +
                        static_cast<size_t>(e) * hidden * moe_inter * sizeof(nv_bfloat16),
                    mBatchedExpertDown.Device,
                    ETensorDType::BF16,
                    std::array<long, 2>{hidden, moe_inter}
                );

                // Dequantize this expert's weights into the slice
                // When experts are offloaded to CPU, we need to stream them to GPU first
                if (mExpertsOffloaded) {
                    stream_and_dequantize_expert(expert_weights.gate_up_proj, gate_up_slice, stream);
                    stream_and_dequantize_expert(expert_weights.down_proj, down_slice, stream);
                } else {
                    dequantize_fp4_weight(expert_weights.gate_up_proj, gate_up_slice, stream);
                    dequantize_fp4_weight(expert_weights.down_proj, down_slice, stream);
                }
            }
        }
        // When selective mode is on, expert dequantization is deferred to dequantize_selected_experts()

        mCurrentLayer = layer_idx;
        mBufferVersion = mStepVersion;
    }

    // Update layer norm pointers
    mDequantBlock.ln1.weight = qblock.ln1_weight;
    mDequantBlock.ln2.weight = qblock.ln2_weight;

    // Update router gate pointer
    if constexpr (has_moe_weights<BlockWeights>::value) {
        mDequantBlock.router.gate = qblock.router_gate;
    }

    // Copy QK-norm weights if present
    if constexpr (requires { mDequantBlock.attention.q_norm_weight; mDequantBlock.attention.k_norm_weight; }) {
        if (qblock.q_norm_weight.has_value() && qblock.k_norm_weight.has_value()) {
            mDequantBlock.attention.q_norm_weight = qblock.q_norm_weight;
            mDequantBlock.attention.k_norm_weight = qblock.k_norm_weight;
        }
    }
}

// ============================================================================
// FP4-specific helper functions
// ============================================================================

template<typename Block>
void FP4WeightProvider<Block>::dequantize_fp4_weight(const FP4BlockQuantizedWeight& src,
                                                      Tensor& dst, cudaStream_t stream) {
    float global_scale = src.global_decode_scale_rowwise();
    dequantize_fp4_block(
        dst.get<nv_bfloat16>(),
        src.data.get<uint8_t>(),
        src.block_scales_rowwise.get<__nv_fp8_e4m3>(),
        global_scale,
        src.M, src.K,
        mDeviceProps, stream);
}

template<typename Block>
void FP4WeightProvider<Block>::allocate_offload_staging_buffers() {
    auto ctx = mAllocator->with_context("FP4_OffloadStaging");

    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;

    // Calculate the maximum FP4 packed size for a single expert weight
    // gate_up_proj: (2 * moe_inter, hidden) -> (2 * moe_inter * hidden) / 2 bytes
    // down_proj: (hidden, moe_inter) -> (hidden * moe_inter) / 2 bytes
    const long gate_up_elems = 2L * moe_inter * hidden;
    const long down_elems = static_cast<long>(hidden) * moe_inter;
    const long max_elems = std::max(gate_up_elems, down_elems);
    const long max_packed_bytes = FP4BlockScaleConfig::packed_data_bytes(
        std::max(2 * moe_inter, hidden),
        std::max(hidden, moe_inter));

    // Allocate staging buffer for FP4 packed data
    mExpertFP4Staging = mAllocator->allocate(ETensorDType::BYTE, "expert_fp4_staging",
                                              EAllocationType::ON_DEVICE, {max_packed_bytes});

    // Calculate max scale tensor size
    // FP4 uses FP8 E4M3 block scales with F8_128x4 alignment
    auto [gate_up_scale_rows, gate_up_scale_cols] = FP4BlockScaleConfig::scale_dims(2 * moe_inter, hidden);
    auto [down_scale_rows, down_scale_cols] = FP4BlockScaleConfig::scale_dims(hidden, moe_inter);
    const long max_scale_elems = std::max(
        static_cast<long>(gate_up_scale_rows) * gate_up_scale_cols,
        static_cast<long>(down_scale_rows) * down_scale_cols);

    mExpertScalesStaging = mAllocator->allocate(ETensorDType::FP8_E4M3, "expert_scales_staging",
                                                 EAllocationType::ON_DEVICE, {max_scale_elems});
}

template<typename Block>
void FP4WeightProvider<Block>::stream_and_dequantize_expert(
    const FP4BlockQuantizedWeight& src, Tensor& dst, cudaStream_t stream)
{
    // This method handles CPU-offloaded FP4 experts:
    // 1. Copy FP4 packed data from pinned CPU memory to GPU staging buffer
    // 2. Copy FP8 block scales from CPU to GPU staging buffer
    // 3. Dequantize from GPU staging buffer to destination

    // Step 1: Copy FP4 packed data (HOST -> DEVICE)
    const size_t data_bytes = FP4BlockScaleConfig::packed_data_bytes(src.M, src.K);
    cudaMemcpyAsync(mExpertFP4Staging.Data, src.data.Data,
                    data_bytes, cudaMemcpyHostToDevice, stream);

    // Step 2: Copy FP8 block scales
    auto [scale_rows, scale_cols] = FP4BlockScaleConfig::scale_dims(src.M, src.K);
    const size_t scale_bytes = static_cast<size_t>(scale_rows) * scale_cols * sizeof(__nv_fp8_e4m3);
    cudaMemcpyAsync(mExpertScalesStaging.Data, src.block_scales_rowwise.Data,
                    scale_bytes, cudaMemcpyHostToDevice, stream);

    // Step 3: Dequantize from staging buffers to destination
    float global_scale = src.global_decode_scale_rowwise();
    dequantize_fp4_block(
        dst.get<nv_bfloat16>(),
        mExpertFP4Staging.get<uint8_t>(),
        mExpertScalesStaging.get<__nv_fp8_e4m3>(),
        global_scale,
        src.M, src.K,
        mDeviceProps, stream);
}

template<typename Block>
void FP4WeightProvider<Block>::dequantize_selected_experts(
    int layer_idx, const SelectiveExpertInfo& selection_info, cudaStream_t stream)
{
    if (!selection_info.enabled || selection_info.num_active == 0) {
        return;
    }

    const auto& qblock = mFP4Weights->get_moe_block(layer_idx);
    const int hidden = mConfig.hidden_size;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;

    // Check if we can reuse the current buffers
    const bool same_layer_step = (mCurrentLayer == layer_idx) && (mBufferVersion == mStepVersion);

    // IMPORTANT: Also check that we're on the same layer - don't reuse layer 0's experts for layer 1!
    bool selection_matches = same_layer_step &&
                             mCurrentExpertLayer == layer_idx &&
                             mCurrentSelection.enabled &&
                             mCurrentSelection.num_active == selection_info.num_active;
    if (selection_matches) {
        for (int i = 0; i < selection_info.num_active && selection_matches; ++i) {
            if (mCurrentSelection.active_experts[i] != selection_info.active_experts[i]) {
                selection_matches = false;
            }
        }
    }

    if (selection_matches) {
        // Cache hit - buffers already contain the right experts
        return;
    }

    // Dequantize selected experts into COMPACT index positions in the buffer
    const int num_to_dequant = selection_info.num_active;

    for (int i = 0; i < num_to_dequant; ++i) {
        const int global_expert_idx = selection_info.active_experts[i];
        const auto& expert_weights = qblock.experts[global_expert_idx];

        // Create slice views at the COMPACT index position (i, not global_expert_idx)
        Tensor gate_up_slice = Tensor::from_pointer(
            static_cast<std::byte*>(mBatchedExpertGateUp.Data) +
                static_cast<size_t>(i) * (2 * moe_inter) * hidden * sizeof(nv_bfloat16),
            mBatchedExpertGateUp.Device,
            ETensorDType::BF16,
            std::array<long, 2>{2 * moe_inter, hidden}
        );

        Tensor down_slice = Tensor::from_pointer(
            static_cast<std::byte*>(mBatchedExpertDown.Data) +
                static_cast<size_t>(i) * hidden * moe_inter * sizeof(nv_bfloat16),
            mBatchedExpertDown.Device,
            ETensorDType::BF16,
            std::array<long, 2>{hidden, moe_inter}
        );

        // Dequantize this expert's weights into the buffer slice
        if (mExpertsOffloaded) {
            stream_and_dequantize_expert(expert_weights.gate_up_proj, gate_up_slice, stream);
            stream_and_dequantize_expert(expert_weights.down_proj, down_slice, stream);
        } else {
            dequantize_fp4_weight(expert_weights.gate_up_proj, gate_up_slice, stream);
            dequantize_fp4_weight(expert_weights.down_proj, down_slice, stream);
        }
    }

    // Update cache state
    mCurrentSelection = selection_info;
    mCurrentExpertLayer = layer_idx;  // Track which layer these experts are from
    mNumActiveExperts = num_to_dequant;

    // Update the expert weights tensor shapes (compact indexing)
    if constexpr (has_moe_weights<BlockWeights>::value) {
        mDequantBlock.experts.gate_up_proj = Tensor::from_pointer(
            mBatchedExpertGateUp.Data,
            mBatchedExpertGateUp.Device,
            ETensorDType::BF16,
            std::array<long, 3>{(long)num_to_dequant, (long)(2 * moe_inter), (long)hidden}
        );
        mDequantBlock.experts.down_proj = Tensor::from_pointer(
            mBatchedExpertDown.Data,
            mBatchedExpertDown.Device,
            ETensorDType::BF16,
            std::array<long, 3>{(long)num_to_dequant, (long)hidden, (long)moe_inter}
        );
        mDequantBlock.experts.use_batched = true;
        mDequantBlock.experts.num_active_experts = num_to_dequant;
    }
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_FP4_WEIGHT_PROVIDER_H
