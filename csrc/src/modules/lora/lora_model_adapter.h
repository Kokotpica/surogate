// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_ADAPTER_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_ADAPTER_H

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "lora_model_core.h"
#include "lora_model_utils.h"
#include "modules/optimizers/adamw_8bit.h"
#include "modules/optimizers/normuon.h"
#include "utilities/safetensors.h"

namespace modules {

namespace detail {

// Helper container to expose 8-bit AdamW optimizer state tensors for checkpointing
class AdamW8BitStateContainer final : public ITensorContainer {
public:
    explicit AdamW8BitStateContainer(LoRAAdamW8BitState* state) : mState(state) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState) return;
        // Check for allocated tensors rather than initialized flag - this allows loading
        // into pre-allocated but not-yet-initialized state buffers
        if (!mState->state1.Data) return;

        // state1/state2 are uint8 buffers stored as BYTE dtype
        callback("lora_adamw8bit.state1", TensorShard(mState->state1));
        callback("lora_adamw8bit.state2", TensorShard(mState->state2));
        callback("lora_adamw8bit.absmax1", TensorShard(mState->absmax1));
        callback("lora_adamw8bit.absmax2", TensorShard(mState->absmax2));
    }

private:
    LoRAAdamW8BitState* mState;
};

// Helper container to expose NorMuon optimizer state tensors for checkpointing
class NorMuonStateContainer final : public ITensorContainer {
public:
    explicit NorMuonStateContainer(LoRANorMuonState* state) : mState(state) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState) return;
        // Check for allocated tensors rather than initialized flag - this allows loading
        // into pre-allocated but not-yet-initialized state buffers
        if (!mState->momentum_state.Data) return;

        callback("lora_normuon.momentum_state", TensorShard(mState->momentum_state));
        callback("lora_normuon.momentum_absmax", TensorShard(mState->momentum_absmax));

        // Variance buffers - use indexed names
        for (size_t i = 0; i < mState->variance_buffers.size(); ++i) {
            callback(fmt::format("lora_normuon.variance_{}", i), TensorShard(mState->variance_buffers[i]));
        }
    }

private:
    LoRANorMuonState* mState;
};

} // namespace detail

template<typename Block>
void ModularLoRAModel<Block>::export_adapter(const std::string& directory, NCCLCommunicator& comm, const std::string& base_model_path) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;
    fs::path dir(directory);
    if (comm.rank() == 0) { fs::create_directories(dir); }
    comm.barrier();
    mLoRAWeights->export_to_file((dir / "adapter_model.safetensors").string(), comm);
    if (comm.rank() == 0) {
        nlohmann::json adapter_config;
        adapter_config["base_model_name_or_path"] = base_model_path;
        adapter_config["peft_type"] = "LORA";
        adapter_config["task_type"] = "CAUSAL_LM";
        adapter_config["r"] = mLoRAConfig.rank;
        adapter_config["lora_alpha"] = mLoRAConfig.alpha;
        adapter_config["lora_dropout"] = mLoRAConfig.dropout;
        adapter_config["fan_in_fan_out"] = false;
        adapter_config["bias"] = "none";
        adapter_config["use_rslora"] = mLoRAConfig.use_rs_lora;
        adapter_config["target_modules"] = detail::targets_to_peft_names(mLoRAConfig);
        std::ofstream config_file(dir / "adapter_config.json");
        config_file << adapter_config.dump(2);
    }
}

template<typename Block>
void ModularLoRAModel<Block>::import_adapter(const std::string& file_name, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    mLoRAWeights->import_from_file(file_name, comm);
}

template<typename Block>
void ModularLoRAModel<Block>::save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;

    // Save LoRA adapter weights
    export_adapter(checkpoint_dir, comm);

    // Save optimizer state based on which optimizer is active
    if (mLoRAAdamW8BitState && mLoRAAdamW8BitState->initialized) {
        detail::AdamW8BitStateContainer container(mLoRAAdamW8BitState.get());
        fs::path opt_file = fs::path(checkpoint_dir) / "lora_optimizer.safetensors";
        write_safetensors(opt_file.string(), container);

        // Save optimizer metadata
        if (comm.rank() == 0) {
            nlohmann::json opt_meta;
            opt_meta["optimizer_type"] = "adamw_8bit";
            opt_meta["total_params"] = mLoRAAdamW8BitState->total_params;
            opt_meta["num_blocks"] = mLoRAAdamW8BitState->num_blocks;
            opt_meta["num_tensors"] = mLoRAAdamW8BitState->num_tensors;
            std::ofstream meta_file(fs::path(checkpoint_dir) / "lora_optimizer.json");
            meta_file << opt_meta.dump(2);
        }
    } else if (mLoRANorMuonState && mLoRANorMuonState->initialized) {
        detail::NorMuonStateContainer container(mLoRANorMuonState.get());
        fs::path opt_file = fs::path(checkpoint_dir) / "lora_optimizer.safetensors";
        write_safetensors(opt_file.string(), container);

        // Save optimizer metadata including variance shapes
        if (comm.rank() == 0) {
            nlohmann::json opt_meta;
            opt_meta["optimizer_type"] = "normuon";
            opt_meta["total_params"] = mLoRANorMuonState->total_params;
            opt_meta["state_elems"] = mLoRANorMuonState->state_elems;
            opt_meta["num_blocks"] = mLoRANorMuonState->num_blocks;
            nlohmann::json shapes = nlohmann::json::array();
            for (const auto& shape : mLoRANorMuonState->variance_shapes) {
                shapes.push_back({shape.first, shape.second});
            }
            opt_meta["variance_shapes"] = shapes;
            std::ofstream meta_file(fs::path(checkpoint_dir) / "lora_optimizer.json");
            meta_file << opt_meta.dump(2);
        }
    }

    comm.barrier();
}

template<typename Block>
void ModularLoRAModel<Block>::load_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;

    // Load LoRA adapter weights
    fs::path adapter_file = fs::path(checkpoint_dir) / "adapter_model.safetensors";
    if (fs::exists(adapter_file)) {
        import_adapter(adapter_file.string(), comm);
    }

    // Load optimizer state if present
    fs::path opt_file = fs::path(checkpoint_dir) / "lora_optimizer.safetensors";
    fs::path opt_meta_file = fs::path(checkpoint_dir) / "lora_optimizer.json";

    if (!fs::exists(opt_file) || !fs::exists(opt_meta_file)) {
        // No optimizer state saved - this is fine for inference or first-time resume
        return;
    }

    // Read optimizer metadata
    std::ifstream meta_stream(opt_meta_file);
    nlohmann::json opt_meta = nlohmann::json::parse(meta_stream);
    std::string optimizer_type = opt_meta["optimizer_type"].get<std::string>();

    if (optimizer_type == "adamw_8bit") {
        if (!mLoRAAdamW8BitState) {
            mLoRAAdamW8BitState = std::make_unique<LoRAAdamW8BitState>();
        }
        auto& state = *mLoRAAdamW8BitState;

        // Restore metadata
        state.total_params = opt_meta["total_params"].get<size_t>();
        state.num_blocks = opt_meta["num_blocks"].get<size_t>();
        state.num_tensors = opt_meta["num_tensors"].get<int>();

        // Allocate buffers if not already done
        if (!state.state1.Data) {
            state.state1 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state1",
                                                 {static_cast<long>(state.total_params)});
            state.state2 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state2",
                                                 {static_cast<long>(state.total_params)});
            state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax1",
                                                  {static_cast<long>(state.num_blocks)});
            state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax2",
                                                  {static_cast<long>(state.num_blocks)});

            // Also allocate quantiles (fixed lookup tables)
            state.quantiles1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles1", {256});
            state.quantiles2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles2", {256});
            std::vector<float> h_quantiles1(256), h_quantiles2(256);
            optimizers::create_adamw8bit_quantiles1(h_quantiles1.data());
            optimizers::create_adamw8bit_quantiles2(h_quantiles2.data());
            CUDA_CHECK(cudaMemcpy(state.quantiles1.Data, h_quantiles1.data(),
                                  256 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state.quantiles2.Data, h_quantiles2.data(),
                                  256 * sizeof(float), cudaMemcpyHostToDevice));
        }

        // Load state tensors from file
        detail::AdamW8BitStateContainer container(&state);
        load_safetensors(opt_file.string(), container, /*allow_cast=*/false);

        // Mark values as restored - runtime buffers will be allocated on first update()
        state.values_restored = true;

    } else if (optimizer_type == "normuon") {
        if (!mLoRANorMuonState) {
            mLoRANorMuonState = std::make_unique<LoRANorMuonState>();
        }
        auto& state = *mLoRANorMuonState;

        // Restore metadata
        state.total_params = opt_meta["total_params"].get<size_t>();
        state.state_elems = opt_meta["state_elems"].get<size_t>();
        state.num_blocks = opt_meta["num_blocks"].get<size_t>();

        // Restore variance shapes
        state.variance_shapes.clear();
        for (const auto& shape : opt_meta["variance_shapes"]) {
            state.variance_shapes.push_back({shape[0].get<int>(), shape[1].get<int>()});
        }

        // Allocate buffers if not already done
        if (!state.momentum_state.Data) {
            state.momentum_quantiles = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_quantiles", {256});
            std::vector<float> h_quantiles(256);
            optimizers::create_normuon_quantiles(h_quantiles.data());
            CUDA_CHECK(cudaMemcpy(state.momentum_quantiles.Data, h_quantiles.data(),
                                  256 * sizeof(float), cudaMemcpyHostToDevice));

            state.momentum_state = mAllocator->allocate(ETensorDType::BYTE, "lora_normuon_momentum",
                                                         {static_cast<long>(state.state_elems)});
            state.momentum_absmax = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_absmax",
                                                          {static_cast<long>(state.num_blocks)});

            // Allocate variance buffers
            for (const auto& shape : state.variance_shapes) {
                int M = shape.first;
                int N = shape.second;
                size_t var_size = optimizers::normuon_variance_buffer_size(M, N);
                Tensor var_buf = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_variance",
                                                       {static_cast<long>(var_size)});
                state.variance_buffers.push_back(std::move(var_buf));
            }

            // Allocate workspace buffers
            state.max_weight_M = 0;
            state.max_weight_N = 0;
            for (const auto& shape : state.variance_shapes) {
                state.max_weight_M = std::max(state.max_weight_M, (size_t)shape.first);
                state.max_weight_N = std::max(state.max_weight_N, (size_t)shape.second);
            }
            size_t max_dim = std::max(state.max_weight_M, state.max_weight_N);
            size_t max_weight_elems = state.max_weight_M * state.max_weight_N;
            size_t polar_workspace_elems = 4 * max_dim * max_dim + 1;
            size_t polar_size = max_weight_elems + polar_workspace_elems;
            state.polar_workspace = mAllocator->allocate(ETensorDType::BF16, "lora_normuon_polar",
                                                          {static_cast<long>(polar_size)});
            state.momentum_temp = mAllocator->allocate(ETensorDType::BF16, "lora_normuon_temp",
                                                        {static_cast<long>(max_weight_elems)});

            CUBLAS_CHECK(cublasCreate(&state.cublas_handle));
        }

        // Load state tensors from file
        detail::NorMuonStateContainer container(&state);
        load_safetensors(opt_file.string(), container, /*allow_cast=*/false);

        // Mark values as restored
        state.values_restored = true;
    }

    comm.barrier();
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_ADAPTER_H
