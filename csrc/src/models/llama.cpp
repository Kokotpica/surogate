// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "models/llama.h"

#include <nlohmann/json.hpp>

#include <fmt/core.h>

#include "utilities/utils.h"

namespace models {

static PretrainedConfig create_llama2_config(int hidden_size, int intermediate_size, int heads, int depth, ETensorDType dtype) {
    return {
        .Architecture = PretrainedConfig::LLAMA,
        .BosTokenId = 1,
        .EosTokenId = 2,
        .PadTokenId = 0,
        .HiddenSize = hidden_size,
        .IntermediateSize = intermediate_size,
        .VocabSize = 32000,
        .NumQueryHeads = heads,
        .NumKeyValHeads = heads,
        .NumLayers = depth,
        .HeadDim = 0,
        .MaxPositionEmbeddings = 4096,
        .RopeTheta = 10000.f,
        .RmsNormEps = 1e-05f,
        .TiedWordEmbeddings = false,
        .UseQKVBias = false,
        .UseQKNorm = false,
        .DType = dtype
    };
}

static PretrainedConfig create_llama3_config(int hidden_size, int intermediate_size, int q_heads, int kv_heads, int depth,
                                             ETensorDType dtype) {
    return {
        .Architecture = PretrainedConfig::LLAMA,
        .BosTokenId = 128000,
        .EosTokenId = 128001,
        .PadTokenId = 128255,
        .HiddenSize = hidden_size,
        .IntermediateSize = intermediate_size,
        .VocabSize = 128256,
        .NumQueryHeads = q_heads,
        .NumKeyValHeads = kv_heads,
        .NumLayers = depth,
        .HeadDim = 0,
        .MaxPositionEmbeddings = 4096,
        .RopeTheta = 500000.f,
        .RmsNormEps = 1e-05f,
        .TiedWordEmbeddings = false,
        .UseQKVBias = false,
        .UseQKNorm = false,
        .DType = dtype
    };
}

PretrainedConfig LlamaArchitecture::load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype) {
    PretrainedConfig result;
    result.Architecture = PretrainedConfig::LLAMA;
    result.DType = dtype;

    result.BosTokenId = config_json.at("bos_token_id").get<int>();
    result.EosTokenId = config_json.at("eos_token_id").get<int>();
    result.PadTokenId = config_json.value("pad_token_id", 0);

    result.HiddenSize = config_json.at("hidden_size").get<int>();
    result.IntermediateSize = config_json.at("intermediate_size").get<int>();
    result.VocabSize = config_json.at("vocab_size").get<int>();
    result.NumQueryHeads = config_json.at("num_attention_heads").get<int>();
    result.NumKeyValHeads = config_json.at("num_key_value_heads").get<int>();
    result.NumLayers = config_json.at("num_hidden_layers").get<int>();
    result.HeadDim = config_json.value("head_dim", 0);

    result.MaxPositionEmbeddings = config_json.at("max_position_embeddings").get<int>();
    result.RopeTheta = config_json.value("rope_theta", 10000.0f);
    result.TiedWordEmbeddings = config_json.value("tie_word_embeddings", false);
    result.RmsNormEps = config_json.value("rms_norm_eps", 1e-05f);

    result.UseQKVBias = config_json.value("attention_bias", false);
    result.UseQKNorm = false;

    return result;
}

void LlamaArchitecture::save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json) {
    config_json["architectures"] = {std::string(kHfArchitectureName)};
    config_json["model_type"] = "llama";
    config_json["bos_token_id"] = config.BosTokenId;
    config_json["eos_token_id"] = config.EosTokenId;
    config_json["pad_token_id"] = config.PadTokenId;
    config_json["hidden_size"] = config.HiddenSize;
    config_json["intermediate_size"] = config.IntermediateSize;
    config_json["vocab_size"] = config.VocabSize;
    config_json["num_attention_heads"] = config.NumQueryHeads;
    config_json["num_key_value_heads"] = config.NumKeyValHeads;
    config_json["num_hidden_layers"] = config.NumLayers;
    config_json["max_position_embeddings"] = config.MaxPositionEmbeddings;
    config_json["rope_theta"] = config.RopeTheta;
    config_json["rms_norm_eps"] = config.RmsNormEps;
    config_json["tie_word_embeddings"] = config.TiedWordEmbeddings;
    config_json["attention_bias"] = false;
    config_json["mlp_bias"] = false;
    config_json["torch_dtype"] = dtype_to_torch_str(config.DType);
}

std::optional<PretrainedConfig> LlamaArchitecture::create_from_preset_name(std::string_view name, ETensorDType dtype) {
    if (iequals(name, "llama-2-7b")) {
        return create_llama2_config(4096, 11008, 32, 32, dtype);
    }
    if (iequals(name, "llama-2-13b")) {
        return create_llama2_config(5120, 13824, 40, 40, dtype);
    }
    if (iequals(name, "llama-3-8b")) {
        return create_llama3_config(4096, 14336, 32, 8, 32, dtype);
    }
    return std::nullopt;
}

} // namespace models

