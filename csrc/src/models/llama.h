// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string_view>

#include <nlohmann/json_fwd.hpp>

#include "config/pretrained_config.h"
#include "models/registry.h"

namespace models {

class LlamaArchitecture {
public:
    static constexpr std::string_view kHfArchitectureName = "LlamaForCausalLM";

    static PretrainedConfig load_from_hf_config_json(const nlohmann::json& config_json, ETensorDType dtype);
    static void save_to_hf_config_json(const PretrainedConfig& config, nlohmann::json& config_json);
    static std::optional<PretrainedConfig> create_from_preset_name(std::string_view name, ETensorDType dtype);

    static constexpr ArchitectureOps ops() {
        return {
            .hf_architecture_name = kHfArchitectureName,
            .id = PretrainedConfig::LLAMA,
            .load_from_hf_config_json = &LlamaArchitecture::load_from_hf_config_json,
            .save_to_hf_config_json = &LlamaArchitecture::save_to_hf_config_json,
            .create_from_preset_name = &LlamaArchitecture::create_from_preset_name,
        };
    }
};

} // namespace models

