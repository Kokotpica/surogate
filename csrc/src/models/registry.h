// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "config/pretrained_config.h"

namespace models {

struct ArchitectureOps {
    std::string_view hf_architecture_name;
    PretrainedConfig::ArchitectureId id;

    PretrainedConfig (*load_from_hf_config_json)(const nlohmann::json& config_json, ETensorDType dtype);
    void (*save_to_hf_config_json)(const PretrainedConfig& config, nlohmann::json& config_json);

    // Optional: create a pretrained config from a from-scratch preset name.
    std::optional<PretrainedConfig> (*create_from_preset_name)(std::string_view name, ETensorDType dtype);
};

const ArchitectureOps& architecture_from_hf_name(std::string_view hf_architecture_name);
const ArchitectureOps& architecture_from_id(PretrainedConfig::ArchitectureId id);

// Try all registered architectures for a from-scratch preset name.
std::optional<PretrainedConfig> create_from_preset_name(std::string_view name, ETensorDType dtype);

std::vector<std::string_view> supported_hf_architectures();

} // namespace models

