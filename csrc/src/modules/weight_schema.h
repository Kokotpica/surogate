// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_WEIGHT_SCHEMA_H
#define SUROGATE_SRC_MODULES_WEIGHT_SCHEMA_H

#include <functional>
#include <string>
#include <vector>

#include "utilities/dtype.h"
#include "utilities/tensor.h"

namespace modules {

/**
 * @brief Describes a single tensor in a module's weight schema
 *
 * Used by weight managers to allocate and organize tensors according to
 * what a module requires. The schema is declarative - modules describe
 * their requirements, managers handle the actual storage.
 */
struct WeightDescriptor {
    std::string name;                   ///< Unique name within the module (e.g., "weight", "bias")
    std::vector<long> shape;            ///< Shape of the tensor
    ETensorDType dtype;                 ///< Data type
    bool requires_grad = true;          ///< Whether this weight needs gradients
    bool is_matrix = true;              ///< Matrices get weight decay in AdamW
    bool is_optional = false;           ///< Whether this weight may be absent

    // Size calculations
    [[nodiscard]] std::size_t numel() const {
        std::size_t n = 1;
        for (long dim : shape) n *= dim;
        return n;
    }

    [[nodiscard]] std::size_t bytes() const {
        return numel() * get_dtype_size(dtype);
    }
};

/**
 * @brief Complete weight schema for a module
 *
 * Contains descriptors for all weights in a module, plus metadata
 * about the module type.
 */
struct ModuleWeightSchema {
    std::string module_type;            ///< Type name (e.g., "Linear", "Attention")
    std::vector<WeightDescriptor> weights;

    // Helper to find a weight by name
    [[nodiscard]] const WeightDescriptor* find(const std::string& name) const {
        for (const auto& w : weights) {
            if (w.name == name) return &w;
        }
        return nullptr;
    }

    // Total bytes required
    [[nodiscard]] std::size_t total_bytes() const {
        std::size_t total = 0;
        for (const auto& w : weights) {
            if (!w.is_optional) total += w.bytes();
        }
        return total;
    }
};

/**
 * @brief Trait to extract weight schema from a module type
 *
 * Specialize this template for each module type to provide its schema.
 * The schema is used by weight managers to allocate appropriate storage.
 *
 * Usage:
 * @code
 * template<>
 * struct ModuleWeightSchemaTrait<LinearModule> {
 *     static ModuleWeightSchema schema(const LinearModule::Config& config) {
 *         ModuleWeightSchema s;
 *         s.module_type = "Linear";
 *         s.weights.push_back({
 *             .name = "weight",
 *             .shape = {config.out_features, config.in_features},
 *             .dtype = ETensorDType::BF16,
 *             .requires_grad = true,
 *             .is_matrix = true
 *         });
 *         if (config.has_bias) {
 *             s.weights.push_back({
 *                 .name = "bias",
 *                 .shape = {config.out_features},
 *                 .dtype = ETensorDType::BF16,
 *                 .requires_grad = true,
 *                 .is_matrix = false
 *             });
 *         }
 *         return s;
 *     }
 * };
 * @endcode
 */
template<typename Module>
struct ModuleWeightSchemaTrait {
    // Default: no schema (must be specialized)
    static ModuleWeightSchema schema(const typename Module::Config&) {
        return {};
    }
};

/**
 * @brief Schema for a complete transformer block's weights
 *
 * Aggregates schemas from all sub-modules in a block.
 */
struct BlockWeightSchema {
    ModuleWeightSchema ln1;
    ModuleWeightSchema attention;
    ModuleWeightSchema ln2;
    ModuleWeightSchema mlp_up;
    ModuleWeightSchema mlp_down;
    // For MoE: router and experts would go here

    [[nodiscard]] std::size_t total_bytes() const {
        return ln1.total_bytes() + attention.total_bytes() +
               ln2.total_bytes() + mlp_up.total_bytes() + mlp_down.total_bytes();
    }
};

/**
 * @brief Schema for non-block weights (embeddings, final norm, lm head)
 */
struct NonBlockWeightSchema {
    ModuleWeightSchema embeddings;
    ModuleWeightSchema lm_head;
    ModuleWeightSchema final_norm;

    [[nodiscard]] std::size_t total_bytes() const {
        return embeddings.total_bytes() + lm_head.total_bytes() + final_norm.total_bytes();
    }
};

/**
 * @brief Complete model weight schema
 */
struct ModelWeightSchema {
    NonBlockWeightSchema non_block;
    BlockWeightSchema block_template;  ///< Schema for each block (all blocks are identical)
    int num_blocks;

    [[nodiscard]] std::size_t total_bytes() const {
        return non_block.total_bytes() + block_template.total_bytes() * num_blocks;
    }
};

/**
 * @brief Mapping function from HuggingFace tensor names to our schema
 *
 * Used during weight import to match external tensor names to internal schema.
 */
struct WeightNameMapping {
    using MapFn = std::function<std::string(const std::string& hf_name)>;

    MapFn hf_to_internal;  ///< HuggingFace name -> internal name
    MapFn internal_to_hf;  ///< Internal name -> HuggingFace name

    // Default LLaMA-style mapping
    static WeightNameMapping llama_mapping();

    // Qwen-style mapping (slightly different names)
    static WeightNameMapping qwen_mapping();
};

// ============================================================================
// Specializations for built-in modules
// ============================================================================

// Forward declarations
class LinearModule;
class RMSNormModule;
class AttentionModule;
class SwiGLUModule;
class EmbeddingModule;
class LMHeadModule;

template<>
struct ModuleWeightSchemaTrait<LinearModule> {
    static ModuleWeightSchema schema(const struct LinearModuleConfig& config, ETensorDType dtype);
};

template<>
struct ModuleWeightSchemaTrait<RMSNormModule> {
    static ModuleWeightSchema schema(const struct RMSNormModuleConfig& config, ETensorDType dtype);
};

template<>
struct ModuleWeightSchemaTrait<AttentionModule> {
    static ModuleWeightSchema schema(const struct AttentionModuleConfig& config, ETensorDType dtype);
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_WEIGHT_SCHEMA_H
