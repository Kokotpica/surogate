// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_IMPORT_EXPORT_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_IMPORT_EXPORT_H

#include <filesystem>
#include <fstream>
#include "lora_model_core.h"

namespace modules {

template<typename Block>
void ModularLoRAModel<Block>::import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    if (qlora_enabled()) {
        import_weights_qlora(file_name, comm);
    } else {
        mBaseModel->import_weights(file_name, allow_cast, comm);
    }
}

template<typename Block>
void ModularLoRAModel<Block>::export_weights(const std::string& file_name, NCCLCommunicator& comm) {
    mBaseModel->export_weights(file_name, comm);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_IMPORT_EXPORT_H
