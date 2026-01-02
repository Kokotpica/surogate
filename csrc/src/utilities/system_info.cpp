// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_info.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <nvml.h>
#include <nccl.h>
#include <cudnn.h>

#include <vector>
#include <stdexcept>

int SystemInfo::get_cuda_driver_version() {
    int driver_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    return driver_version;
}

int SystemInfo::get_cuda_runtime_version() {
    int runtime_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    return runtime_version;
}

int SystemInfo::get_nccl_version() {
    int nccl_version = 0;
    ncclResult_t result = ncclGetVersion(&nccl_version);
    if (result != ncclSuccess) {
        throw std::runtime_error("Failed to get NCCL version");
    }
    return nccl_version;
}

int SystemInfo::get_cudnn_version() {
    return static_cast<int>(cudnnGetVersion());
}

std::vector<GPUInfo> SystemInfo::get_gpu_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::vector<GPUInfo> gpu_infos;
    gpu_infos.reserve(device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        GPUInfo info;
        info.device_id = i;
        info.name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;

        gpu_infos.push_back(std::move(info));
    }

    return gpu_infos;
}
