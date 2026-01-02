// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILITIES_NVML_H
#define SUROGATE_SRC_UTILITIES_NVML_H

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct GPUUtilInfo {
    unsigned int clock;
    unsigned int max_clock;
    unsigned int power;
    unsigned int power_limit;
    unsigned int fan;
    unsigned int temperature;
    unsigned int temp_slowdown;

    std::size_t mem_free;
    std::size_t mem_total;
    std::size_t mem_reserved;
    std::size_t mem_used;

    float gpu_utilization;
    float mem_utilization;
    const char* throttle_reason;

    std::size_t pcie_rx;    // in bytes/µs
    std::size_t pcie_tx;
};

std::size_t get_mem_reserved();
std::string get_gpu_name();
//! Sets the CPU affinity of the calling thread to be optimal for the current device
//! Returns true if successful
bool set_cpu_affinity();

class IGPUUtilTracker {
public:
    IGPUUtilTracker() = default;
    virtual ~IGPUUtilTracker() = default;
    virtual const GPUUtilInfo& update() = 0;

    static std::unique_ptr<IGPUUtilTracker> create();
};

struct GPUInfo {
    int device_id;
    std::string name;
    std::size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
};

class SystemInfo {
public:
    static int get_cuda_driver_version();
    static int get_cuda_runtime_version();
    static int get_nccl_version();
    static int get_cudnn_version();
    static std::vector<GPUInfo> get_gpu_info();
};

#endif //SUROGATE_SRC_UTILITIES_NVML_H
