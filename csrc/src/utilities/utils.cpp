// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.h"

#include <algorithm>

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>

/**
 * @brief Throws a cuda_error if a CUDA runtime call failed.
 *
 * Builds a detailed error message containing file/line, the failed statement,
 * the CUDA error name, and the CUDA error string. Also clears the current CUDA
 * error state via cudaGetLastError() before throwing to avoid leaking stale
 * errors into exception handlers.
 *
 * @param status The CUDA runtime status returned by the evaluated statement.
 * @param statement The stringified CUDA statement/expression that produced @p status.
 * @param file Source file where the error occurred.
 * @param line Source line where the error occurred.
 *
 * @throws cuda_error If @p status != cudaSuccess.
 */
void cuda_throw_on_error(cudaError_t status, const char* statement, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("Cuda Error in ") + file + ":" + std::to_string(line) + " (" + std::string(statement) + "): " + cudaGetErrorName(status) + ": ";
        msg += cudaGetErrorString(status);
        // make sure we have a clean cuda error state before launching the exception
        // otherwise, if there are calls to the CUDA API in the exception handler,
        // they will return the old error.
        [[maybe_unused]] cudaError_t clear_error = cudaGetLastError();
        throw cuda_error(status, msg);
    }
}

/**
 * @brief Begins an NVTX range using a C-string label.
 *
 * Pushes an NVTX range on construction. Intended for RAII-style profiling scopes.
 *
 * @param s Null-terminated label for the NVTX range (must remain valid for the call).
 *
 * @note This constructor is noexcept; it assumes nvtxRangePush does not throw.
 */
NvtxRange::NvtxRange(const char* s) noexcept { nvtxRangePush(s); }

/**
 * @brief Begins an NVTX range labeled as "<base_str> <number>".
 *
 * Concatenates the base string and number, then pushes the resulting label
 * as an NVTX range.
 *
 * @param base_str Base label prefix.
 * @param number Suffix number appended after a space.
 */
NvtxRange::NvtxRange(const std::string& base_str, int number) {
    std::string range_string = base_str + " " + std::to_string(number);
    nvtxRangePush(range_string.c_str());
}

/**
 * @brief Ends the current NVTX range.
 *
 * Pops the NVTX range that was pushed by the constructor.
 *
 * @note This destructor is noexcept; it assumes nvtxRangePop does not throw.
 */
NvtxRange::~NvtxRange() noexcept { nvtxRangePop(); }

/**
 * @brief Creates a CUDA stream and assigns it an NVTX name.
 *
 * @param name Null-terminated stream name used for NVTX visualization.
 * @return Newly created CUDA stream handle.
 *
 * @throws cuda_error If cudaStreamCreate fails (via CUDA_CHECK).
 *
 * @note The caller owns the returned stream and must destroy it with cudaStreamDestroy().
 */
cudaStream_t create_named_stream(const char* name) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    nvtxNameCudaStreamA(stream, name);
    return stream;
}

/**
 * @brief Creates a CUDA event and assigns it an NVTX name.
 *
 * @param name Null-terminated event name used for NVTX visualization.
 * @param timing If true, enable timing (cudaEventDefault). If false, disable timing
 *               (cudaEventDisableTiming) to reduce overhead when timing is not needed.
 * @return Newly created CUDA event handle.
 *
 * @throws cuda_error If cudaEventCreateWithFlags fails (via CUDA_CHECK).
 *
 * @note The caller owns the returned event and must destroy it with cudaEventDestroy().
 */
cudaEvent_t create_named_event(const char* name, bool timing) {
    cudaEvent_t event;
    int flags = timing ? cudaEventDefault : cudaEventDisableTiming;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
    nvtxNameCudaEventA(event, name);
    return event;
}

/**
 * @brief Case-insensitive equality comparison for two string views (ASCII-ish semantics).
 *
 * Compares the two views element-wise after converting each byte to lowercase via
 * std::tolower on an unsigned-char widened value.
 *
 * @param lhs Left-hand string view.
 * @param rhs Right-hand string view.
 * @return True if both views have the same length and match case-insensitively; false otherwise.
 *
 * @note Locale-dependent behavior may apply due to std::tolower; intended for basic
 *       case-insensitive comparisons on typical ASCII identifiers.
 */
bool iequals(std::string_view lhs, std::string_view rhs) {
    return std::ranges::equal(
        lhs, rhs, [](unsigned char a, unsigned char b) {
            return std::tolower(a) == std::tolower(b);
    });
}

/**
 * @brief Throws a std::runtime_error indicating a non-divisible division attempt.
 *
 * @param dividend The dividend that could not be evenly divided.
 * @param divisor The divisor used in the attempted division.
 *
 * @throws std::runtime_error Always throws with a formatted error message.
 */
[[noreturn]] void throw_not_divisible(long long dividend, long long divisor) {
    throw std::runtime_error(fmt::format("Cannot divide {} by {}", dividend, divisor));
}
