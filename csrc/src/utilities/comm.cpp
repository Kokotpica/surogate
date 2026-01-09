// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "comm.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <variant>
#include <future>

#include <nccl.h>
#include <fmt/core.h>

#include "gpu_info.h"
#include "kernels/kernels.h"
#include "tensor.h"
#include "utils.h"

/**
 * @brief Throws a std::runtime_error if an NCCL call returned an error.
 *
 * @param status NCCL status code returned by an NCCL API call.
 * @param file Source file where the failing call was made.
 * @param line Source line where the failing call was made.
 *
 * @throws std::runtime_error Always thrown when @p status != ncclSuccess.
 */
void nccl_check(ncclResult_t status, const char* file, int line) {
    if (status != ncclSuccess) {
        throw std::runtime_error(fmt::format("NCCL error at {}:{}: {}", file, line, ncclGetErrorString(status)));
    }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

struct NCCLCommunicator::CommandBuffer
{
    struct Gather {
        std::byte* Src;
        std::byte* Dst;
        std::size_t Bytes;
    };

    struct ScatterReduce {
        ETensorDType DType;
        std::byte* Tensor;
        std::size_t Elements;
    };


    struct Send {
        const std::byte* Tensor;
        std::size_t Bytes;
        int Target;
    };

    struct Recv {
        std::byte* Tensor;
        std::size_t Bytes;
        int Source;
    };

    std::vector<std::variant<Gather, ScatterReduce, Send, Recv>> Commands;
    cudaEvent_t Ready = nullptr;
};

/**
 * @brief Construct an NCCLCommunicator for a given rank and world size.
 *
 * Sets the CUDA device to @p rank, initializes the NCCL communicator using @p nccl_id,
 * and creates a dedicated comms stream and sync event on that device.
 *
 * @param rank Local rank (also used as CUDA device index).
 * @param world Total number of ranks in the communicator.
 * @param nccl_id Pointer to an ncclUniqueId shared across all ranks.
 *
 * @throws std::runtime_error If NCCL initialization fails.
 */
NCCLCommunicator::NCCLCommunicator(int rank, int world, const void* nccl_id) :
    mRank(rank), mWorld(world), mNcclComm(nullptr), mCmdBuf(std::make_unique<CommandBuffer>())
{
    CUDA_CHECK(cudaSetDevice(mRank));
    ncclCheck(ncclCommInitRank(&mNcclComm, mWorld, *reinterpret_cast<const ncclUniqueId*>(nccl_id), mRank));

    // must be created _after_ we set the device
    mCommsStream = create_named_stream("nccl_stream");
    mCommsSync = create_named_event("nccl_sync");  // todo disable timing for max perf
}

#include <pthread.h>

/**
 * @brief Destructor that attempts to terminate NCCL without hanging the main thread.
 *
 * NCCL finalization can hang (notably with Python bindings). To localize hangs, NCCL teardown
 * is attempted in a helper thread with a timeout; on timeout, the future is intentionally leaked.
 *
 * Always destroys the internal CUDA event/stream (best-effort; never throws).
 */
NCCLCommunicator::~NCCLCommunicator() {
    // When used with the python bindings, ncclCommFinalize() can hang forever;
    // I haven't found a fix, so here we just make sure that the hang gets localized
    // to a helper thread (which we leak, but generally ~NCCLCommunicator is expected
    // to run at program shutdown anyway)
    auto terminate_future = std::async(std::launch::async, [this]() {
        this->terminate_nccl();
    });

    if (terminate_future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
        fprintf(stderr, "NCCL termination timed out, detaching\n");
        // this *will* leak resources, but at least we're not hanging forever
        new auto(std::move(terminate_future));
    }
    // Destructor must not throw, especially during stack unwinding.
    // Best-effort cleanup: report CUDA errors and continue.
    if (mCommsSync) {
        const cudaError_t st = cudaEventDestroy(mCommsSync);
        if (st != cudaSuccess) {
            fprintf(stderr, "WARNING: cudaEventDestroy(nccl_sync) failed: %s\n", cudaGetErrorString(st));
            fflush(stderr);
            (void)cudaGetLastError();
        }
        mCommsSync = nullptr;
    }
    if (mCommsStream) {
        const cudaError_t st = cudaStreamDestroy(mCommsStream);
        if (st != cudaSuccess) {
            fprintf(stderr, "WARNING: cudaStreamDestroy(nccl_stream) failed: %s\n", cudaGetErrorString(st));
            fflush(stderr);
            (void)cudaGetLastError();
        }
        mCommsStream = nullptr;
    }
}

/**
 * @brief Performs NCCL teardown (finalize/destroy or abort depending on state).
 *
 * If no exception is active and NCCL reports no async error, performs a "nice" shutdown:
 * synchronize streams/devices, finalize, then destroy communicator.
 * Otherwise aborts the communicator.
 *
 * @note Intended to be called from a helper thread in the destructor to avoid process-wide hangs.
 */
void NCCLCommunicator::terminate_nccl() {
    ncclResult_t result;
    ncclCheck(ncclCommGetAsyncError(mNcclComm, &result));
    // do "nice" shutdown if we're in a good state,
    // just abort if there is something weird going on.
    if (std::uncaught_exceptions() == 0 && result == ncclSuccess) {
        CUDA_CHECK(cudaStreamSynchronize(mCommsStream));
        CUDA_CHECK(cudaDeviceSynchronize());
        ncclCheck(ncclCommFinalize(mNcclComm));
        ncclCheck(ncclCommDestroy(mNcclComm));
    } else {
        ncclCheck(ncclCommAbort(mNcclComm));
    }
}

/**
 * @brief Begin a transaction by specifying an event that indicates inputs are ready.
 *
 * @param ready CUDA event that must be completed before communication work may start.
 *
 * @throws std::runtime_error If the internal command buffer is not empty.
 */
void NCCLCommunicator::begin_transaction(cudaEvent_t ready) {
    if (!mCmdBuf->Commands.empty()) {
        throw std::runtime_error("start_comms: Buffer not empty");
    }
    mCmdBuf->Ready = ready;
}

/**
 * @brief Begin a transaction by recording a readiness event on a given stream.
 *
 * Records an internal event on @p wait_for_stream and uses it as the transaction "ready" marker.
 *
 * @param wait_for_stream Stream on which preceding compute producing the communication inputs was enqueued.
 */
void NCCLCommunicator::begin_transaction(cudaStream_t wait_for_stream) {
    CUDA_CHECK(cudaEventRecord(mCommsSync, wait_for_stream));
    begin_transaction(mCommsSync);
}

/**
 * @brief Visitor that executes buffered communication commands by dispatching to NCCLCommunicator methods.
 */
struct NCCLCommunicator::CommandVisitor {
    NCCLCommunicator* Comm;

    /**
     * @brief Execute a Gather command via NCCL all-gather (or derived override).
     * @param cmd Gather command containing src/dst pointers and byte size.
     */
    void operator()(CommandBuffer::Gather& cmd) const {
        Comm->gather_weight(cmd.Src, cmd.Dst, cmd.Bytes);
    }

    /**
     * @brief Execute a ScatterReduce command via NCCL reduce-scatter.
     * @param cmd ScatterReduce command containing dtype, tensor pointer, and element count.
     *
     * @throws std::runtime_error If @p cmd.DType is not supported.
     */
    void operator()(CommandBuffer::ScatterReduce& cmd) const {
        switch (cmd.DType) {
        case ETensorDType::FP32:
            Comm->scatter_grad(reinterpret_cast<float*>(cmd.Tensor), cmd.Elements);
            break;
        case ETensorDType::BF16:
            Comm->scatter_grad(reinterpret_cast<nv_bfloat16*>(cmd.Tensor), cmd.Elements);
            break;
        default:
            throw std::runtime_error("scatter: Unsupported dtype");
        }
    }

    /**
     * @brief Execute a Send command (point-to-point send).
     * @param cmd Send command containing source pointer, byte size, and target rank.
     */
    void operator()(CommandBuffer::Send& cmd) const {
        Comm->send(cmd.Tensor, cmd.Target, cmd.Bytes);
    }

    /**
     * @brief Execute a Recv command (point-to-point receive).
     * @param cmd Recv command containing destination pointer, byte size, and source rank.
     */
    void operator()(CommandBuffer::Recv& cmd) const {
        Comm->recv(cmd.Tensor, cmd.Source, cmd.Bytes);
    }

};

/**
 * @brief Execute all scheduled commands in the current transaction and signal completion.
 *
 * Calls the transaction hooks (on_execute_transaction / on_finish_transaction), executes each buffered command,
 * and performs launch-queue throttling syncs before and after to avoid multi-rank deadlocks.
 *
 * Launch Queue Throttling Strategy:
 * - BEFORE transaction: Ensures all ranks are ready to begin enqueuing collective operations together.
 *   This prevents a fast rank from enqueueing its collectives while slower ranks are still processing
 *   previous work, which could lead to queue exhaustion before all ranks reach the collective barrier.
 *
 * - AFTER transaction: Ensures all ranks have completed enqueuing the collective operations before any
 *   rank continues with subsequent work. This prevents a fast rank from filling the launch queue with
 *   post-transaction kernels, which would block slower ranks from enqueuing future collectives.
 *
 * Note: In multi-process mode (MPI), _launch_queue_throttle_sync() is a no-op because each process
 * has its own independent launch queue. This mechanism only applies to multi-threaded mode where
 * all GPU worker threads share a single per-process CUDA launch queue.
 *
 * Optimization: Throttling is only applied when the transaction contains NCCL collective operations
 * (ScatterReduce or Gather), since only these operations have implicit global barriers that can
 * cause deadlocks. Point-to-point operations (Send/Recv) don't require throttling.
 *
 * @param signal CUDA event that will be recorded on the comms stream to signal completion of the transaction.
 *
 * @throws std::runtime_error Propagates errors from command execution or hooks.
 */
void NCCLCommunicator::execute_transaction(cudaEvent_t signal) {
    // Check if this transaction contains NCCL collective operations that require throttling
    bool has_collectives = std::any_of(mCmdBuf->Commands.begin(), mCmdBuf->Commands.end(),
        [](const auto& cmd) {
            return std::holds_alternative<CommandBuffer::ScatterReduce>(cmd) ||
                   std::holds_alternative<CommandBuffer::Gather>(cmd);
        });

    // Synchronize CPU threads before enqueuing collective operations
    if (has_collectives) {
        _launch_queue_throttle_sync();
    }

    on_execute_transaction(*mCmdBuf);

    CommandVisitor visitor{this};
    for (auto& cmd: mCmdBuf->Commands) {
        std::visit(visitor, cmd);
    }

    on_finish_transaction(signal);

    // Synchronize CPU threads after enqueuing collective operations
    if (has_collectives) {
        _launch_queue_throttle_sync();
    }

    mCmdBuf->Commands.clear();
}

/**
 * @brief Schedule an in-place reduce-scatter of gradients for a tensor.
 *
 * @param tensor Tensor whose device buffer is reduced across ranks and scattered so each rank keeps its shard.
 *
 * @throws std::runtime_error If @p tensor.Data is null.
 */
void NCCLCommunicator::schedule_reduce_scatter(Tensor& tensor) {
    if (tensor.Data == nullptr) {
        throw std::runtime_error("scatter: Source tensor is null");
    }

    mCmdBuf->Commands.emplace_back(CommandBuffer::ScatterReduce{.DType = tensor.DType, .Tensor = tensor.Data, .Elements = tensor.nelem()});
}

/**
 * @brief Schedule an all-gather of a sharded tensor into a full target tensor.
 *
 * @param src Source shard (device pointer and dtype describe the local shard).
 * @param tgt Target full tensor receiving concatenated shards (device pointer must be valid).
 *
 * @throws std::runtime_error If source/target pointers are null or dtypes mismatch.
 */
void NCCLCommunicator::schedule_all_gather(const TensorShard& src, Tensor& tgt) {
    if (src.Data == nullptr) {
        throw std::runtime_error("gather: Source tensor is null");
    }

    if (tgt.Data == nullptr) {
        throw std::runtime_error("gather: Target tensor is null");
    }

    if (src.DType != tgt.DType) {
        throw std::runtime_error("gather: Mismatched dtypes");
    }

    mCmdBuf->Commands.emplace_back(CommandBuffer::Gather{.Src = src.Data, .Dst = tgt.Data, .Bytes = tgt.bytes()});
}

/**
 * @brief All-reduce a single scalar loss value across ranks using average.
 *
 * @param loss Device pointer to a single float (input/output in-place).
 * @param stream CUDA stream to enqueue the NCCL all-reduce on.
 */
void NCCLCommunicator::reduce_loss(float* loss, cudaStream_t stream) {
    ncclCheck(ncclAllReduce(loss, loss, 1, ncclFloat, ncclAvg, mNcclComm, stream));
}

/**
 * @brief All-reduce an array of floats across ranks using maximum.
 *
 * @param values Device pointer to @p n floats (input/output in-place).
 * @param n Number of float elements.
 * @param stream CUDA stream to enqueue on; if null, uses the internal comms stream.
 */
void NCCLCommunicator::reduce_max(float* values, int n, cudaStream_t stream) {
    ncclCheck(ncclAllReduce(values, values, n, ncclFloat, ncclMax, mNcclComm, stream ? stream : mCommsStream));
}

/**
 * @brief All-reduce a single scalar value across ranks using sum (e.g., norm-squared accumulation).
 *
 * @param norm_squared Device pointer to a single float (input/output in-place).
 * @param stream CUDA stream to enqueue the NCCL all-reduce on.
 */
void NCCLCommunicator::reduce_norm(float* norm_squared, cudaStream_t stream) {
    ncclCheck(ncclAllReduce(norm_squared, norm_squared, 1, ncclFloat, ncclSum, mNcclComm, stream));
}

/**
 * @brief All-reduce INT32 values across ranks using sum.
 *
 * Used for aggregating counters like valid-token counts when masking is enabled.
 *
 * @param values Device pointer to @p n int32 values (input/output in-place).
 * @param n Number of int32 elements.
 * @param stream CUDA stream to enqueue the NCCL all-reduce on.
 */
void NCCLCommunicator::all_reduce_sum_int(int* values, int n, cudaStream_t stream) {
    if (mWorld == 1) return;
    ncclCheck(ncclAllReduce(values, values, n, ncclInt32, ncclSum, mNcclComm, stream));
}

/**
 * @brief All-reduce tensor data in-place using average.
 *
 * Used for gradient averaging in data parallelism (e.g., LoRA training).
 * Supports FP32 and BF16 tensors.
 *
 * @param tensor Tensor to all-reduce in-place (must be on device).
 * @param stream CUDA stream to enqueue the NCCL all-reduce on.
 *
 * @throws std::runtime_error if tensor dtype is not supported.
 */
void NCCLCommunicator::all_reduce_avg(Tensor& tensor, cudaStream_t stream) {
    if (mWorld == 1) return;  // No-op for single GPU

    ncclDataType_t nccl_dtype;
    switch (tensor.DType) {
        case ETensorDType::FP32:
            nccl_dtype = ncclFloat;
            break;
        case ETensorDType::BF16:
            nccl_dtype = ncclBfloat16;
            break;
        case ETensorDType::FP16:
            nccl_dtype = ncclFloat16;
            break;
        default:
            throw std::runtime_error(fmt::format(
                "NCCLCommunicator::all_reduce_avg: unsupported tensor dtype {}",
                dtype_to_str(tensor.DType)));
    }

    ncclCheck(ncclAllReduce(tensor.Data, tensor.Data, tensor.nelem(), nccl_dtype, ncclAvg, mNcclComm, stream));
}

/**
 * @brief Reduce-scatter FP32 gradients across ranks using average.
 *
 * Input is interpreted as a full buffer of @p size elements; output shard is written to the local shard region.
 *
 * @param value Device pointer to the full buffer (input) and shard region (output).
 * @param size Total number of float elements in the full buffer (must be divisible by world size).
 */
void NCCLCommunicator::scatter_grad(float* value, std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    ptrdiff_t shard_offset = (ptrdiff_t)shard_size * mRank;
    ncclCheck(ncclReduceScatter(
        value, value + shard_offset,
        shard_size,
        ncclFloat, ncclAvg,
        mNcclComm, mCommsStream
    ));
}

/**
 * @brief Reduce-scatter BF16 gradients across ranks using average.
 *
 * Input is interpreted as a full buffer of @p size elements; output shard is written to the local shard region.
 *
 * @param value Device pointer to the full buffer (input) and shard region (output).
 * @param size Total number of BF16 elements in the full buffer (must be divisible by world size).
 */
void NCCLCommunicator::scatter_grad(nv_bfloat16* value, std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    ptrdiff_t shard_offset = (ptrdiff_t)shard_size * mRank;
    ncclCheck(ncclReduceScatter(
        value, value + shard_offset,
        shard_size,
        ncclBfloat16, ncclAvg,
        mNcclComm, mCommsStream
    ));
}

/**
 * @brief All-gather a sharded weight buffer into a full buffer.
 *
 * @param src Device pointer to the local shard (or full buffer if in-place).
 * @param dst Device pointer to the full destination buffer.
 * @param size Total byte size of the full buffer (must be divisible by world size).
 *
 * @note If @p src == @p dst, performs an in-place all-gather by offsetting @p src to the local shard.
 */
void NCCLCommunicator::gather_weight(const std::byte* src, std::byte* dst,  std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    if(src == dst) {
        src += shard_size * mRank; // in-place
    }
    ncclCheck(ncclAllGather(src,
                            dst,
                            shard_size, ncclInt8,
                            mNcclComm, mCommsStream));
}

/**
 * @brief Enqueue a point-to-point send of raw bytes.
 *
 * @param src Device pointer to bytes to send.
 * @param peer Destination rank.
 * @param size Number of bytes to send.
 */
void NCCLCommunicator::send(const std::byte* src, int peer, std::size_t size) {
    ncclCheck(ncclSend(src, size, ncclInt8, peer, mNcclComm, mCommsStream));
}

/**
 * @brief Enqueue a point-to-point receive of raw bytes.
 *
 * @param dst Device pointer to receive buffer.
 * @param peer Source rank.
 * @param size Number of bytes to receive.
 */
void NCCLCommunicator::recv(std::byte* dst, int peer, std::size_t size) {
    ncclCheck(ncclRecv(dst, size, ncclInt8, peer, mNcclComm, mCommsStream));
}

/**
 * @brief Make a compute stream wait until the communicator sync event is complete.
 *
 * @param compute_stream CUDA stream that should wait on the internal comms sync event.
 */
void NCCLCommunicator::wait_on_comms(cudaStream_t compute_stream) {
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, mCommsSync, 0));
}

#if USE_MPI

// macro conflict :(
#undef HOST
#include <mpi.h>

/**
 * @brief Throws a std::runtime_error if an MPI call returned an error.
 *
 * @param status MPI status code returned by an MPI API call.
 * @param file Source file where the failing call was made.
 * @param line Source line where the failing call was made.
 *
 * @throws std::runtime_error Always thrown when @p status != MPI_SUCCESS.
 */
void mpi_check(int status, const char *file, int line) {
    if (status != MPI_SUCCESS) {
        char mpi_error[4096];
        int mpi_error_len = 0;
        if(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) == MPI_SUCCESS) {
            throw std::runtime_error(fmt::format("Failed to create MPI error string for error at {}:{} ({})", file, line, status));
        }
        throw std::runtime_error(fmt::format("MPI error at {}:{}: {}", file, line, mpi_error));
    }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))

/**
 * @brief MPI-backed NCCL communicator variant (multi-process configuration).
 *
 * Uses MPI for barriers and host-side (all-)gathers needed by higher-level code.
 * NCCL grouping is started/ended around a transaction to batch NCCL operations.
 */
class NCCLCommunicatorMPI : public NCCLCommunicator {
public:
    using NCCLCommunicator::NCCLCommunicator;

    /**
     * @brief Finalizes MPI if initialized and no exception is active.
     *
     * @note MPI_Finalize may hang if called during stack unwinding in some environments; guarded accordingly.
     */
    ~NCCLCommunicatorMPI() override;

    /**
     * @brief Global barrier across all MPI ranks in MPI_COMM_WORLD.
     */
    void barrier() override;

    /**
     * @brief No-op in multi-process mode (launch queue throttling not needed across processes).
     */
    void _launch_queue_throttle_sync() override {};

    /**
     * @brief Gather fixed-size byte blobs from all ranks onto rank 0.
     *
     * @param recv Host buffer on rank 0 (size must be world*size); ignored on other ranks.
     * @param object Host pointer to this rank's byte blob of length @p size.
     * @param size Number of bytes contributed per rank.
     */
    void gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;

    /**
     * @brief All-gather fixed-size byte blobs from all ranks onto all ranks.
     *
     * @param recv Host buffer receiving concatenated blobs (size must be world*size).
     * @param object Host pointer to this rank's byte blob of length @p size.
     * @param size Number of bytes contributed per rank.
     */
    void all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;

    /**
     * @brief Transaction hook: wait for readiness and start NCCL group.
     *
     * @param cmd Command buffer containing the readiness event to wait on.
     */
    void on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) override;

    /**
     * @brief Transaction hook: end NCCL group and record completion event.
     *
     * @param signal CUDA event to record on the comms stream when NCCL work for the transaction is enqueued.
     */
    void on_finish_transaction(cudaEvent_t signal) override;
};


/**
 * @brief Destructor joins MPI resources and finalizes MPI if no exception is active.
 */
NCCLCommunicatorMPI::~NCCLCommunicatorMPI() {
    int is_init = 0;
    mpiCheck(MPI_Initialized(&is_init));
    // I've observed that (at least in some circumstances), when
    // an exception is active, MPI_Finalize just blocked forever...
    if(is_init && std::uncaught_exceptions() == 0) {
        mpiCheck(MPI_Finalize());
    }
}

/**
 * @brief Global barrier across all MPI ranks in MPI_COMM_WORLD.
 */
void NCCLCommunicatorMPI::barrier() {
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
}

/**
 * @brief Gather fixed-size byte blobs from all ranks onto rank 0.
 *
 * @param recv Host buffer on rank 0 (size must be world*size); ignored on other ranks.
 * @param object Host pointer to this rank's byte blob of length @p size.
 * @param size Number of bytes contributed per rank.
 */
void NCCLCommunicatorMPI::gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    mpiCheck(MPI_Gather(object, size, MPI_BYTE, recv, size, MPI_BYTE, 0, MPI_COMM_WORLD));
}

/**
 * @brief All-gather fixed-size byte blobs from all ranks onto all ranks.
 *
 * @param recv Host buffer receiving concatenated blobs (size must be world*size).
 * @param object Host pointer to this rank's byte blob of length @p size.
 * @param size Number of bytes contributed per rank.
 */
void NCCLCommunicatorMPI::all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    mpiCheck(MPI_Allgather(object, size, MPI_BYTE, recv, size, MPI_BYTE, MPI_COMM_WORLD));
}

/**
 * @brief Transaction hook: wait for readiness and start NCCL group.
 *
 * @param cmd Command buffer containing the readiness event to wait on.
 */
void NCCLCommunicatorMPI::on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) {
    CUDA_CHECK(cudaStreamWaitEvent(stream(), cmd.Ready));
    ncclCheck(ncclGroupStart());
}

/**
 * @brief Transaction hook: end NCCL group and record completion event.
 *
 * @param signal CUDA event to record on the comms stream when NCCL work for the transaction is enqueued.
 */
void NCCLCommunicatorMPI::on_finish_transaction(cudaEvent_t signal) {
    ncclCheck(ncclGroupEnd());
    CUDA_CHECK(cudaEventRecord(signal, stream()));
}

/**
 * @brief Construct an MPI-based NCCLCommunicator for the current MPI rank.
 *
 * Initializes MPI, obtains rank/world, broadcasts an ncclUniqueId from rank 0, and constructs the communicator.
 *
 * @return A communicator instance configured for MPI multi-process execution.
 */
std::unique_ptr<NCCLCommunicator> NCCLCommunicator::make_mpi_communicator() {
    mpiCheck(MPI_Init(nullptr, nullptr));
    int rank, world;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &world));

    ncclUniqueId nccl_id;
    if (rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
    }
    mpiCheck(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    return std::make_unique<NCCLCommunicatorMPI>(rank, world, &nccl_id);
}

#else
std::unique_ptr<NCCLCommunicator> NCCLCommunicator::make_mpi_communicator() {
    throw std::runtime_error("MPI communicator not available.");
}


#endif

#if USE_THREADS

#include <thread>
#include <barrier>

/**
 * @brief Thread-based NCCL communicator variant (multi-thread, single-process configuration).
 *
 * Provides barriers and host-side exchange using shared memory, and can optionally replace some NCCL ops
 * (all-gather and/or send/recv) with device-to-device memcpy coordinated via barriers.
 */
class NCCLCommunicatorThreads : public NCCLCommunicator {
public:
    struct SharedState {
        std::unique_ptr<std::barrier<>> Barrier;
        std::vector<std::byte*> Buffer;     // one pointer per thread
        std::vector<std::exception_ptr> Exceptions;
        std::mutex Mutex;
    };

    /**
     * @brief Construct a thread communicator for a given rank in a shared process.
     *
     * @param rank Thread/rank index (also used as CUDA device index).
     * @param world Total number of ranks/threads.
     * @param memcpy_allgather If true, all-gather may be implemented via host-coordinated D2D memcpy.
     * @param memcpy_send_recv If true, point-to-point send/recv may be implemented via host-coordinated D2D memcpy.
     * @param nccl_id Pointer to shared ncclUniqueId used for ncclCommInitRank.
     * @param state Shared synchronization/exchange state shared by all ranks in the process.
     */
    NCCLCommunicatorThreads(int rank, int world, bool memcpy_allgather, bool memcpy_send_recv, const void* nccl_id, std::shared_ptr<SharedState> state);

    /**
     * @brief Drops out of the shared barrier on destruction (if present).
     */
    ~NCCLCommunicatorThreads() override;

    /**
     * @brief Barrier across all ranks/threads in the process.
     */
    void barrier() override;

    /**
     * @brief Throttle launch queue in multithreaded mode by synchronizing CPU threads.
     */
    void _launch_queue_throttle_sync() override;

    /**
     * @brief Gather fixed-size host byte blobs onto rank 0 using shared memory and barriers.
     *
     * @param recv Host buffer on rank 0 to receive all blobs (world*size).
     * @param object Host pointer to this rank's blob of length @p size.
     * @param size Number of bytes per rank.
     */
    void gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;

    /**
     * @brief All-gather fixed-size host byte blobs onto all ranks using shared memory and barriers.
     *
     * @param recv Host buffer to receive concatenated blobs (world*size).
     * @param object Host pointer to this rank's blob of length @p size.
     * @param size Number of bytes per rank.
     */
    void all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;

    /**
     * @brief All-gather weights either via NCCL or via D2D memcpy depending on configuration.
     *
     * @param src Device pointer to local shard (or full buffer if in-place).
     * @param tgt Device pointer to full target buffer.
     * @param size Total bytes in the full buffer (must be divisible by world size).
     */
    void gather_weight(const std::byte* src, std::byte* tgt, std::size_t size) override;

    /**
     * @brief Send bytes to @p peer using NCCL or deferred memcpy emulation.
     *
     * @param src Device pointer to send buffer.
     * @param peer Destination rank.
     * @param size Number of bytes.
     */
    void send(const std::byte* src, int peer, std::size_t size) override;

    /**
     * @brief Receive bytes from @p peer using NCCL or deferred memcpy emulation.
     *
     * @param tgt Device pointer to receive buffer.
     * @param peer Source rank.
     * @param size Number of bytes.
     */
    void recv(std::byte* tgt, int peer, std::size_t size) override;

    /**
     * @brief Transaction hook: decide whether this transaction uses NCCL and/or memcpy emulation.
     *
     * If memcpy emulation is used, waits on readiness events from all workers before proceeding.
     * If NCCL is used, starts an NCCL group after waiting for readiness on this rank.
     *
     * @param cmd Command buffer containing queued commands and the readiness event.
     */
    void on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) override;

    /**
     * @brief Transaction hook: complete NCCL group and/or perform memcpy-based send/recv matching.
     *
     * For memcpy-emulated recv, assumes all workers enqueue the same number of receives and uses barriers
     * and per-rank sync events to ensure correct ordering/visibility across devices.
     *
     * @param signal CUDA event recorded on the comms stream to signal completion/progress to peers.
     *
     * @throws std::runtime_error If a send/recv size mismatch is detected.
     */
    void on_finish_transaction(cudaEvent_t signal) override;

private:
    std::shared_ptr<SharedState> mShare;
    bool mAllGatherUseMemcpy = false;
    bool mSendRecvUseMemcpy = true;

    // transaction status
    bool mUseMemcpy;
    bool mUseNCCL;

    struct sSendParams {
        const std::byte* Data;
        std::size_t Size;
        int Peer;
        bool Matched = false;
    };
    std::vector<sSendParams> mSendParams;

    struct sRecvParams {
        std::byte* Data;
        std::size_t Size;
        int Peer;
    };
    std::vector<sRecvParams> mRecvParams;
};

/**
 * @brief Construct a thread communicator for a given rank in a shared process.
 *
 * @param rank Thread/rank index (also used as CUDA device index).
 * @param world Total number of ranks/threads.
 * @param memcpy_allgather If true, all-gather may be implemented via host-coordinated D2D memcpy.
 * @param memcpy_send_recv If true, point-to-point send/recv may be implemented via host-coordinated D2D memcpy.
 * @param nccl_id Pointer to shared ncclUniqueId used for ncclCommInitRank.
 * @param state Shared synchronization/exchange state shared by all ranks in the process.
 */
NCCLCommunicatorThreads::NCCLCommunicatorThreads(int rank, int world, bool memcpy_allgather, bool memcpy_send_recv, const void* nccl_id, std::shared_ptr<SharedState> state):
    NCCLCommunicator(rank, world, nccl_id), mShare(std::move(state)), mAllGatherUseMemcpy(memcpy_allgather), mSendRecvUseMemcpy(memcpy_send_recv) {
}

/**
 * @brief Drops out of the shared barrier on destruction (if present).
 */
NCCLCommunicatorThreads::~NCCLCommunicatorThreads() {
    if(mShare && mShare->Barrier) {
        mShare->Barrier->arrive_and_drop();
    }
}

class ThreadsPackImp : public CommunicatorThreadsPack {
public:
    ThreadsPackImp(std::vector<std::jthread> threads, std::shared_ptr<NCCLCommunicatorThreads::SharedState> state) :
        mThreads(std::move(threads)), mState(std::move(state)){

    }

    ~ThreadsPackImp() override {
        // Destructors are `noexcept` by default; never throw here or we'll `std::terminate`.
        try {
            join_impl();
        } catch (...) {
            // Swallow: `run_threads_communicators()` calls join() explicitly so exceptions
            // are surfaced there. If multiple worker threads threw, joining may rethrow
            // again during stack unwinding; ignore in destructor.
        }
    }

    void join() override {
        join_impl();
    }

    bool has_exception() const override {
        std::lock_guard<std::mutex> lock(mState->Mutex);
        for(int t = 0; t < mThreads.size(); ++t) {
            if (auto error = mState->Exceptions[t]; error) {
                return true;
            }
        }
        return false;
    }
private:
    void join_impl() {
        // if any worker thread has already crashed, raise that exception in the main thread
        check_exceptions();

        for(auto& t: mThreads) {
            if(t.joinable()) {
                t.join();
            }
        }

        // ok, now that everyone has terminated, check again for proper exit
        check_exceptions();
    }

    void check_exceptions() {
        std::lock_guard<std::mutex> lock(mState->Mutex);
        for(int t = 0; t < mThreads.size(); ++t) {
            if(auto error = mState->Exceptions[t]; error) {
                fprintf(stderr, "Thread %d exited with uncaught exception\n", t);
                fflush(stderr);
                // reset the exception and rethrow it
                mState->Exceptions[t] = nullptr;
                std::rethrow_exception(error);
            }
        }
    }

    std::vector<std::jthread> mThreads;
    std::shared_ptr<NCCLCommunicatorThreads::SharedState> mState;
};

/**
 * @brief Convenience helper that runs a communicator-per-GPU in threads and waits for completion.
 *
 * @param ngpus Number of GPU ranks/threads to launch.
 * @param memcpy_allgather Enable memcpy-based all-gather emulation in threaded mode.
 * @param memcpy_send_recv Enable memcpy-based send/recv emulation in threaded mode.
 * @param work Callable invoked once per rank with that rank's communicator.
 */
void NCCLCommunicator::run_threads_communicators(int ngpus, bool memcpy_allgather, bool memcpy_send_recv, std::function<void(NCCLCommunicator& comm)> work) {
    auto threads = launch_threads_communicators(ngpus, memcpy_allgather, memcpy_send_recv, std::move(work));
    threads->join();
}

/**
 * @brief Launch a communicator-per-GPU in threads and return a joinable pack.
 *
 * @param ngpus Number of GPU ranks/threads to launch.
 * @param memcpy_allgather Enable memcpy-based all-gather emulation in threaded mode.
 * @param memcpy_send_recv Enable memcpy-based send/recv emulation in threaded mode.
 * @param work Callable invoked once per rank with that rank's communicator.
 *
 * @return A CommunicatorThreadsPack that can be joined and queried for exceptions.
 */
std::unique_ptr<CommunicatorThreadsPack> NCCLCommunicator::launch_threads_communicators(
            int ngpus, bool memcpy_allgather, bool memcpy_send_recv, std::function<void(NCCLCommunicator& comm)> work)
{
    std::vector<std::jthread> threads;
    ncclUniqueId nccl_id;
    ncclCheck(ncclGetUniqueId(&nccl_id));
    threads.reserve(ngpus);
    auto bar = std::make_shared<NCCLCommunicatorThreads::SharedState>(std::make_unique<std::barrier<>>(ngpus), std::vector<std::byte*>(ngpus));
    bar->Exceptions.resize(ngpus);
    for(int i = 0; i < ngpus; ++i) {
        threads.emplace_back([i, ngpus, nccl_id, memcpy_allgather, memcpy_send_recv, work, bar]() {
            try {
                if (!set_cpu_affinity()) {
                    fprintf(stderr, "WARNING: Failed to set CPU affinity for rank %d\n", i);
                }
                NCCLCommunicatorThreads comm(i, ngpus, memcpy_allgather, memcpy_send_recv, &nccl_id, bar);
                work(comm);
                bar->Barrier->arrive_and_wait();
            } catch(...) {
                std::lock_guard<std::mutex> lock(bar->Mutex);
                bar->Exceptions[i] = std::current_exception();
            }
        }
        );
    }
    return std::make_unique<ThreadsPackImp>(std::move(threads), std::move(bar));
}

/**
 * @brief Schedule a destructive all-to-all rotation using explicit send/recv pairs.
 *
 * Splits @p tensor into @c world_size shards and schedules point-to-point exchanges such that each rank
 * sends shard slices to peers and overwrites local storage positions with received shards.
 *
 * @param tensor Tensor whose device storage is partitioned and exchanged; contents are overwritten in-place.
 */
void NCCLCommunicator::schedule_destructive_all_to_all(const Tensor& tensor) {
    std::size_t shard_size = (ptrdiff_t)tensor.bytes() / world_size();
    for(int n = 1; n < world_size(); ++n) {
        int dst = (n + rank()) % world_size();
        int src = (rank() - n + world_size()) % world_size();
        int store = (rank() + n - 1 + world_size()) % world_size();
        mCmdBuf->Commands.emplace_back(CommandBuffer::Send{
            .Tensor = tensor.Data + dst * shard_size,
            .Bytes = shard_size,
            .Target = dst
            }
            );
        mCmdBuf->Commands.emplace_back(CommandBuffer::Recv{
            .Tensor = tensor.Data + store * shard_size,
            .Bytes = shard_size,
            .Source = src
        });
    }
}

void NCCLCommunicatorThreads::send(const std::byte* src, int peer, std::size_t size) {
    if (!mSendRecvUseMemcpy) {
        NCCLCommunicator::send(src, peer, size);
    } else {
        mSendParams.emplace_back(sSendParams{src, size, peer});
    }
}

void NCCLCommunicatorThreads::recv(std::byte* tgt, int peer, std::size_t size) {
    if (!mSendRecvUseMemcpy) {
        NCCLCommunicator::recv(tgt, peer, size);
    } else {
        mRecvParams.emplace_back(sRecvParams{tgt, size, peer});
    }
}

void NCCLCommunicatorThreads::gather_weight(const std::byte* src, std::byte* tgt, std::size_t size) {
    if(mAllGatherUseMemcpy) {
        auto wgt_list = host_all_gather(src);
        std::size_t shard_size = size / world_size();
        for (int i = 0; i < world_size(); ++i) {
            if (tgt + shard_size * i != wgt_list[i]) {
                CUDA_CHECK(cudaMemcpyAsync(tgt + shard_size * i, wgt_list[i], shard_size, cudaMemcpyDeviceToDevice, stream()));
            }
        }
    } else {
        NCCLCommunicator::gather_weight(src, tgt, size);
    }
}

/**
 * @brief Transaction hook: decide whether this transaction uses NCCL and/or memcpy emulation.
 *
 * If memcpy emulation is used, waits on readiness events from all workers before proceeding.
 * If NCCL is used, starts an NCCL group after waiting for readiness on this rank.
 *
 * @param cmd Command buffer containing queued commands and the readiness event.
 */
void NCCLCommunicatorThreads::on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) {
    mUseMemcpy = false;
    mUseNCCL = false;
    for (auto& cmd: cmd.Commands) {
        if (std::holds_alternative<CommandBuffer::ScatterReduce>(cmd)) {
            mUseNCCL = true;
        }
        if (std::holds_alternative<CommandBuffer::Gather>(cmd)) {
            if (!mAllGatherUseMemcpy) mUseNCCL = true;
            if (mAllGatherUseMemcpy) mUseMemcpy = true;
        }
        if (std::holds_alternative<CommandBuffer::Send>(cmd)) {
            if (!mSendRecvUseMemcpy) mUseNCCL = true;
            if (mSendRecvUseMemcpy) mUseMemcpy = true;
        }
    }

    assert(mUseNCCL || mUseMemcpy);

    if(mUseMemcpy) {
        // ensure every worker has set-up commands.Ready to the most recent version
        barrier();
        // get the ready event from all workers
        auto event_list = host_all_gather(cmd.Ready);
        // make sure to block the comms thread until the data is ready on every worker
        for (auto event: event_list) {
            CUDA_CHECK(cudaStreamWaitEvent(stream(), event, 0));
        }
    }

    if(mUseNCCL){
        CUDA_CHECK(cudaStreamWaitEvent(stream(), cmd.Ready, 0));
        ncclCheck(ncclGroupStart());
    }
}

/**
 * @brief Transaction hook: complete NCCL group and/or perform memcpy-based send/recv matching.
 *
 * For memcpy-emulated recv, assumes all workers enqueue the same number of receives and uses barriers
 * and per-rank sync events to ensure correct ordering/visibility across devices.
 *
 * @param signal CUDA event recorded on the comms stream to signal completion/progress to peers.
 *
 * @throws std::runtime_error If a send/recv size mismatch is detected.
 */
void NCCLCommunicatorThreads::on_finish_transaction(cudaEvent_t signal) {
    if (!mRecvParams.empty()) {
        // get send-queues from peers
        std::vector<std::vector<sSendParams>*> send_params = host_all_gather(&mSendParams);
        std::vector<cudaEvent_t> sync_events = host_all_gather(signal);
        // ok, now iterate all recv's
        for (auto& recv: mRecvParams) {
            // find matching send
            for (auto& send : *send_params.at(recv.Peer)) {
                if (send.Peer != rank() || send.Matched) continue;
                // copy data
                if (recv.Size != send.Size) {
                    throw std::runtime_error("Size mismatch in recv/send");
                }
                CUDA_CHECK(cudaMemcpyAsync(recv.Data, send.Data, recv.Size, cudaMemcpyDeviceToDevice, stream()));
                send.Matched = true;
                break;
            }

            CUDA_CHECK(cudaEventRecord(signal, stream()));
            barrier();      // assumes _all_ workers have the same number of receives!
            for (int j = 0; j < world_size(); ++j) {
                if (j != rank()) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream(), sync_events[j], 0));
                }
            }
        }

        barrier();
        mRecvParams.clear();
        mSendParams.clear();
    }
    if(mUseNCCL) {
        ncclCheck(ncclGroupEnd());
    }

    CUDA_CHECK(cudaEventRecord(signal, stream()));
}

/**
 * @brief Barrier across all ranks/threads in the process.
 */
void NCCLCommunicatorThreads::barrier() {
    mShare->Barrier->arrive_and_wait();
}

/**
 * @brief Throttle launch queue in multithreaded mode by synchronizing CPU threads.
 *
 * This prevents deadlocks that occur in multi-threaded (single-process) NCCL configurations
 * but not in multi-process configurations.
 *
 * Root Cause:
 * CUDA maintains a per-process launch queue for kernel submissions. In multi-threaded mode,
 * all GPU worker threads share this single queue. NCCL collective operations contain an
 * implicit global barrier - all ranks must reach the collective before any can proceed.
 *
 * Deadlock Scenario:
 * 1. GPU 0 (fast) enqueues its collective operation
 * 2. GPU 0 continues and fills the per-process launch queue with subsequent kernels
 * 3. GPU 1 (slow) tries to enqueue work needed to reach the collective, but queue is full
 * 4. GPU 0 is blocked on the collective barrier waiting for GPU 1
 * 5. GPU 1 cannot enqueue because GPU 0 filled the queue → deadlock
 *
 * Solution:
 * CPU-side thread synchronization (not GPU synchronization) ensures all threads reach the
 * collective submission point together before any thread can continue enqueueing work.
 * This prevents the launch queue from being exhausted by a single fast GPU.
 *
 * Note: Multi-process mode doesn't need this because each process has its own launch queue.
 */
void NCCLCommunicatorThreads::_launch_queue_throttle_sync() {
    this->barrier();
}

/**
 * @brief Gather fixed-size host byte blobs onto rank 0 using shared memory and barriers.
 *
 * @param recv Host buffer on rank 0 to receive all blobs (world*size).
 * @param object Host pointer to this rank's blob of length @p size.
 * @param size Number of bytes per rank.
 */
void NCCLCommunicatorThreads::gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    if(rank() == 0) {
        mShare->Buffer[0] = recv;
    }
    barrier();
    std::memcpy(mShare->Buffer[0] + rank() * size, object, size);
    barrier();
    if(rank() == 0) {
        mShare->Buffer[0] = nullptr;
    }
}

/**
 * @brief All-gather fixed-size host byte blobs onto all ranks using shared memory and barriers.
 *
 * @param recv Host buffer to receive concatenated blobs (world*size).
 * @param object Host pointer to this rank's blob of length @p size.
 * @param size Number of bytes per rank.
 */
void NCCLCommunicatorThreads::all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    barrier();
    mShare->Buffer[rank()] = const_cast<std::byte*>(object);
    barrier();
    for(int i = 0; i < world_size(); ++i) {
        std::memcpy(recv + i * size,  mShare->Buffer[i], size);
    }
    barrier();
    mShare->Buffer[rank()] = nullptr;
}
#else
void NCCLCommunicator::launch_threads_communicators(int zero_level) {
    throw std::runtime_error("threads communicator not available.");
}

#endif
