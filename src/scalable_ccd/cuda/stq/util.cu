
#include "util.cuh"

#include <scalable_ccd/cuda/stq/queue.cuh>
#include <scalable_ccd/utils/logger.hpp>

namespace scalable_ccd::cuda::stq {

void setup(
    int device_id, int& shared_memory_size, int& threads, int& boxes_per_thread)
{
    int max_shared_memory;
    cudaDeviceGetAttribute(
        &max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
    logger().trace(
        "Max shared memory per block: {:g} KB",
        max_shared_memory / double(1 << 10));

    int max_threads;
    cudaDeviceGetAttribute(
        &max_threads, cudaDevAttrMaxThreadsPerBlock, device_id);
    logger().trace("Max threads per block: {:d} threads", max_threads);

    if (!boxes_per_thread) {
        boxes_per_thread =
            std::max(max_shared_memory / sizeof(Aabb) / max_threads, 1ul);
    }
    logger().trace("Boxes per thread: {:d}", boxes_per_thread);

    // divide threads by an arbitrary number as long as its reasonable >64
    if (!threads) {
        cudaDeviceGetAttribute(
            &threads, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);

        logger().trace("Max threads per multiprocessor: {:d} threads", threads);
    }
    shared_memory_size =
        HEAP_SIZE * sizeof(int2); // boxes_per_thread * threads * sizeof(Aabb);

    if (shared_memory_size > max_shared_memory) {
        logger().error(
            "Shared memory size exceeds max shared memory per block!");
        logger().error(
            "Max shared memory per block: {:d} B", max_shared_memory);
        logger().error("Shared memory size: {:d} B", shared_memory_size);
        throw std::runtime_error(
            "Shared memory size exceeds max shared memory per block");
    }
    // while (shared_memory_size > max_shared_memory || threads > max_threads) {
    //   // threads--;
    //   // shared_memory_size = boxes_per_thread * threads * sizeof(Aabb);
    //   shared_memory_size /= 2;
    // }
    logger().trace("Actual threads per block: {:d} threads", threads);
    logger().trace("Shared memory allocated: {:d} B", shared_memory_size);
}

template <typename... Arguments>
void dispatch(
    const std::string& tag,
    int gs,
    int bs,
    size_t mem,
    void (*f)(Arguments...),
    Arguments... args)
{
    if (!mem) {
        f<<<gs, bs>>>(args...);
    } else {
        f<<<gs, bs, mem>>>(args...);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        logger().trace(
            "Kernel launch failure {:s}\nTrying device-kernel launch",
            cudaGetErrorString(error));

        f(args...);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::runtime_error(fmt::format(
                "Device-kernel launch failure {:s}", cudaGetErrorString(err)));
        }
    }
}

template <typename... Arguments>
void dispatch(
    const std::string& tag,
    int gs,
    int bs,
    void (*f)(Arguments...),
    Arguments... args)
{
    size_t mem = 0;
    dispatch(tag, gs, bs, mem, f, args...);
}

} // namespace scalable_ccd::cuda::stq