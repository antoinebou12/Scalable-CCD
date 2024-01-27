
#include <scalable_ccd/cuda/stq/aabb.cuh>
#include <scalable_ccd/cuda/stq/util.cuh>
#include <scalable_ccd/cuda/stq/queue.cuh>

#include <spdlog/spdlog.h>

namespace scalable_ccd::cuda::stq {

// void setup(int devId, int &smemSize, int &threads, int &nboxes);

void setup(int devId, int& smemSize, int& threads, int& nbox)
{
    int maxSmem;
    cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlock, devId);
    spdlog::trace("Max shared Memory per Block: {:d} B", maxSmem);

    int maxThreads;
    cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, devId);
    spdlog::trace("Max threads per Block: {:d} thrds", maxThreads);

    nbox =
        nbox ? nbox : std::max((int)(maxSmem / sizeof(Aabb)) / maxThreads, 1);
    spdlog::trace("Boxes per Thread: {:d}", nbox);

    // divide threads by an arbitrary number as long as its reasonable >64
    if (!threads) {
        cudaDeviceGetAttribute(
            &threads, cudaDevAttrMaxThreadsPerMultiProcessor, devId);

        spdlog::trace("Max threads per Multiprocessor: {:d} thrds", threads);
    }
    smemSize = HEAP_SIZE * sizeof(int2); // nbox * threads * sizeof(Aabb);

    if (smemSize > maxSmem) {
        spdlog::error("Shared memory size exceeds max shared memory per block");
        spdlog::error("Max shared memory per block: {:d} B", maxSmem);
        spdlog::error("Shared memory size: {:d} B", smemSize);
        exit(1);
    }
    // while (smemSize > maxSmem || threads > maxThreads) {
    //   // threads--;
    //   // smemSize = nbox * threads * sizeof(Aabb);
    //   smemSize /= 2;
    // }
    spdlog::trace("Actual threads per Block: {:d} thrds", threads);
    spdlog::trace("Shared mem alloc: {:d} B", smemSize);
    return;
}

} // namespace scalable_ccd::cuda::stq