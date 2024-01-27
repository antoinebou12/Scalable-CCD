#pragma once

#include <spdlog/spdlog.h>

namespace scalable_ccd::stq::gpu {

#define cudaCheckErrors(msg)                                                   \
    do {                                                                       \
        cudaError_t __err = cudaGetLastError();                                \
        if (__err != cudaSuccess) {                                            \
            spdlog::error(                                                     \
                "Fatal error: {:s} ({:s} at {:s}:{:d})", msg,                  \
                cudaGetErrorString(__err), __FILE__, __LINE__);                \
            spdlog::error("FAILED - ABORTING");                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

void setup(int devId, int& smemSize, int& threads, int& nbox);

struct sorter { };
struct sort_aabb_x : sorter {
    __device__ bool operator()(const Aabb& a, const Aabb& b) const
    {
        return (a.min.x < b.min.x);
    }

    __device__ bool operator()(const Scalar3& a, const Scalar3& b) const
    {
        return (a.x < b.x);
    }

    __device__ bool operator()(const Scalar2& a, const Scalar2& b) const
    {
        return (a.x < b.x);
    }
};

} // namespace scalable_ccd::stq::gpu