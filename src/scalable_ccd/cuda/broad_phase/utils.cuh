#pragma once

#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>

namespace scalable_ccd::cuda {

struct sort_aabb_x {
    __device__ bool operator()(const AABB& a, const AABB& b) const
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

void setup(
    int device_id,
    int& shared_memory_size,
    int& threads,
    int& boxes_per_thread);

/// @brief Dispatch a kernel with the given grid and block size
/// @tparam ...Arguments Arguments to the kernel
/// @param tag Name for profiling
/// @param gs
/// @param bs
/// @param mem
/// @param f
/// @param ...args
template <typename... Arguments>
void dispatch(
    const std::string& tag,
    int gs,
    int bs,
    size_t mem,
    void (*f)(Arguments...),
    Arguments... args)
{
    SCALABLE_CCD_GPU_PROFILE_POINT(tag);

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

} // namespace scalable_ccd::cuda