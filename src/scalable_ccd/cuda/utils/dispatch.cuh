#pragma once

#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>

namespace scalable_ccd::cuda {

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