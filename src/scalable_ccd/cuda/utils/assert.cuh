#pragma once

#include <scalable_ccd/utils/logger.hpp>

#define gpuErrchk(ans)                                                         \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }

namespace scalable_ccd::cuda {

inline void gpuAssert(
    cudaError_t code, const std::string& file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        logger().error(
            "GPUassert: {} {} {:d}", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
} // namespace scalable_ccd::cuda