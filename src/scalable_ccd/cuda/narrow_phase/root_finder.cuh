#pragma once

#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/memory_handler.hpp>

#include <thrust/device_vector.h>

#include <vector>

namespace scalable_ccd::cuda {

class CCDData;
class CCDBuffer;

/// @brief Compute the tolerance for the CCD algorithm.
/// @param data CCD query data to populate with the tolerance.
/// @param query_size The number of queries.
template <bool is_vf>
__global__ void compute_tolerance(CCDData* data, const int query_size);

/// @brief Run the CCD algorithm on the GPU.
/// @param buffer CCD buffer for intervals
/// @param data CCD query data
/// @param toi Time of impact
template <bool is_vf>
__global__ void
ccd_kernel(CCDBuffer* const buffer, CCDData* const data, Scalar* const toi);

/// @brief Run the narrow phase CCD algorithm on the GPU.
/// @tparam is_vf If true, run vertex-face CCD, otherwise run edge-edge CCD.
/// @param d_data CCD query data.
/// @param memory_handler Memory handler.
/// @param parallel_nbr The number of parallel queries.
/// @param max_iter The maximum number of iterations for the algorithm.
/// @param tol The tolerance for the algorithm.
/// @param use_ms True if using a minimum separation.
/// @param allow_zero_toi If true, allow the algorithm to return a time of impact of 0.
/// @param toi The time of impact.
/// @return True if the algorithm overflows the maximum memory, false otherwise.
template <bool is_vf>
bool ccd(
    thrust::device_vector<CCDData>& d_data,
    const std::shared_ptr<MemoryHandler> memory_handler,
    const int parallel_nbr,
    const int max_iter,
    const Scalar tol,
    const bool use_ms,
    const bool allow_zero_toi,
    Scalar& toi);

} // namespace scalable_ccd::cuda