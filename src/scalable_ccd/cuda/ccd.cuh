#pragma once

#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/scalar.cuh>
#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>

#include <vector>
#include <memory>

namespace scalable_ccd::cuda {

/// @brief Run broad and narrow phase CCD to compute the earliest time of impact
/// @param vertices_t0 Vertex positions at time t=0
/// @param vertices_t1 Vertex positions at time t=1
/// @param boxes Axis-aligned bounding boxes for the vertices, edges, and faces
/// @param max_iterations Maximum number of iterations in the narrow phase
/// @param tolerance Tolerance for the time of impact
/// @param minimum_separation_distance Minimum separation distance
/// @param allow_zero_toi If true, allow a time of impact of zero
/// @param result_list List of collision pairs
/// @param memory_limit_GB Maximum GPU memory usage in GB
/// @return The earliest time of impact
Scalar
ccd(const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const std::vector<AABB>& boxes,
    const int max_iterations,
    const Scalar tolerance,
    const Scalar minimum_separation_distance,
    const bool allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<int>& result_list
#endif
    const int memory_limit_GB = 0);

} // namespace scalable_ccd::cuda