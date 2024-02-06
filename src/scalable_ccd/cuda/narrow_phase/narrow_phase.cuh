#pragma once
#include <scalable_ccd/cuda/scalar.cuh>
#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/utils/device_matrix.cuh>

#include <thrust/device_vector.h>
#include <vector>

namespace scalable_ccd::cuda {

/// @brief
/// @param boxes
/// @param memory_handler
/// @param vertices_t0
/// @param vertices_t1
/// @param N
/// @param nbox
/// @param parallel
/// @param devcount
/// @param limitGB
/// @param overlaps
/// @param result_list
/// @param allow_zero_toi
/// @param min_distance
/// @return
// double run_ccd(
//     const std::vector<AABB> boxes,
//     std::shared_ptr<MemoryHandler> memory_handler,
//     const Eigen::MatrixXd& vertices_t0,
//     const Eigen::MatrixXd& vertices_t1,
//     int N,
//     int& nbox,
//     int& parallel,
//     int& devcount,
//     int& limitGB,
//     std::vector<std::pair<int, int>>& overlaps,
//     std::vector<int>& result_list,
//     const bool allow_zero_toi,
//     Scalar& min_distance);

/// @brief Run the CCD narrow phase on the GPU
/// @param d_vertices_t0
/// @param d_vertices_t1
/// @param d_boxes The list of AABBs
/// @param d_overlaps The list of pairs of indices of the boxes that overlap
/// @param num_vertices
/// @param threads
/// @param max_iter
/// @param tol
/// @param ms
/// @param allow_zero_toi
/// @param memory_handler
/// @param result_list
/// @param toi
/// @return
void run_narrow_phase(
    const DeviceMatrix<Scalar>& d_vertices_t0,
    const DeviceMatrix<Scalar>& d_vertices_t1,
    const thrust::device_vector<AABB>& d_boxes,
    const thrust::device_vector<int2>& d_overlaps,
    const int threads,
    const int max_iter,
    const Scalar tol,
    const Scalar ms,
    const bool allow_zero_toi,
    std::shared_ptr<MemoryHandler> memory_handler,
    std::vector<int>& result_list,
    Scalar& toi);

} // namespace scalable_ccd::cuda