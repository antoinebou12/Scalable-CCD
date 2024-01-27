#pragma once

#include <vector>

#include <scalable_ccd/cuda/stq/aabb.cuh>

namespace scalable_ccd::cuda::stq {

void constructBoxes(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    std::vector<Aabb>& boxes,
    int threads = -1,
    Scalar inflation_radius = 0);

typedef tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>>
    ThreadSpecificOverlaps;
void merge_local_overlaps(
    const ThreadSpecificOverlaps& storages,
    std::vector<std::pair<int, int>>& overlaps);

} // namespace scalable_ccd::cuda::stq