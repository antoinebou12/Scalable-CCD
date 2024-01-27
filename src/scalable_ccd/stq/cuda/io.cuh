#pragma once

#include <vector>

#include <scalable_ccd/stq/cuda/aabb.cuh>

namespace stq::gpu {

void constructBoxes(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    std::vector<Aabb>& boxes,
    int threads = -1,
    Scalar inflation_radius = 0);

void parseMesh(
    const char* filet0,
    const char* filet1,
    Eigen::MatrixXd& V0,
    Eigen::MatrixXd& V1,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& E);

typedef tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>>
    ThreadSpecificOverlaps;
void merge_local_overlaps(
    const ThreadSpecificOverlaps& storages,
    std::vector<std::pair<int, int>>& overlaps);

} // namespace stq::gpu