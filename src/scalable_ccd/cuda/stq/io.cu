#include <scalable_ccd/cuda/stq/io.cuh>

#include <spdlog/spdlog.h>

#include <tbb/global_control.h>

namespace scalable_ccd::cuda::stq {

void constructBoxes(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    std::vector<Aabb>& boxes,
    int threads,
    Scalar inflation_radius)
{
    if (threads <= 0)
        threads = CPU_THREADS;
    spdlog::trace("constructBoxes threads : {}", threads);
    tbb::global_control thread_limiter(
        tbb::global_control::max_allowed_parallelism, threads);
    addVertices(vertices_t0, vertices_t1, inflation_radius, boxes);
    addEdges(vertices_t0, vertices_t1, edges, inflation_radius, boxes);
    addFaces(vertices_t0, vertices_t1, faces, inflation_radius, boxes);
}

void merge_local_overlaps(
    const ThreadSpecificOverlaps& storages,
    std::vector<std::pair<int, int>>& overlaps)
{
    overlaps.clear();
    size_t num_overlaps = overlaps.size();
    for (const auto& local_overlaps : storages) {
        num_overlaps += local_overlaps.size();
    }
    // serial merge!
    overlaps.reserve(num_overlaps);
    for (const auto& local_overlaps : storages) {
        overlaps.insert(
            overlaps.end(), local_overlaps.begin(), local_overlaps.end());
    }
}

} // namespace scalable_ccd::cuda::stq