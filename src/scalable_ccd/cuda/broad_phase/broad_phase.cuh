#pragma once

#include <scalable_ccd/cuda/memory_handler.cuh>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>

#include <thrust/device_vector.h>

#include <memory>
#include <vector>
#include <utility>

namespace scalable_ccd::cuda {

class BroadPhase {
public:
    BroadPhase() : BroadPhase(std::make_shared<MemoryHandler>()) { }

    BroadPhase(std::shared_ptr<MemoryHandler> _memory_handler)
        : memory_handler(_memory_handler)
    {
    }

    void clear();

    const thrust::device_vector<AABB>& build(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& F,
        std::vector<AABB>& boxes,
        double inflation_radius = 0)
    {
        return build(V, V, E, F, boxes, inflation_radius);
    }

    const thrust::device_vector<AABB>& build(
        const Eigen::MatrixXd& V0,
        const Eigen::MatrixXd& V1,
        const Eigen::MatrixXi& E,
        const Eigen::MatrixXi& F,
        std::vector<AABB>& boxes,
        double inflation_radius = 0)
    {
        constructBoxes(V0, V1, E, F, boxes, inflation_radius);
        return build(boxes);
    }

    /// @brief Build the broad phase data structure.
    /// @param boxes Vector of AABBs
    /// @return A reference to the boxes stored on the GPU
    const thrust::device_vector<AABB>& build(const std::vector<AABB>& boxes);

    /// @brief Run the STQ broad phase algorithm on the GPU.
    /// This function is called multiple times until all boxes are processed.
    /// @return A reference to the resulting overlaps stored on the GPU
    const thrust::device_vector<int2>& detect_overlaps_partial();

    /// @brief Run the STQ broad phase algorithm on the GPU.
    /// This function is called once to process all boxes and stores them on the
    /// CPU.
    /// @return A CPU vector of pairs of overlapping box indices
    std::vector<std::pair<int, int>> detect_overlaps();

    /// @brief Is the broad phase algorithm complete?
    bool is_complete() const { return thread_start_box_id >= d_boxes.size(); }

    /// @brief Get the boxes stored on the GPU (unsorted).
    const thrust::device_vector<AABB>& boxes() { return d_boxes; }

    /// @brief Get the number of boxes stored on the GPU.
    size_t num_boxes() const { return d_boxes.size(); }

    /// @brief Get the resulting overlaps stored on the GPU.
    const thrust::device_vector<int2>& overlaps() { return d_overlaps; }

private:
    Dimension calc_sort_dimension() const;

    int grid_dim_1d() const { return d_boxes.size() / threads_per_block + 1; }

    std::shared_ptr<MemoryHandler> memory_handler;

    thrust::device_vector<AABB> d_boxes;
    thrust::device_vector<Scalar2> d_sm;
    thrust::device_vector<MiniBox> d_mini;

    thrust::device_vector<int2> d_overlaps;

    const int device_init_id = 0;
    const int memory_limit_GB = 0;

    int num_boxes_per_thread = 0;
    int threads_per_block = 32;
    int thread_start_box_id = 0;
    int num_devices = 1;

    int smemSize;
};

} // namespace scalable_ccd::cuda