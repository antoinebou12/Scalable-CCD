#pragma once

#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/broad_phase/utils.cuh>

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

    /// @brief Build the broad phase data structure.
    /// @param boxes Vector of AABBs
    /// @return A reference to the boxes stored on the GPU
    void build(const std::shared_ptr<DeviceAABBs> boxes);

    /// @brief Build the broad phase data structure.
    /// @param boxes Vector of AABBs
    /// @return A reference to the boxes stored on the GPU
    void build(
        const std::shared_ptr<DeviceAABBs> boxesA,
        const std::shared_ptr<DeviceAABBs> boxesB);

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
    bool is_complete() const { return thread_start_box_id >= num_boxes(); }

    /// @brief Get the boxes stored on the GPU (unsorted).
    std::shared_ptr<DeviceAABBs> boxes() { return d_boxes; }

    /// @brief Get the number of boxes stored on the GPU.
    size_t num_boxes() const { return d_boxes ? d_boxes->size() : 0; }

    /// @brief Get the resulting overlaps stored on the GPU.
    /// @see detect_overlaps_partial
    const thrust::device_vector<int2>& overlaps() { return d_overlaps; }

    int threads_per_block = 32;

private:
    Dimension
    calc_sort_dimension(const thrust::device_vector<AABB>& d_boxes) const;

    int grid_dim_1d() const
    {
        assert(d_boxes);
        return num_boxes() / threads_per_block + 1;
    }

    std::shared_ptr<MemoryHandler> memory_handler;

    /// @brief Boxes stored on the GPU
    std::shared_ptr<DeviceAABBs> d_boxes;

    /// @brief Populated with the indices of overlapping boxes upon detect_overlaps_partial()
    thrust::device_vector<int2> d_overlaps;

    const int device_init_id = 0;

    int num_boxes_per_thread = 0;
    int thread_start_box_id = 0;
    int num_devices = 1;

    int shared_memory_size;

    /// @brief Was the broad phase constructed with two lists of boxes?
    bool is_two_lists;
};

} // namespace scalable_ccd::cuda