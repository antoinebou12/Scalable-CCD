#pragma once

#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/collision.cuh>
#include <scalable_ccd/cuda/utils/device_buffer.cuh>

namespace scalable_ccd::cuda {

/// @brief Add an overlap (xid, yid) to overlaps if there is enough space.
/// Do not keep count of overlaps that do not fit.
/// @param xid First box id
/// @param yid Second box id
/// @param max_overlap_size Maximum number of overlaps allocated
/// @param overlaps Array of overlaps
/// @param count Current number of overlaps in overlaps
__device__ void add_overlap(
    const int xid,
    const int yid,
    const int max_overlap_size,
    int2* overlaps,
    int* count);

/// @brief
/// @param boxes
/// @param count
/// @param overlaps
/// @param num_boxes
/// @param guess
/// @param nbox
/// @param start
/// @param end
/// @return
__global__ void retrieve_collision_pairs(
    const AABB* const boxes,
    int* count,
    int2* overlaps,
    int num_boxes,
    int guess,
    int nbox,
    int start = 0,
    int end = INT_MAX);

} // namespace scalable_ccd::cuda