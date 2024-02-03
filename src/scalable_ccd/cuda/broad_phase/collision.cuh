#pragma once

#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/utils/device_buffer.cuh>

namespace scalable_ccd::cuda {

/// @brief Check if boxes a and b overlap.
/// @param a Fist box
/// @param b Second box
/// @return True if a and b overlap.
__device__ bool does_collide(const AABB& a, const AABB& b);

/// @brief Check if boxes a and b overlap.
/// @param a Fist box
/// @param b Second box
/// @return True if a and b overlap.
__device__ bool does_collide(const MiniBox& a, const MiniBox& b);

/// @brief Determine if any of the vertices of a and b are the same.
/// @param a First vertices
/// @param b Second vertices
/// @return True if any of the vertices of a and b are the same.
__device__ bool covertex(const int3& a, const int3& b);

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

/// @brief Add an overlap (xid, yid) to overlaps if there is enough space.
/// Keep count of overlaps that do not fit.
/// @param xid First box id
/// @param yid Second box id
/// @param max_overlap_size Maximum number of overlaps allocated
/// @param overlaps Array of overlaps
/// @param count Current number of overlaps in overlaps
/// @param real_count Actual number of overlaps found
__device__ void add_overlap(
    const int xid,
    const int yid,
    RawDeviceBuffer<int2>& overlaps,
    int& real_count);

} // namespace scalable_ccd::cuda