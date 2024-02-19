#pragma once

#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/utils/device_buffer.cuh>

namespace scalable_ccd::cuda {

/// @brief Flip the sign of an id.
/// @param id The id to flip
/// @return The flipped id
__device__ inline int flip_id(const int id) { return -id - 1; }

/// @brief Determine if any of the vertices of a and b are the same.
/// @param a First vertices
/// @param b Second vertices
/// @return True if any of the vertices of a and b are the same.
__device__ inline bool share_a_vertex(const int3& a, const int3& b)
{
    return a.x == b.x || a.x == b.y || a.x == b.z || a.y == b.x || a.y == b.y
        || a.y == b.z || a.z == b.x || a.z == b.y || a.z == b.z;
}

/// @brief Is the pair (id_a, id_b) valid?
/// @param id_a The first id
/// @param id_b The second id
/// @return True if the pair is valid
template <bool is_two_lists>
__device__ inline bool is_valid_pair(const long id_a, const long id_b)
{
    if constexpr (is_two_lists) {
        return (id_a >= 0 && id_b < 0) || (id_a < 0 && id_b >= 0);
    } else {
        return true;
    }
}

/// @brief Add an overlap (xid, yid) to overlaps if there is enough space.
/// Keep count of overlaps that do not fit.
/// @param xid First box id
/// @param yid Second box id
/// @param max_overlap_size Maximum number of overlaps allocated
/// @param overlaps Array of overlaps
/// @param count Current number of overlaps in overlaps
/// @param real_count Actual number of overlaps found
__device__ inline void add_overlap(
    const int xid,
    const int yid,
    RawDeviceBuffer<int2>& overlaps,
    int& real_count)
{
    if (atomicAdd(&real_count, 1) < overlaps.capacity) {
        overlaps.push(make_int2(xid, yid));
    }
}

} // namespace scalable_ccd::cuda