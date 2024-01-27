#pragma once

#include <scalable_ccd/stq/aabb.hpp>
#include <vector>

namespace scalable_ccd::stq {

/// @brief Is the element a vertex?
/// @param vids The vertex ids of the element.
/// @return True if the element is a vertex, false otherwise.
inline bool is_vertex(const std::array<int, 3>& vids)
{
    return vids[2] < 0 && vids[1] < 0;
}

/// @brief Is the element an edge?
/// @param vids The vertex ids of the element.
/// @return True if the element is an edge, false otherwise.
inline bool is_edge(const std::array<int, 3>& vids)
{
    return vids[2] < 0 && vids[1] >= 0;
}

/// @brief Is the element a face?
/// @param vids The vertex ids of the element.
/// @return True if the element is a face, false otherwise.
inline bool is_face(const std::array<int, 3>& vids) { return vids[2] >= 0; }

/// @brief Determine if two elements can be a valid pair.
/// @param a The first element.
/// @param b The second element.
/// @return True if the elements can be a valid pair, false otherwise.
bool is_valid_pair(const std::array<int, 3>& a, const std::array<int, 3>& b);

/// @brief Sort the boxes along the given axis.
/// @param axis The axis to sort along.
/// @param boxes The boxes to sort.
void sort_along_axis(const int axis, std::vector<Aabb>& boxes);

/// @brief Sweep the boxes along the given axis.
/// @param[in] boxes Boxes to sweep.
/// @param[in] n Number of boxes.
/// @param[in,out] sort_axis Axis to sweep along and the next axis to sort along.
/// @param[out] overlaps Overlaps to populate.
void sweep(
    const std::vector<Aabb>& boxes,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps);

/// @brief Run the sort and sweep (a.k.a. sweep and prune) algorithm.
/// @param boxes The boxes to check for overlaps.
/// @param sort_axis The axis to sort along.
/// @param overlaps The overlaps to populate.
void sort_and_sweep(
    std::vector<Aabb>& boxes,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps);

} // namespace scalable_ccd::stq