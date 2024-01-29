#pragma once

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <vector>

namespace scalable_ccd {

/// @brief Sort the boxes along the given axis.
/// @param axis The axis to sort along.
/// @param boxes The boxes to sort.
void sort_along_axis(const int axis, std::vector<AABB>& boxes);

/// @brief Sweep the boxes along the given axis.
/// @param[in] boxes Boxes to sweep.
/// @param[in] n Number of boxes.
/// @param[in,out] sort_axis Axis to sweep along and the next axis to sort along.
/// @param[out] overlaps Overlaps to populate.
template <bool is_two_lists>
void sweep(
    const std::vector<AABB>& boxes,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps);

/// @brief Run the sort and sweep (a.k.a. sweep and prune) algorithm for a set of boxes.
/// @param[in,out] boxes The boxes to check for overlaps.
/// @param[in,out] sort_axis The axis to sort along.
/// @param[out] overlaps The overlaps to populate.
void sort_and_sweep(
    std::vector<AABB> boxes,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps);

/// @brief Run the sort and sweep (a.k.a. sweep and prune) algorithm for two sets of boxes.
/// @param[in,out] boxesA The first set of boxes to check for overlaps.
/// @param[in,out] boxesB The second set of boxes to check for overlaps.
/// @param[in,out] sort_axis The axis to sort along.
/// @param[out] overlaps The overlaps to populate.
void sort_and_sweep(
    std::vector<AABB> boxesA,
    std::vector<AABB> boxesB,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps);

} // namespace scalable_ccd