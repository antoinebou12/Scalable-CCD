#pragma once

#include <scalable_ccd/cuda/broad_phase/aabb.cuh>

#include <vector>
#include <utility>

namespace scalable_ccd::cuda {

void runBroadPhaseMultiGPU(
    const std::vector<AABB>& boxes,
    int N,
    int nbox,
    std::vector<std::pair<int, int>>& overlaps,
    int& threads,
    int& devcount);

} // namespace scalable_ccd::cuda