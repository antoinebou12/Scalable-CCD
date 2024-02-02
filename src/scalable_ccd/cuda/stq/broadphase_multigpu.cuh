#pragma once

#include <scalable_ccd/cuda/stq/aabb.cuh>

#include <vector>
#include <utility>

namespace scalable_ccd::cuda::stq {

void runBroadPhaseMultiGPU(
    const std::vector<AABB>& boxes,
    int N,
    int nbox,
    std::vector<std::pair<int, int>>& overlaps,
    int& threads,
    int& devcount);

} // namespace scalable_ccd::cuda::stq