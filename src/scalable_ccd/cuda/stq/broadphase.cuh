#pragma once

#include <scalable_ccd/cuda/memory_handler.cuh>
#include <scalable_ccd/cuda/stq/aabb.cuh>

#include <vector>
#include <utility>

namespace scalable_ccd::cuda::stq {

void runBroadPhaseMultiGPU(
    const Aabb* boxes,
    int N,
    int nbox,
    std::vector<std::pair<int, int>>& finOverlaps,
    int& threads,
    int& devcount);

void runBroadPhase(
    const Aabb* boxes,
    MemoryHandler* memory_handler,
    int N,
    int nbox,
    std::vector<std::pair<int, int>>& finOverlaps,
    int2*& d_overlaps,
    int*& d_count,
    int& threads,
    int& tidstart,
    int& devcount,
    const int memlimit = 0);
} // namespace scalable_ccd::cuda::stq