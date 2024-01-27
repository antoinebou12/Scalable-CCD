#pragma once

#include <scalable_ccd/stq/cuda/aabb.cuh>
#include <scalable_ccd/stq/cuda/memory.cuh>

#include <vector>
#include <utility>

namespace stq::gpu {

void runBroadPhaseMultiGPU(
    const Aabb* boxes,
    int N,
    int nbox,
    std::vector<std::pair<int, int>>& finOverlaps,
    int& threads,
    int& devcount);

void runBroadPhase(
    const Aabb* boxes,
    MemHandler* memhandle,
    int N,
    int nbox,
    std::vector<std::pair<int, int>>& finOverlaps,
    int2*& d_overlaps,
    int*& d_count,
    int& threads,
    int& tidstart,
    int& devcount,
    const int memlimit = 0);
} // namespace stq::gpu