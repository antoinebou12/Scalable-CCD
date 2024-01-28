#pragma once

#include <scalable_ccd/cuda/stq/collision.cuh>

namespace scalable_ccd::cuda::stq {

__global__ void retrieve_collision_pairs(
    const Aabb* const boxes,
    int* count,
    int2* overlaps,
    int N,
    int guess,
    int nbox,
    int start = 0,
    int end = INT_MAX);

// for balancing
__global__ void splitBoxes(
    Aabb* boxes, Scalar2* sortedmin, MiniBox* mini, int N, Dimension axis);

__global__ void calc_variance(Aabb* boxes, Scalar3* var, int N, Scalar3* mean);

__global__ void calc_mean(Aabb* boxes, Scalar3* mean, int N);

__global__ void runSTQ(
    Scalar2* sm,
    const MiniBox* const mini,
    int2* overlaps,
    int N,
    int* count,
    int* start,
    MemoryHandler* memory_handler);

__global__ void runSAPVanilla(
    Scalar2* xbox,
    const MiniBox* const boxVerts,
    int2* overlaps,
    int N,
    int* count,
    int* start,
    MemoryHandler* mem);

} // namespace scalable_ccd::cuda::stq