#pragma once

#include <scalable_ccd/cuda/broad_phase/aabb.cuh>

namespace scalable_ccd::cuda {

enum Dimension { x, y, z };

struct SortIntervals {
    __device__ bool operator()(const Scalar2& a, const Scalar2& b) const
    {
        return a.x < b.x;
    }
};

void setup(
    int device_id,
    int& shared_memory_size,
    int& threads,
    int& boxes_per_thread);

} // namespace scalable_ccd::cuda