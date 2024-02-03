#pragma once

#include <scalable_ccd/cuda/broad_phase/aabb.cuh>

namespace scalable_ccd::cuda::stq {

struct sort_aabb_x {
    __device__ bool operator()(const AABB& a, const AABB& b) const
    {
        return (a.min.x < b.min.x);
    }

    __device__ bool operator()(const Scalar3& a, const Scalar3& b) const
    {
        return (a.x < b.x);
    }

    __device__ bool operator()(const Scalar2& a, const Scalar2& b) const
    {
        return (a.x < b.x);
    }
};

void setup(
    int device_id,
    int& shared_memory_size,
    int& threads,
    int& boxes_per_thread);

} // namespace scalable_ccd::cuda::stq