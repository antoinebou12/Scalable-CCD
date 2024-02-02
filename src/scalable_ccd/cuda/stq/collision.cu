#include <scalable_ccd/cuda/stq/collision.cuh>

namespace scalable_ccd::cuda::stq {

__device__ bool does_collide(const AABB& a, const AABB& b)
{
    return
        //    a.max.x >= b.min.x && a.min.x <= b.max.x &&
        a.max.y >= b.min.y && a.min.y <= b.max.y && a.max.z >= b.min.z
        && a.min.z <= b.max.z;
}

__device__ bool does_collide(const MiniBox& a, const MiniBox& b)
{
    return a.max.x >= b.min.x && a.min.x <= b.max.x && a.max.y >= b.min.y
        && a.min.y <= b.max.y;
}

__device__ bool covertex(const int3& a, const int3& b)
{

    return a.x == b.x || a.x == b.y || a.x == b.z || a.y == b.x || a.y == b.y
        || a.y == b.z || a.z == b.x || a.z == b.y || a.z == b.z;
}

__device__ void add_overlap(
    const int xid,
    const int yid,
    const int max_overlap_size,
    int2* overlaps,
    int* count)
{
    int i = atomicAdd(count, 1);

    if (i < max_overlap_size) {
        overlaps[i] = make_int2(xid, yid);
    }
}

__device__ void add_overlap(
    const int xid,
    const int yid,
    RawDeviceBuffer<int2>& overlaps,
    int& real_count)
{
    if (atomicAdd(&real_count, 1) < overlaps.capacity) {
        overlaps.push(make_int2(xid, yid));
    }
}

} // namespace scalable_ccd::cuda::stq