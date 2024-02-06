#include "sweep.cuh"

namespace scalable_ccd::cuda {

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

__global__ void retrieve_collision_pairs(
    const AABB* const boxes,
    int* count,
    int2* overlaps,
    int num_boxes,
    int guess,
    int nbox,
    int start,
    int end)
{
    extern __shared__ AABB s_objects[];
    int tid = start + threadIdx.x + blockIdx.x * blockDim.x;
    int ltid = threadIdx.x;

    if (tid >= num_boxes || tid >= end)
        return;
    s_objects[ltid] = boxes[tid];

    __syncthreads();

    int t = tid + 0 * blockDim.x;
    int l = 0 * blockDim.x + ltid;

    int ntid = t + 1;
    int nltid = l + 1;

    if (ntid >= num_boxes)
        return;

    const AABB& a = s_objects[l];
    AABB b = nltid < blockDim.x ? s_objects[nltid] : boxes[ntid];
    int i = 0;
    while (a.max.x >= b.min.x) // boxes can touch and collide
    {
        i++;
        if (does_collide(a, b) && !covertex(a.vertex_ids, b.vertex_ids)) {
            add_overlap(a.box_id, b.box_id, guess, overlaps, count);
        }

        ntid++;
        nltid++;
        if (ntid >= num_boxes)
            return;
        b = nltid < blockDim.x ? s_objects[nltid] : boxes[ntid];
    }
    if (tid == 0)
        printf("final count for box 0: %i\n", i);
}

} // namespace scalable_ccd::cuda