#include <cuda/pipeline>

// #include <scalable_ccd/cuda/stq/aabb.cuh>
#include <scalable_ccd/cuda/stq/queue.cuh>
#include <scalable_ccd/cuda/stq/sweep.cuh>
#include <scalable_ccd/utils/logger.hpp>

namespace scalable_ccd::cuda::stq {

__global__ void retrieve_collision_pairs(
    const Aabb* const boxes,
    int* count,
    int2* overlaps,
    int num_boxes,
    int guess,
    int nbox,
    int start,
    int end)
{
    extern __shared__ Aabb s_objects[];
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

    const Aabb& a = s_objects[l];
    Aabb b = nltid < blockDim.x ? s_objects[nltid] : boxes[ntid];
    int i = 0;
    while (a.max.x >= b.min.x) // boxes can touch and collide
    {
        i++;
        if (does_collide(a, b) && !covertex(a.vertexIds, b.vertexIds))
            add_overlap(a.id, b.id, count, overlaps, guess);

        ntid++;
        nltid++;
        if (ntid >= num_boxes)
            return;
        b = nltid < blockDim.x ? s_objects[nltid] : boxes[ntid];
    }
    if (tid == 0)
        printf("final count for box 0: %i\n", i);
}

__global__ void
calc_mean(const Aabb* const boxes, const int num_boxes, Scalar3* mean)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_boxes)
        return;

    // add to mean

    // min + max / 2 / num_boxes
    const Scalar3 mx =
        __fdividef(boxes[tid].min + boxes[tid].max, 2 * num_boxes);
    atomicAdd(&mean[0].x, mx.x);
    atomicAdd(&mean[0].y, mx.y);
    atomicAdd(&mean[0].z, mx.z);
}

__global__ void calc_variance(
    const Aabb* const boxes,
    const int num_boxes,
    const Scalar3* const mean,
    Scalar3* var)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_boxes)
        return;

    // |min - mean|² + |max - mean|²
    const Scalar3 fx = __powf(abs(boxes[tid].min - mean[0]), 2.0)
        + __powf(abs(boxes[tid].max - mean[0]), 2.0);
    atomicAdd(&var[0].x, fx.x);
    atomicAdd(&var[0].y, fx.y);
    atomicAdd(&var[0].z, fx.z);
}

__global__ void splitBoxes(
    const Aabb* const boxes,
    Scalar2* sortedmin,
    MiniBox* mini,
    const int num_boxes,
    const Dimension axis)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_boxes)
        return;

    Scalar* min;
    Scalar* max;

    if (axis == x) {
        sortedmin[tid] = make_Scalar2(boxes[tid].min.x, boxes[tid].max.x);
        min = (Scalar[2]) { boxes[tid].min.y, boxes[tid].min.z };
        max = (Scalar[2]) { boxes[tid].max.y, boxes[tid].max.z };
    } else if (axis == y) {
        sortedmin[tid] = make_Scalar2(boxes[tid].min.y, boxes[tid].max.y);
        min = (Scalar[2]) { boxes[tid].min.x, boxes[tid].min.z };
        max = (Scalar[2]) { boxes[tid].max.x, boxes[tid].max.z };
    } else {
        sortedmin[tid] = make_Scalar2(boxes[tid].min.z, boxes[tid].max.z);
        min = (Scalar[2]) { boxes[tid].min.x, boxes[tid].min.y };
        max = (Scalar[2]) { boxes[tid].max.x, boxes[tid].max.y };
    }

    mini[tid] = MiniBox(tid, min, max, boxes[tid].vertexIds);
}

__global__ void runSAPVanilla(
    const Scalar2* const boxMinor,
    const MiniBox* const boxVerts,
    const int num_boxes,
    int2* overlaps,
    int* count,
    int* start,
    MemoryHandler* memory_handler)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x + *start;
    int ntid = tid + 1;
    int delta = 1;

    if (tid >= memory_handler->MAX_OVERLAP_CUTOFF + *start)
        return;

    if (tid >= num_boxes || ntid >= num_boxes)
        return;

    const Scalar& a = boxMinor[tid];
    // Scalar b_min_x = tid + 1 < num_boxes
    //     ? __shfl_sync(0xffffffff, a.min.x, threadIdx.x + 1)
    //     : boxes[ntid].min.x;
    // Scalar2 b = boxMinor[ntid];
    Scalar b_x;
    b_x = __shfl_down_sync(0xffffffff, a.x, delta);
    b_x = boxMinor[ntid].x;

    MiniBox amini = boxVerts[tid];
    MiniBox bmini = boxVerts[ntid];

    while (a.y >= b_x && ntid < num_boxes) {
        if (does_collide(amini, bmini)
            && is_valid_pair(amini.vertexIds, bmini.vertexIds)
            && !covertex(amini.vertexIds, bmini.vertexIds)) {
            add_overlap(
                amini.id, bmini.id, count, overlaps, start, memory_handler);
        }

        ntid++;
        delta++;
        if (ntid < num_boxes) {
            b_x = __shfl_down_sync(0xffffffff, a.x, delta);
            b_x = boxMinor[ntid].x;
            bmini = boxVerts[ntid];
        }
    }
}

__global__ void runSTQ(
    const Scalar2* const sortedMajorAxis,
    const MiniBox* const boxVerts,
    const int num_boxes,
    int2* overlaps,
    int* count,
    int* start,
    MemoryHandler* memory_handler)
{
    // Initialize shared queue for threads to push collisions onto
    // __shared__ Queue queue; // WARNING: This results in a compiler warning
    Queue queue;
    queue.heap_size = HEAP_SIZE;
    queue.start = 0;
    queue.end = 0;

    int tid = threadIdx.x + blockIdx.x * blockDim.x + *start;

    if (tid >= num_boxes || tid + 1 >= num_boxes)
        return;

    // If the number of boxes is to large for gpu memory, split the workload and
    // start where left off
    if (tid >= memory_handler->MAX_OVERLAP_CUTOFF + *start)
        return;

    Scalar amax = sortedMajorAxis[tid].y;
    Scalar bmin = sortedMajorAxis[tid + 1].x;

    // check if tid and tid+1 boxes collide on major axis
    // if they do, push them onto the queue
    if (amax >= bmin) {
        int2 val = make_int2(tid, tid + 1);
        queue.push(val);
    }
    __syncthreads();
    queue.nbr_per_loop = queue.end - queue.start;

    // Retrieve the next pair of boxes from the queue and check if they collide
    // along non-major axes
    while (queue.nbr_per_loop > 0) {
        if (threadIdx.x >= queue.nbr_per_loop)
            return;
        int2 res = queue.pop();
        MiniBox ax = boxVerts[res.x];
        MiniBox bx = boxVerts[res.y];

        // Check for collision, matching simplex pair (edge-edge, vertex-face)
        // and not sharing same vertex
        if (does_collide(ax, bx) && is_valid_pair(ax.vertexIds, bx.vertexIds)
            && !covertex(ax.vertexIds, bx.vertexIds)) {
            add_overlap(ax.id, bx.id, count, overlaps, start, memory_handler);
        }

        // Repeat major axis check and push to queue if they collide
        if (res.y + 1 >= num_boxes)
            return;
        amax = sortedMajorAxis[res.x].y;
        bmin = sortedMajorAxis[res.y + 1].x;
        if (amax >= bmin) {
            res.y += 1;
            queue.push(res);
        }
        __syncthreads();
        // Update the number of boxes to be processed in the queue
        queue.nbr_per_loop = (queue.end - queue.start + HEAP_SIZE) % HEAP_SIZE;
    }
}

} // namespace scalable_ccd::cuda::stq