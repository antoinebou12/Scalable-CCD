#include <cuda/pipeline>

// #include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/broad_phase/queue.cuh>
#include <scalable_ccd/cuda/broad_phase/sweep.cuh>
#include <scalable_ccd/utils/logger.hpp>

namespace scalable_ccd::cuda::stq {

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

__global__ void
calc_mean(const AABB* const boxes, const int num_boxes, Scalar3* mean)
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
    const AABB* const boxes,
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
    const AABB* const boxes,
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

    mini[tid] = MiniBox(min, max, boxes[tid].vertex_ids, tid);
}

__global__ void runSAP(
    const Scalar2* const sortedMajorAxis,
    const MiniBox* const boxVerts,
    const int num_boxes,
    const int start_box_id,
    RawDeviceBuffer<int2> overlaps,
    MemoryHandler* memory_handler)
{
    const int box_id = threadIdx.x + blockIdx.x * blockDim.x + start_box_id;

    if (box_id >= start_box_id + memory_handler->MAX_OVERLAP_CUTOFF)
        return;

    int next_box_id = box_id + 1;
    int delta = 1;

    if (box_id >= num_boxes || next_box_id >= num_boxes)
        return;

    const Scalar2& a = sortedMajorAxis[box_id];

    Scalar b_x;
    b_x = __shfl_down_sync(0xffffffff, a.x, delta); // ???
    b_x = sortedMajorAxis[next_box_id].x;

    const MiniBox a_mini = boxVerts[box_id];
    MiniBox b_mini = boxVerts[next_box_id];

    while (a.y >= b_x && next_box_id < num_boxes) {
        if (does_collide(a_mini, b_mini)
            && AABB::is_valid_pair(a_mini.vertex_ids, b_mini.vertex_ids)
            && !covertex(a_mini.vertex_ids, b_mini.vertex_ids)) {
            add_overlap(
                a_mini.box_id, b_mini.box_id, overlaps,
                memory_handler->real_count);
        }

        next_box_id++;
        delta++;
        if (next_box_id < num_boxes) {
            b_x = __shfl_down_sync(0xffffffff, a.x, delta);
            b_x = sortedMajorAxis[next_box_id].x;
            b_mini = boxVerts[next_box_id];
        }
    }
}

__global__ void runSTQ(
    const Scalar2* const sortedMajorAxis,
    const MiniBox* const boxVerts,
    const int num_boxes,
    const int start_box_id,
    RawDeviceBuffer<int2> overlaps,
    MemoryHandler* memory_handler)
{
    // Initialize shared queue for threads to push collisions onto
    __shared__ Queue queue; // WARNING: This results in a compiler warning
    queue.heap_size = HEAP_SIZE;
    queue.start = 0;
    queue.end = 0;

    const int box_id = threadIdx.x + blockIdx.x * blockDim.x + start_box_id;
    if (box_id >= num_boxes || box_id + 1 >= num_boxes)
        return;

    // If the number of boxes is to large for gpu memory, split the workload and
    // start where left off.
    if (box_id >= memory_handler->MAX_OVERLAP_CUTOFF + start_box_id)
        return;

    Scalar a_max = sortedMajorAxis[box_id].y;
    Scalar b_min = sortedMajorAxis[box_id + 1].x;

    // check if box_id and box_id+1 boxes collide on major axis
    // if they do, push them onto the queue.
    if (a_max >= b_min) {
        queue.push(make_int2(box_id, box_id + 1));
    }
    __syncthreads();
    queue.nbr_per_loop = queue.end - queue.start;

    // Retrieve the next pair of boxes from the queue and check if they collide
    // along non-major axes.
    while (queue.nbr_per_loop > 0) {
        if (threadIdx.x >= queue.nbr_per_loop)
            return;
        int2 res = queue.pop();
        MiniBox ax = boxVerts[res.x];
        MiniBox bx = boxVerts[res.y];

        // Check for collision, matching simplex pair (edge-edge, vertex-face)
        // and not sharing same vertex.
        if (does_collide(ax, bx)
            && AABB::is_valid_pair(ax.vertex_ids, bx.vertex_ids)
            && !covertex(ax.vertex_ids, bx.vertex_ids)) {
            add_overlap(
                ax.box_id, bx.box_id, overlaps, memory_handler->real_count);
        }

        // Repeat major axis check and push to queue if they collide.
        if (res.y + 1 >= num_boxes)
            return;

        a_max = sortedMajorAxis[res.x].y;
        b_min = sortedMajorAxis[res.y + 1].x;
        if (a_max >= b_min) {
            res.y += 1;
            queue.push(res);
        }
        __syncthreads();
        // Update the number of boxes to be processed in the queue
        queue.nbr_per_loop = (queue.end - queue.start + HEAP_SIZE) % HEAP_SIZE;
    }
}

} // namespace scalable_ccd::cuda::stq