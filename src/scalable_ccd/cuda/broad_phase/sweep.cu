#include <cuda/pipeline>

// #include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/broad_phase/queue.cuh>
#include <scalable_ccd/cuda/broad_phase/sweep.cuh>
#include <scalable_ccd/utils/logger.hpp>

namespace scalable_ccd::cuda {

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

// -----------------------------------------------------------------------------

template <bool is_two_lists>
__global__ void sweep_and_prune(
    const Scalar2* const sorted_major_axis,
    const MiniBox* const mini_boxes,
    const int num_boxes,
    const int start_box_id,
    RawDeviceBuffer<int2> overlaps,
    MemoryHandler* memory_handler)
{
    const int box_id = threadIdx.x + blockIdx.x * blockDim.x + start_box_id;

    if (box_id >= start_box_id + memory_handler->MAX_OVERLAP_CUTOFF)
        return;

    int next_box_id = box_id + 1;

    if (box_id >= num_boxes || next_box_id >= num_boxes)
        return;

    const Scalar2 a = sorted_major_axis[box_id];
    const MiniBox a_mini = mini_boxes[box_id];

    Scalar b_x = sorted_major_axis[next_box_id].x;
    MiniBox b_mini = mini_boxes[next_box_id];

    while (a.y >= b_x && next_box_id < num_boxes) {
        if (is_valid_pair<is_two_lists>(a_mini.element_id, b_mini.element_id)
            && a_mini.intersects(b_mini)
            && !share_a_vertex(a_mini.vertex_ids, b_mini.vertex_ids)) {

            if constexpr (is_two_lists) {
                // Negative IDs are from the first list
                add_overlap(
                    flip_id(min(a_mini.element_id, b_mini.element_id)),
                    max(a_mini.element_id, b_mini.element_id), //
                    overlaps, memory_handler->real_count);
            } else {
                assert(a_mini.element_id >= 0 && b_mini.element_id >= 0);
                add_overlap(
                    min(a_mini.element_id, b_mini.element_id),
                    max(a_mini.element_id, b_mini.element_id), //
                    overlaps, memory_handler->real_count);
            }
        }

        ++next_box_id;
        if (next_box_id < num_boxes) {
            b_x = sorted_major_axis[next_box_id].x;
            b_mini = mini_boxes[next_box_id];
        }
    }
}

template <bool is_two_lists>
__global__ void sweep_and_tiniest_queue(
    const Scalar2* const sorted_major_axis,
    const MiniBox* const mini_boxes,
    const int num_boxes,
    const int start_box_id,
    RawDeviceBuffer<int2> overlaps,
    MemoryHandler* memory_handler)
{
    assert(blockDim.x <= warpSize); // Allow for warp-synchronous programming

    // Initialize shared queue for threads to push collisions onto
    __shared__ Queue queue;
    queue.start = 0;
    queue.end = 0;

    const int box_id = threadIdx.x + blockIdx.x * blockDim.x + start_box_id;
    if (box_id >= num_boxes || box_id + 1 >= num_boxes)
        return;

    // If the number of boxes is to large for gpu memory, split the workload and
    // start where left off.
    if (box_id >= memory_handler->MAX_OVERLAP_CUTOFF + start_box_id)
        return;

    Scalar a_max = sorted_major_axis[box_id].y;
    Scalar b_min = sorted_major_axis[box_id + 1].x;

    // If box_id and box_id+1 boxes collide on major axis, then push them onto
    // the queue.
    if (a_max >= b_min) {
        queue.push(make_int2(box_id, box_id + 1));
    }
    __syncwarp();
    queue.nbr_per_loop = queue.end - queue.start;

    // Retrieve the next pair of boxes from the queue and check if they collide
    // along non-major axes.
    while (queue.nbr_per_loop > 0) {
        if (threadIdx.x >= queue.nbr_per_loop)
            return;
        int2 res = queue.pop();
        MiniBox a_mini = mini_boxes[res.x];
        MiniBox b_mini = mini_boxes[res.y];

        // Check for collision, matching simplex pair (edge-edge, vertex-face)
        // and not sharing same vertex.
        if (is_valid_pair<is_two_lists>(a_mini.element_id, b_mini.element_id)
            && a_mini.intersects(b_mini)
            && !share_a_vertex(a_mini.vertex_ids, b_mini.vertex_ids)) {

            if constexpr (is_two_lists) {
                // Negative IDs are from the first list
                add_overlap(
                    flip_id(min(a_mini.element_id, b_mini.element_id)),
                    max(a_mini.element_id, b_mini.element_id), overlaps,
                    memory_handler->real_count);
            } else {
                assert(a_mini.element_id >= 0 && b_mini.element_id >= 0);
                add_overlap(
                    min(a_mini.element_id, b_mini.element_id),
                    max(a_mini.element_id, b_mini.element_id), overlaps,
                    memory_handler->real_count);
            }
        }

        // Repeat major axis check and push to queue if they collide.
        if (res.y + 1 >= num_boxes)
            return;

        a_max = sorted_major_axis[res.x].y;
        b_min = sorted_major_axis[res.y + 1].x;
        if (a_max >= b_min) {
            ++res.y;
            queue.push(res);
        }
        __syncwarp();
        // Update the number of boxes to be processed in the queue
        queue.nbr_per_loop =
            (queue.end - queue.start + QUEUE_SIZE) % QUEUE_SIZE;
    }
}

// === Template instantiation =================================================

template __global__ void sweep_and_prune<false>(
    const Scalar2* const,
    const MiniBox* const,
    const int,
    const int,
    RawDeviceBuffer<int2>,
    MemoryHandler*);
template __global__ void sweep_and_prune<true>(
    const Scalar2* const,
    const MiniBox* const,
    const int,
    const int,
    RawDeviceBuffer<int2>,
    MemoryHandler*);

template __global__ void sweep_and_tiniest_queue<false>(
    const Scalar2* const,
    const MiniBox* const,
    const int,
    const int,
    RawDeviceBuffer<int2>,
    MemoryHandler*);
template __global__ void sweep_and_tiniest_queue<true>(
    const Scalar2* const,
    const MiniBox* const,
    const int,
    const int,
    RawDeviceBuffer<int2>,
    MemoryHandler*);

} // namespace scalable_ccd::cuda