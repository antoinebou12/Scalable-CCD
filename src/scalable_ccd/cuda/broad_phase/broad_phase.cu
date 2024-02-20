#include "broad_phase.cuh"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/broad_phase/sweep.cuh>
#include <scalable_ccd/cuda/broad_phase/utils.cuh>
#include <scalable_ccd/cuda/broad_phase/collision.cuh>
#include <scalable_ccd/utils/profiler.hpp>
#include <scalable_ccd/cuda/utils/device_variable.cuh>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <tbb/parallel_for.h>

// #define SCALABLE_CCD_USE_CUDA_SAP // for comparison with SAP

namespace scalable_ccd::cuda {

namespace {
    __global__ void flip_element_ids(MiniBox* const boxes, int size)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            boxes[idx].element_id = flip_id(boxes[idx].element_id);
        }
    }
} // namespace

void BroadPhase::build(const std::shared_ptr<DeviceAABBs> boxes)
{
    assert(boxes);
    assert(thrust::is_sorted(
        boxes->sorted_major_intervals.begin(),
        boxes->sorted_major_intervals.end(), SortIntervals()));

    clear();

    logger().debug("Broad-phase: building (# boxes: {:d})", boxes->size());

    if (memory_handler->MAX_OVERLAP_CUTOFF == 0) {
        memory_handler->MAX_OVERLAP_CUTOFF = boxes->size();
        logger().trace(
            "Setting MAX_OVERLAP_CUTOFF to {:d}",
            memory_handler->MAX_OVERLAP_CUTOFF);
    }

    setup(
        device_init_id, shared_memory_size, threads_per_block,
        num_boxes_per_thread);
    cudaSetDevice(device_init_id);

    this->d_boxes = boxes; // fast: copying a shared pointer

    is_two_lists = false; // default
}

void BroadPhase::build(
    const std::shared_ptr<DeviceAABBs> _boxesA,
    const std::shared_ptr<DeviceAABBs> boxesB)
{
    assert(_boxesA && boxesB);
    assert(thrust::is_sorted(
        _boxesA->sorted_major_intervals.begin(),
        _boxesA->sorted_major_intervals.end(), SortIntervals()));
    assert(thrust::is_sorted(
        boxesB->sorted_major_intervals.begin(),
        boxesB->sorted_major_intervals.end(), SortIntervals()));

    // Explicit copy of boxesA to modify element_id
    DeviceAABBs boxesA;
    {
        SCALABLE_CCD_GPU_PROFILE_POINT("copy_device_aabbs");
        boxesA = *_boxesA;
    }

    {
        SCALABLE_CCD_GPU_PROFILE_POINT("flip_element_ids");
        constexpr int N = 1024;
        flip_element_ids<<<boxesA.size() / N + 1, N>>>(
            thrust::raw_pointer_cast(boxesA.mini_boxes.data()), boxesA.size());
    }

    std::shared_ptr<DeviceAABBs> boxes = std::make_shared<DeviceAABBs>();
    {
        SCALABLE_CCD_GPU_PROFILE_POINT("merge_boxes");
        boxes->sorted_major_intervals.resize(boxesA.size() + boxesB->size());
        boxes->mini_boxes.resize(boxesA.size() + boxesB->size());
        thrust::merge_by_key(
            boxesA.sorted_major_intervals.begin(),
            boxesA.sorted_major_intervals.end(),
            boxesB->sorted_major_intervals.begin(),
            boxesB->sorted_major_intervals.end(), //
            boxesA.mini_boxes.begin(), boxesB->mini_boxes.begin(),
            boxes->sorted_major_intervals.begin(), boxes->mini_boxes.begin(),
            SortIntervals());
    }

    build(boxes); // build with the merged boxes

    is_two_lists = true; // override the default
}

void BroadPhase::clear()
{
    *memory_handler = MemoryHandler();

    if (d_boxes) {
        d_boxes->clear();
        d_boxes->shrink_to_fit();
    }

    d_overlaps.clear();
    d_overlaps.shrink_to_fit();

    num_boxes_per_thread = 0;
    threads_per_block = 32;
    thread_start_box_id = 0;
    num_devices = 1;
}

const thrust::device_vector<int2>& BroadPhase::detect_overlaps_partial()
{
    if (!d_boxes) {
        throw std::runtime_error(
            "Must initialize build broad phase before detecting overlaps!");
    }

    logger().debug("Broad-phase: detecting overlaps (partial)");

    memory_handler->setOverlapSize();
    logger().trace(
        "Max overlap size: {:d} ({:g} GB)", memory_handler->MAX_OVERLAP_SIZE,
        memory_handler->MAX_OVERLAP_SIZE * sizeof(int2) / 1e9);
    logger().trace(
        "Max overlap cutoff: {:d}", memory_handler->MAX_OVERLAP_CUTOFF);

    // Device memory_handler to keep track of vars
    DeviceVariable<MemoryHandler> d_memory_handler;

    // Allocate a large chunk of memory for overlaps
    DeviceBuffer<int2> d_overlaps_buffer;
    do {
        // Allocate a large chunk of memory for overlaps
        d_overlaps_buffer.clear(); // Reset size to 0
        d_overlaps_buffer.reserve(memory_handler->MAX_OVERLAP_SIZE);

        memory_handler->real_count = 0;     // Reset real count
        d_memory_handler = *memory_handler; // Update memory handler on device

        {
#ifdef SCALABLE_CCD_USE_CUDA_SAP
            SCALABLE_CCD_GPU_PROFILE_POINT("sweep_and_prune");
            if (is_two_lists) {
                sweep_and_prune<true><<<grid_dim_1d(), threads_per_block>>>(
                    thrust::raw_pointer_cast(
                        d_boxes->sorted_major_intervals.data()),
                    thrust::raw_pointer_cast(d_boxes->mini_boxes.data()),
                    num_boxes(), thread_start_box_id, d_overlaps_buffer,
                    &d_memory_handler);
            } else {
                sweep_and_prune<false><<<grid_dim_1d(), threads_per_block>>>(
                    thrust::raw_pointer_cast(
                        d_boxes->sorted_major_intervals.data()),
                    thrust::raw_pointer_cast(d_boxes->mini_boxes.data()),
                    num_boxes(), thread_start_box_id, d_overlaps_buffer,
                    &d_memory_handler);
            }
#else
            SCALABLE_CCD_GPU_PROFILE_POINT("sweep_and_tiniest_queue");
            if (is_two_lists) {
                sweep_and_tiniest_queue<true>
                    <<<grid_dim_1d(), threads_per_block>>>(
                        thrust::raw_pointer_cast(
                            d_boxes->sorted_major_intervals.data()),
                        thrust::raw_pointer_cast(d_boxes->mini_boxes.data()),
                        num_boxes(), thread_start_box_id, d_overlaps_buffer,
                        &d_memory_handler);
            } else {
                sweep_and_tiniest_queue<false>
                    <<<grid_dim_1d(), threads_per_block>>>(
                        thrust::raw_pointer_cast(
                            d_boxes->sorted_major_intervals.data()),
                        thrust::raw_pointer_cast(d_boxes->mini_boxes.data()),
                        num_boxes(), thread_start_box_id, d_overlaps_buffer,
                        &d_memory_handler);
            }
#endif

            gpuErrchk(cudaDeviceSynchronize());
        }

        *memory_handler = d_memory_handler;

        if (d_overlaps_buffer.size() < memory_handler->real_count) {
            logger().debug(
                "Found {:d} overlaps, but {:d} exist; re-running",
                d_overlaps_buffer.size(), memory_handler->real_count);

            // Increase MAX_OVERLAP_SIZE (or decrease MAX_OVERLAP_CUTOFF)
            memory_handler->handleBroadPhaseOverflow(
                memory_handler->real_count);
        }
    } while (d_overlaps_buffer.size() < memory_handler->real_count);
    assert(memory_handler->real_count == d_overlaps_buffer.size());

    // Increase thread_start_box_id for next run
    thread_start_box_id += memory_handler->MAX_OVERLAP_CUTOFF;

    // Move overlaps from buffer to d_overlaps
    if (d_overlaps_buffer.size() > 0) {
        SCALABLE_CCD_GPU_PROFILE_POINT("copy_overlaps_from_buffer");
        d_overlaps = thrust::device_vector<int2>(
            d_overlaps_buffer.begin(), d_overlaps_buffer.end());
    } else {
        d_overlaps.clear();
    }

    logger().debug(
        "Final count for device {:d}: {:d} ({:g} GB)", 0, d_overlaps.size(),
        d_overlaps.size() * sizeof(int2) / 1e9);
    logger().trace("Next starting box id: {:d}", thread_start_box_id);

    return d_overlaps;
}

std::vector<std::pair<int, int>> BroadPhase::detect_overlaps()
{
    logger().debug("Broad-phase: detecting overlaps");

    // Increase the maximum overlap size because we only need to store the
    // overlaps on the host.
    memory_handler->setMemoryLimitForBroadPhaseOnly();

    std::vector<std::pair<int, int>> overlaps;

    while (!is_complete()) {
        detect_overlaps_partial();

        {
            SCALABLE_CCD_CPU_PROFILE_POINT("copy_overlaps_to_host");
            const int n = overlaps.size();
            overlaps.resize(n + d_overlaps.size());
            gpuErrchk(cudaMemcpy(
                &overlaps[n], thrust::raw_pointer_cast(d_overlaps.data()),
                d_overlaps.size() * sizeof(int2), cudaMemcpyDeviceToHost));
        }
    }

    logger().debug("Complete overlaps size {:d}", overlaps.size());

    return overlaps;
}

// ----------------------------------------------------------------------------

Dimension BroadPhase::calc_sort_dimension(
    const thrust::device_vector<AABB>& d_boxes) const
{
    // mean of all box points (used to find best axis)
    thrust::device_vector<Scalar3> d_mean(1, make_Scalar3(0, 0, 0));
    calc_mean<<<grid_dim_1d(), threads_per_block>>>(
        thrust::raw_pointer_cast(d_boxes.data()), d_boxes.size(),
        thrust::raw_pointer_cast(d_mean.data()));

    // temporary
    const Scalar3 mean = d_mean[0];
    logger().trace("mean: x {:.6f} y {:.6f} z {:.6f}", mean.x, mean.y, mean.z);

    // calculate variance and determine which axis to sort on
    DeviceVariable<Scalar3> d_variance(make_Scalar3(0, 0, 0));

    calc_variance<<<grid_dim_1d(), threads_per_block>>>(
        thrust::raw_pointer_cast(d_boxes.data()), d_boxes.size(),
        thrust::raw_pointer_cast(d_mean.data()), &d_variance);
    cudaDeviceSynchronize();

    const Scalar3 variance = d_variance;
    logger().trace(
        "var: x {:.6f} y {:.6f} z {:.6f}", variance.x, variance.y, variance.z);
    const Scalar max_variance =
        std::max({ variance.x, variance.y, variance.z });

    Dimension axis;
    if (max_variance == variance.x) {
        axis = x;
    } else if (max_variance == variance.y) {
        axis = y;
    } else {
        axis = z;
    }
    logger().trace("Axis: {:s}", axis == x ? "x" : (axis == y ? "y" : "z"));
    return axis;
}

} // namespace scalable_ccd::cuda