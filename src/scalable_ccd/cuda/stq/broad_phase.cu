#include "broad_phase.cuh"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/stq/sweep.cuh>
#include <scalable_ccd/cuda/stq/util.cuh>
#include <scalable_ccd/cuda/utils/profiler.hpp>
#include <scalable_ccd/cuda/utils/device_variable.cuh>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <tbb/parallel_for.h>

#define SCALABLE_CCD_USE_CUDA_SAP // for comparison with SAP

namespace scalable_ccd::cuda::stq {

void BroadPhase::clear()
{
    *memory_handler = MemoryHandler();

    d_boxes.clear();
    d_boxes.shrink_to_fit();

    d_sm.clear();
    d_sm.shrink_to_fit();

    d_mini.clear();
    d_mini.shrink_to_fit();

    d_overlaps.clear();
    d_overlaps.shrink_to_fit();

    num_boxes_per_thread = 0;
    threads_per_block = 32;
    thread_start_box_id = 0;
    num_devices = 1;
}

const thrust::device_vector<cuda::stq::AABB>&
BroadPhase::build(const std::vector<cuda::stq::AABB>& boxes)
{
    logger().debug("Broad-phase: building (# boxes: {:d})", boxes.size());

    if (memory_handler->MAX_OVERLAP_CUTOFF == 0) {
        memory_handler->MAX_OVERLAP_CUTOFF = boxes.size();
        logger().trace(
            "Setting MAX_OVERLAP_CUTOFF to {:d}",
            memory_handler->MAX_OVERLAP_CUTOFF);
    }

    if (memory_limit_GB) {
        logger().trace("Setting memory limit to {:d} GB", memory_limit_GB);
        memory_handler->limitGB = memory_limit_GB;
    }

    setup(device_init_id, smemSize, threads_per_block, num_boxes_per_thread);
    cudaSetDevice(device_init_id);

    d_boxes = boxes; // copy to device
    d_sm.resize(boxes.size());
    d_mini.resize(boxes.size());

    // const Dimension axis = calc_sort_dimension();
    const Dimension axis = x;

    // Initialize d_sm and d_mini
    {
        SCALABLE_CCD_GPU_PROFILE_POINT("splitBoxes");
        splitBoxes<<<grid_dim_1d(), threads_per_block>>>(
            thrust::raw_pointer_cast(d_boxes.data()),
            thrust::raw_pointer_cast(d_sm.data()),
            thrust::raw_pointer_cast(d_mini.data()), d_boxes.size(), axis);
        gpuErrchk(cudaDeviceSynchronize());
    }

    {
        SCALABLE_CCD_GPU_PROFILE_POINT("sortingBoxes");
        // Only sort the split boxes and keep the d_boxes unsorted
        thrust::sort_by_key(
            thrust::device, d_sm.begin(), d_sm.end(), d_mini.begin(),
            sort_aabb_x());
    }

    gpuErrchk(cudaGetLastError());

    return d_boxes;
}

const thrust::device_vector<int2>& BroadPhase::detect_overlaps_partial()
{
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
            SCALABLE_CCD_GPU_PROFILE_POINT("runSTQ");

#ifdef SCALABLE_CCD_USE_CUDA_SAP
            runSAP<<<grid_dim_1d(), threads_per_block>>>(
                thrust::raw_pointer_cast(d_sm.data()),
                thrust::raw_pointer_cast(d_mini.data()), num_boxes(),
                thread_start_box_id, d_overlaps_buffer, &d_memory_handler);
#else
            runSTQ<<<grid_dim_1d(), threads_per_block>>>(
                thrust::raw_pointer_cast(d_sm.data()),
                thrust::raw_pointer_cast(d_mini.data()), num_boxes(),
                thread_start_box_id, d_overlaps_buffer, &d_memory_handler);
#endif

            gpuErrchk(cudaDeviceSynchronize());
        }

        *memory_handler = d_memory_handler;

        if (d_overlaps_buffer.size() < memory_handler->real_count) {
            logger().debug(
                "Found {:d} overlaps, but {:d} exist; re-running.",
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

        const int n = overlaps.size();
        overlaps.resize(n + d_overlaps.size());

        gpuErrchk(cudaMemcpy(
            &overlaps[n], thrust::raw_pointer_cast(d_overlaps.data()),
            d_overlaps.size() * sizeof(int2), cudaMemcpyDeviceToHost));
    }

    logger().debug("Complete overlaps size {:d}", overlaps.size());

    return overlaps;
}

// ----------------------------------------------------------------------------

Dimension BroadPhase::calc_sort_dimension() const
{
    // mean of all box points (used to find best axis)
    thrust::device_vector<Scalar3> d_mean(1, make_Scalar3(0, 0, 0));
    calc_mean<<<grid_dim_1d(), threads_per_block, smemSize>>>(
        thrust::raw_pointer_cast(d_boxes.data()), d_boxes.size(),
        thrust::raw_pointer_cast(d_mean.data()));

    // temporary
    const Scalar3 mean = d_mean[0];
    logger().trace("mean: x {:.6f} y {:.6f} z {:.6f}", mean.x, mean.y, mean.z);

    // calculate variance and determine which axis to sort on
    DeviceVariable<Scalar3> d_variance(make_Scalar3(0, 0, 0));

    calc_variance<<<grid_dim_1d(), threads_per_block, smemSize>>>(
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

} // namespace scalable_ccd::cuda::stq