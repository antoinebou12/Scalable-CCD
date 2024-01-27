#include "broadphase.cuh"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/memory_handler.cuh>
#include <scalable_ccd/cuda/stq/broadphase.cuh>
#include <scalable_ccd/cuda/stq/collision.cuh>
#include <scalable_ccd/cuda/stq/queue.cuh>
#include <scalable_ccd/cuda/stq/sweep.cuh>
#include <scalable_ccd/cuda/stq/timer.cuh>
#include <scalable_ccd/cuda/stq/util.cuh>
#include <scalable_ccd/cuda/stq/io.cuh>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

#include <spdlog/spdlog.h>

namespace scalable_ccd::cuda::stq {

extern MemoryHandler* memory_handler;

void runBroadPhaseMultiGPU(
    const Aabb* boxes,
    int N,
    int nbox,
    std::vector<std::pair<int, int>>& finOverlaps,
    int& threads,
    int& devcount)
{
    spdlog::critical("default threads {}", tbb::info::default_concurrency());
    ThreadSpecificOverlaps storages;

    float milliseconds = 0;
    int device_init_id = 0;

    int smemSize;
    setup(device_init_id, smemSize, threads, nbox);

    cudaSetDevice(device_init_id);

    finOverlaps.clear();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate boxes to GPU
    Aabb* d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb) * N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);

    dim3 block(threads);
    int grid_dim_1d = (N / threads + 1);
    dim3 grid(grid_dim_1d);
    spdlog::trace("Grid dim (1D): {:d}", grid_dim_1d);
    spdlog::trace("Box size: {:d}", sizeof(Aabb));

    // Thrust sort (can be improved by sort_by_key)
    cudaEventRecord(start);
    try {
        thrust::sort(thrust::device, d_boxes, d_boxes + N, sort_aabb_x());
    } catch (thrust::system_error& e) {
        spdlog::trace("Error: {:s} ", e.what());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    spdlog::trace("Elapsed time for sort: {:.6f} ms", milliseconds);
    cudaDeviceSynchronize();

    int devices_count;
    cudaGetDeviceCount(&devices_count);
    devices_count = devcount ? devcount : devices_count;
    int range = ceil((float)N / devices_count);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEvent_t starts[devices_count];
    cudaEvent_t stops[devices_count];
    float millisecondss[devices_count];

    tbb::parallel_for(0, devices_count, 1, [&](int& device_id) {
        auto& local_overlaps = storages.local();

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        spdlog::trace(
            "{:s} -> unifiedAddressing = {:d}", prop.name,
            prop.unifiedAddressing);

        cudaSetDevice(device_id);

        // cudaEvent_t start, stop;
        cudaEventCreate(&starts[device_id]);
        cudaEventCreate(&stops[device_id]);

        int is_able;

        cudaDeviceCanAccessPeer(&is_able, device_id, device_init_id);
        cudaDeviceSynchronize();
        if (is_able) {
            cudaDeviceEnablePeerAccess(device_init_id, 0);
            cudaDeviceSynchronize();
        } else if (device_init_id != device_id)
            spdlog::trace(
                "Device {:d} cant access Device {:d}", device_id,
                device_init_id);

        int range_start = range * device_id;
        int range_end = range * (device_id + 1);
        spdlog::trace(
            "device_id: {:d} [{:d}, {:d})", device_id, range_start, range_end);

        Aabb* d_b;
        cudaMalloc((void**)&d_b, sizeof(Aabb) * N);
        cudaMemcpy(d_b, d_boxes, sizeof(Aabb) * N, cudaMemcpyDefault);
        cudaDeviceSynchronize();

        cudaDeviceCanAccessPeer(&is_able, device_id, device_init_id);
        cudaDeviceSynchronize();
        if (is_able) {
            cudaDeviceDisablePeerAccess(device_init_id);
            cudaDeviceSynchronize();
        } else if (device_init_id != device_id)
            spdlog::trace(
                "Device {:d} cant access Device {:d}", device_id,
                device_init_id);

        // Allocate counter to GPU + set to 0 collisions
        int* d_count;
        gpuErrchk(cudaMalloc((void**)&d_count, sizeof(int)));
        gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));
        gpuErrchk(cudaGetLastError());

        // Find overlapping pairs
        int guess = N * 200;
        spdlog::trace("Guess {:d}", guess);

        int2* d_overlaps;
        cudaMalloc((void**)&d_overlaps, sizeof(int2) * (guess));
        gpuErrchk(cudaGetLastError());

        int grid_dim_1d = (range / threads + 1);
        dim3 grid(grid_dim_1d);

        int count;
        cudaEventRecord(starts[device_id]);
        retrieve_collision_pairs<<<grid, block, smemSize>>>(
            d_b, d_count, d_overlaps, N, guess, nbox, range_start, range_end);
        cudaEventRecord(stops[device_id]);
        cudaEventSynchronize(stops[device_id]);
        cudaEventElapsedTime(
            &millisecondss[device_id], starts[device_id], stops[device_id]);
        cudaDeviceSynchronize();
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        spdlog::trace("count for device {:d} : {:d}", device_id, count);

        if (count > guess) {
            spdlog::trace("Running again");
            cudaFree(d_overlaps);
            cudaMalloc((void**)&d_overlaps, sizeof(int2) * (count));
            // cudaMemset(d_overlaps, 0, sizeof(int2)*(count));
            cudaMemset(d_count, 0, sizeof(int));
            cudaEventRecord(starts[device_id]);
            retrieve_collision_pairs<<<grid, block, smemSize>>>(
                d_b, d_count, d_overlaps, N, count, nbox, range_start,
                range_end);
            cudaEventRecord(stops[device_id]);
            cudaEventSynchronize(stops[device_id]);
            cudaEventElapsedTime(
                &millisecondss[device_id], starts[device_id], stops[device_id]);
            cudaDeviceSynchronize();
            cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
            spdlog::trace("count2 for device {:d} : {:d}", device_id, count);
        }
        int2* overlaps = (int2*)malloc(sizeof(int2) * count);
        gpuErrchk(cudaMemcpy(
            overlaps, d_overlaps, sizeof(int2) * (count),
            cudaMemcpyDeviceToHost));
        gpuErrchk(cudaGetLastError());

        spdlog::trace("Final count for device {:d}:  {:d}", device_id, count);

        for (size_t i = 0; i < count; i++) {
            // local_overlaps.emplace_back(overlaps[i].x, overlaps[i].y);
            // finOverlaps.push_back();
            int aid = overlaps[i].x;
            int bid = overlaps[i].y;
            Aabb a = boxes[aid];
            Aabb b = boxes[bid];

            if (is_vertex(a) && is_face(b)) // vertex, face
                local_overlaps.emplace_back(aid, bid);
            else if (is_edge(a) && is_edge(b))
                local_overlaps.emplace_back(min(aid, bid), max(aid, bid));
            else if (is_face(a) && is_vertex(b))
                local_overlaps.emplace_back(bid, aid);
        }

        spdlog::trace(
            "Total(filt.) overlaps for devid {:d}: {:d}", device_id,
            local_overlaps.size());
    }); // end tbb for loop

    merge_local_overlaps(storages, finOverlaps);

    float longest = 0;
    for (int i = 0; i < devices_count; i++) {
        for (int j = 0; j < devices_count; j++) {
            cudaEventElapsedTime(&milliseconds, starts[i], stops[j]);
            longest = milliseconds > longest ? milliseconds : longest;
        }
    }
    printf("\n");
    spdlog::trace("Elapsed time: {:.6f} ms", longest);
    spdlog::trace("Merged overlaps: {:d}", finOverlaps.size());
    printf("\n");
}

/**
 * @brief
 *
 * @param boxes Input boxes
 * @param memory_handler Memory handler
 * @param N Number of boxes
 * @param nbox Number of boxes per thread to process
 * @param finOverlaps Cpu output overlaps of collision pairs
 * @param d_overlaps Gpu output overlaps of collision pairs
 * @param d_count Gpu output number of collision pairs
 * @param threads Number of threads per block
 * @param tidstart Start thread id
 * @param devcount Number of devices
 * @param memlimit Memory limit in GB
 */
void runBroadPhase(
    const Aabb* boxes,
    MemoryHandler* memory_handler,
    int N,
    int nbox,
    std::vector<std::pair<int, int>>& finOverlaps,
    int2*& d_overlaps,
    int*& d_count,
    int& threads,
    int& tidstart,
    int& devcount,
    const int memlimit)
{
    cudaDeviceSynchronize();
    spdlog::trace("Number of boxes: {:d}", N);

    if (!memory_handler->MAX_OVERLAP_CUTOFF)
        memory_handler->MAX_OVERLAP_CUTOFF = N;
    if (memlimit) {
        memory_handler->limitGB = memlimit;
        spdlog::trace("Limit set to {:d}", memory_handler->limitGB);
    }

    int device_init_id = 0;

    int smemSize;
    setup(device_init_id, smemSize, threads, nbox);

    cudaSetDevice(device_init_id);

    // Allocate boxes to GPU
    Aabb* d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb) * N);
    cudaMemcpy(d_boxes, boxes, sizeof(Aabb) * N, cudaMemcpyHostToDevice);

    int grid_dim_1d = (N / threads + 1);
    spdlog::trace("Grid dim (1D): {:d}", grid_dim_1d);
    spdlog::trace("Box size: {:d}", sizeof(Aabb));
    spdlog::trace("Scalar3 size: {:d}", sizeof(Scalar3));
    spdlog::trace("sizeof(queue) size: {:d}", sizeof(Queue));

    Scalar2* d_sm;
    cudaMalloc((void**)&d_sm, sizeof(Scalar2) * N);

    MiniBox* d_mini;
    cudaMalloc((void**)&d_mini, sizeof(MiniBox) * N);

    // mean of all box points (used to find best axis)
    //   Scalar3 *d_mean;
    //   cudaMalloc((void **)&d_mean, sizeof(Scalar3));
    //   cudaMemset(d_mean, 0, sizeof(Scalar3));

    //   // recordLaunch("splitBoxes", grid_dim_1d, threads, smemSize,
    //   splitBoxes,
    //   // d_boxes, d_sm, d_mini, N, d_mean);
    //   recordLaunch("calc_mean", grid_dim_1d, threads, smemSize, calc_mean,
    //   d_boxes,
    //                d_mean, N);

    //   // temporary
    //   Scalar3 mean;
    //   cudaMemcpy(&mean, d_mean, sizeof(Scalar3),
    //   cudaMemcpyDeviceToHost); spdlog::trace("mean: x {:.6f} y {:.6f} z
    //   {:.6f}", mean.x, mean.y, mean.z);

    //   // calculate variance and determine which axis to sort on
    //   Scalar3 *d_var; // 2 vertices per box
    //   cudaMalloc((void **)&d_var, sizeof(Scalar3));
    //   cudaMemset(d_var, 0, sizeof(Scalar3));
    //   // calc_variance(boxes, d_var, N, d_mean);
    //   recordLaunch("calc_variance", grid_dim_1d, threads, smemSize,
    //   calc_variance,
    //                d_boxes, d_var, N, d_mean);
    //   cudaDeviceSynchronize();

    //   Scalar3 var3d;
    //   cudaMemcpy(&var3d, d_var, sizeof(Scalar3),
    //   cudaMemcpyDeviceToHost); float maxVar = max(max(var3d.x, var3d.y),
    //   var3d.z);

    //   spdlog::trace("var: x {:.6f} y {:.6f} z {:.6f}", var3d.x, var3d.y,
    //   var3d.z);

    Dimension axis;
    //   if (maxVar == var3d.x)
    //     axis = x;
    //   else if (maxVar == var3d.y)
    //     axis = y;
    //   else
    //     axis = z;
    //   // hack
    axis = x;

    spdlog::trace("Axis: {:s}", axis == x ? "x" : (axis == y ? "y" : "z"));

    recordLaunch<Aabb*, Scalar2*, MiniBox*, int, Dimension>(
        "splitBoxes", grid_dim_1d, threads, smemSize, splitBoxes, d_boxes, d_sm,
        d_mini, N, axis);

    try {
        thrust::sort_by_key(
            thrust::device, d_sm, d_sm + N, d_mini, sort_aabb_x());
        thrust::sort(thrust::device, d_boxes, d_boxes + N, sort_aabb_x());
    } catch (thrust::system_error& e) {
        spdlog::trace("Thrust error: {:s} ", e.what());
    }
    spdlog::trace("Thrust sort finished");

    gpuErrchk(cudaGetLastError());

    memory_handler->setOverlapSize();
    spdlog::trace("Guess cutoff: {:d}", memory_handler->MAX_OVERLAP_CUTOFF);
    size_t overlaps_size = memory_handler->MAX_OVERLAP_SIZE * sizeof(int2);
    spdlog::info("overlaps_size: {:d}", overlaps_size);
    gpuErrchk(cudaGetLastError());

    int* d_start;

    gpuErrchk(cudaMalloc((void**)&d_start, sizeof(int)));
    gpuErrchk(
        cudaMemcpy(d_start, &tidstart, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMalloc((void**)&d_count, sizeof(int)));
    gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));

    // Device memory_handler to keep track of vars
    MemoryHandler* d_memory_handler;
    gpuErrchk(cudaMalloc((void**)&d_memory_handler, sizeof(MemoryHandler)));
    cudaMemcpy(
        d_memory_handler, memory_handler, sizeof(MemoryHandler),
        cudaMemcpyHostToDevice);

    // int2 * d_overlaps;
    spdlog::trace("Allocating overlaps memory");
    gpuErrchk(cudaMalloc((void**)&d_overlaps, overlaps_size));

    spdlog::trace("Starting two stage_queue");
    spdlog::trace("Starting tid {:d}", tidstart);
    // recordLaunch<Scalar2 *, const MiniBox *, int2 *, int, int *, int *,
    //              MemoryHandler *>("runSTQ", grid_dim_1d, threads, runSTQ,
    //              d_sm,
    //                            d_mini, d_overlaps, N, d_count, d_start,
    //                            d_memory_handler);
    // write recordLaunch for SAPVanilla
    recordLaunch<
        Scalar2*, const MiniBox*, int2*, int, int*, int*, MemoryHandler*>(
        "runSAPVanilla", grid_dim_1d, threads, runSAPVanilla, d_sm, d_mini,
        d_overlaps, N, d_count, d_start, d_memory_handler);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaGetLastError());

    int count;
    gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    spdlog::debug("1st count for device {:d}:  {:d}", device_init_id, count);

    int realcount;
    gpuErrchk(cudaMemcpy(
        &realcount, &(d_memory_handler->realcount), sizeof(int),
        cudaMemcpyDeviceToHost));
    spdlog::trace(
        "Real count for device {:d}:  {:d}", device_init_id, realcount);

    spdlog::debug(
        "realcount: {:d}, overlap_size {:d} -> Batching", realcount,
        memory_handler->MAX_OVERLAP_SIZE);
    while (count > memory_handler->MAX_OVERLAP_SIZE) {
        gpuErrchk(cudaFree(d_overlaps));

        memory_handler->handleBroadPhaseOverflow(count);

        gpuErrchk(cudaMalloc(
            (void**)&d_overlaps,
            sizeof(int2) * (memory_handler->MAX_OVERLAP_SIZE)));

        gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));

        cudaMemcpy(
            d_memory_handler, memory_handler, sizeof(MemoryHandler),
            cudaMemcpyHostToDevice);

        // recordLaunch<Scalar2 *, const MiniBox *, int2 *, int, int *, int *,
        //              MemoryHandler *>("runSTQ", grid_dim_1d, threads, runSTQ,
        //              d_sm,
        //                            d_mini, d_overlaps, N, d_count, d_start,
        //                            d_memory_handler);
        recordLaunch<
            Scalar2*, const MiniBox*, int2*, int, int*, int*, MemoryHandler*>(
            "runSAPVanilla", grid_dim_1d, threads, runSAPVanilla, d_sm, d_mini,
            d_overlaps, N, d_count, d_start, d_memory_handler);

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(
            cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(
            &realcount, &(d_memory_handler->realcount), sizeof(int),
            cudaMemcpyDeviceToHost));
        spdlog::trace("Real count for loop:  {:d}", realcount);
        spdlog::trace("Count for loop:  {:d}", count);
        spdlog::debug(
            "Count {:d}, max size {:d}", realcount,
            memory_handler->MAX_OVERLAP_SIZE);
    }
    tidstart += memory_handler->MAX_OVERLAP_CUTOFF;

    gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(
        d_count, &(d_memory_handler->realcount), sizeof(int),
        cudaMemcpyDeviceToDevice));

    cudaFree(d_boxes);
    cudaFree(d_mini);
    cudaFree(d_sm);
    cudaFree(d_start);
    cudaFree(d_memory_handler);

#ifdef SCALABLE_CCD_KEEP_CPU_OVERLAPS
    int2* overlaps = (int2*)malloc(sizeof(int2) * count);
    gpuErrchk(cudaMemcpy(
        overlaps, d_overlaps, sizeof(int2) * (count), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaGetLastError());

    spdlog::trace("Final count for device {:d}:  {:d}", 0, count);

    finOverlaps.reserve(finOverlaps.size() + count);
    for (int i = 0; i < count; i++) {
        assert(overlaps[i].x < overlaps[i].y);
        finOverlaps.emplace_back(overlaps[i].x, overlaps[i].y);
    }

    free(overlaps);

    spdlog::trace(
        "Total(filt.) overlaps for devid {:d}: {:d}", 0, finOverlaps.size());
#endif
    spdlog::trace("Next threadstart {:d}", tidstart);
}
} // namespace scalable_ccd::cuda::stq