#include "broadphase.cuh"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/utils/merge_local_overlaps.hpp>
#include <scalable_ccd/cuda/stq/sweep.cuh>
#include <scalable_ccd/cuda/stq/util.cuh>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <tbb/parallel_for.h>

namespace scalable_ccd::cuda::stq {

void runBroadPhaseMultiGPU(
    const AABB* boxes,
    int num_boxes,
    int num_boxes_per_thread,
    std::vector<std::pair<int, int>>& finOverlaps,
    int& threads_per_block,
    int& devcount)
{
    logger().critical(
        "default threads_per_block {}", tbb::info::default_concurrency());
    ThreadSpecificOverlaps storages;

    float milliseconds = 0;
    int device_init_id = 0;

    int smemSize;
    setup(device_init_id, smemSize, threads_per_block, num_boxes_per_thread);

    cudaSetDevice(device_init_id);

    finOverlaps.clear();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate boxes to GPU
    AABB* d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(AABB) * num_boxes);
    cudaMemcpy(
        d_boxes, boxes, sizeof(AABB) * num_boxes, cudaMemcpyHostToDevice);

    dim3 block(threads_per_block);
    int grid_dim_1d = (num_boxes / threads_per_block + 1);
    dim3 grid(grid_dim_1d);
    logger().trace("Grid dim (1D): {:d}", grid_dim_1d);
    logger().trace("Box size: {:d}", sizeof(AABB));

    // Thrust sort (can be improved by sort_by_key)
    cudaEventRecord(start);
    try {
        thrust::sort(
            thrust::device, d_boxes, d_boxes + num_boxes, sort_aabb_x());
    } catch (thrust::system_error& e) {
        logger().trace("Error: {:s} ", e.what());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    logger().trace("Elapsed time for sort: {:.6f} ms", milliseconds);
    cudaDeviceSynchronize();

    int devices_count;
    cudaGetDeviceCount(&devices_count);
    devices_count = devcount ? devcount : devices_count;
    int range = ceil((float)num_boxes / devices_count);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEvent_t starts[devices_count] = {};
    cudaEvent_t stops[devices_count] = {};
    float millisecondss[devices_count] = {};

    tbb::parallel_for(0, devices_count, 1, [&](int& device_id) {
        auto& local_overlaps = storages.local();

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        logger().trace(
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
            logger().trace(
                "Device {:d} cant access Device {:d}", device_id,
                device_init_id);

        int range_start = range * device_id;
        int range_end = range * (device_id + 1);
        logger().trace(
            "device_id: {:d} [{:d}, {:d})", device_id, range_start, range_end);

        AABB* d_b;
        cudaMalloc((void**)&d_b, sizeof(AABB) * num_boxes);
        cudaMemcpy(d_b, d_boxes, sizeof(AABB) * num_boxes, cudaMemcpyDefault);
        cudaDeviceSynchronize();

        cudaDeviceCanAccessPeer(&is_able, device_id, device_init_id);
        cudaDeviceSynchronize();
        if (is_able) {
            cudaDeviceDisablePeerAccess(device_init_id);
            cudaDeviceSynchronize();
        } else if (device_init_id != device_id)
            logger().trace(
                "Device {:d} cant access Device {:d}", device_id,
                device_init_id);

        // Allocate counter to GPU + set to 0 collisions
        int* d_count;
        gpuErrchk(cudaMalloc((void**)&d_count, sizeof(int)));
        gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));
        gpuErrchk(cudaGetLastError());

        // Find overlapping pairs
        int guess = num_boxes * 200;
        logger().trace("Guess {:d}", guess);

        int2* d_overlaps;
        cudaMalloc((void**)&d_overlaps, sizeof(int2) * (guess));
        gpuErrchk(cudaGetLastError());

        int grid_dim_1d = (range / threads_per_block + 1);
        dim3 grid(grid_dim_1d);

        int count;
        cudaEventRecord(starts[device_id]);
        retrieve_collision_pairs<<<grid, block, smemSize>>>(
            d_b, d_count, d_overlaps, num_boxes, guess, num_boxes_per_thread,
            range_start, range_end);
        cudaEventRecord(stops[device_id]);
        cudaEventSynchronize(stops[device_id]);
        cudaEventElapsedTime(
            &millisecondss[device_id], starts[device_id], stops[device_id]);
        cudaDeviceSynchronize();
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        logger().trace("count for device {:d} : {:d}", device_id, count);

        if (count > guess) {
            logger().trace("Running again");
            cudaFree(d_overlaps);
            cudaMalloc((void**)&d_overlaps, sizeof(int2) * (count));
            // cudaMemset(d_overlaps, 0, sizeof(int2)*(count));
            cudaMemset(d_count, 0, sizeof(int));
            cudaEventRecord(starts[device_id]);
            retrieve_collision_pairs<<<grid, block, smemSize>>>(
                d_b, d_count, d_overlaps, num_boxes, count,
                num_boxes_per_thread, range_start, range_end);
            cudaEventRecord(stops[device_id]);
            cudaEventSynchronize(stops[device_id]);
            cudaEventElapsedTime(
                &millisecondss[device_id], starts[device_id], stops[device_id]);
            cudaDeviceSynchronize();
            cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
            logger().trace("count2 for device {:d} : {:d}", device_id, count);
        }
        int2* overlaps = (int2*)malloc(sizeof(int2) * count);
        gpuErrchk(cudaMemcpy(
            overlaps, d_overlaps, sizeof(int2) * (count),
            cudaMemcpyDeviceToHost));
        gpuErrchk(cudaGetLastError());

        logger().trace("Final count for device {:d}:  {:d}", device_id, count);

        for (size_t i = 0; i < count; i++) {
            // local_overlaps.emplace_back(overlaps[i].x, overlaps[i].y);
            // finOverlaps.push_back();
            int aid = overlaps[i].x;
            int bid = overlaps[i].y;
            AABB a = boxes[aid];
            AABB b = boxes[bid];

            if (is_vertex(a) && is_face(b)) // vertex, face
                local_overlaps.emplace_back(aid, bid);
            else if (is_edge(a) && is_edge(b))
                local_overlaps.emplace_back(min(aid, bid), max(aid, bid));
            else if (is_face(a) && is_vertex(b))
                local_overlaps.emplace_back(bid, aid);
        }

        logger().trace(
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
    logger().trace("Elapsed time: {:.6f} ms", longest);
    logger().trace("Merged overlaps: {:d}", finOverlaps.size());
    printf("\n");
}

} // namespace scalable_ccd::cuda::stq