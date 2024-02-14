#include "narrow_phase.cuh"

#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/narrow_phase/root_finder.cuh>
#include <scalable_ccd/cuda/utils/device_buffer.cuh>
#include <scalable_ccd/utils/profiler.hpp>

#include <thrust/host_vector.h>
#include <thrust/copy.h>

namespace scalable_ccd::cuda {

namespace {

    /// @brief Split the heterogeneous array of overlaps into two array of vertex-face and edge-edge overlaps.
    /// @param[in] boxes The array of AABBs
    /// @param[in] overlaps The array of pairs of indices of the boxes that overlap
    /// @param[in] n_overlaps The number of overlaps
    /// @param[out] vf_overlaps The output array of pairs of indices of the vertex-face overlaps
    /// @param[out] ee_overlaps The output array of pairs of indices of the edge-edge overlaps
    __global__ void split_overlaps(
        const AABB* const boxes,
        const int2* const overlaps,
        const int n_overlaps,
        RawDeviceBuffer<int2> vf_overlaps,
        RawDeviceBuffer<int2> ee_overlaps)
    {
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= n_overlaps)
            return;

        const int a_id = min(overlaps[tid].x, overlaps[tid].y);
        const int b_id = max(overlaps[tid].x, overlaps[tid].y);

        if (boxes[a_id].is_vertex() && boxes[b_id].is_face()) {
            vf_overlaps.push(make_int2(a_id, b_id));
        } else if (boxes[a_id].is_edge() && boxes[b_id].is_edge()) {
            ee_overlaps.push(make_int2(a_id, b_id));
        } else {
            printf(
                "Invalid overlap: %d %d [(%d %d %d); (%d %d %d)]\n", a_id, b_id,
                boxes[a_id].is_vertex(), boxes[a_id].is_edge(),
                boxes[a_id].is_face(), boxes[b_id].is_vertex(),
                boxes[b_id].is_edge(), boxes[b_id].is_face());
            assert(false);
        }
    }

    /// @brief Populate the CCDData array with the necessary data for the narrow phase.
    /// @param V0 Vertex positions at time t=0
    /// @param V1 Vertex positions at time t=1
    /// @param n_vertices The number of vertices
    /// @param boxes The array of AABBs
    /// @param overlaps The array of pairs of indices of the boxes that overlap
    /// @param ms Minimum separation distance
    /// @param data The output array of CCDData
    __global__ void add_data(
        const Scalar* const V0,
        const Scalar* const V1,
        const int n_vertices,
        const AABB* const boxes,
        const RawDeviceBuffer<int2> overlaps,
        const Scalar ms,
        CCDData* data)
    {
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= *overlaps.size)
            return;

        data[tid].ms = ms;

        const int minner = min(overlaps[tid].x, overlaps[tid].y);
        const int maxxer = max(overlaps[tid].x, overlaps[tid].y);
        const int3 avids = boxes[minner].vertex_ids;
        const int3 bvids = boxes[maxxer].vertex_ids;

#ifdef SCALABLE_CCD_TOI_PER_QUERY
        data[tid].toi = INFINITY;
        data[tid].aid = minner;
        data[tid].bid = maxxer;
#endif

        if (AABB::is_vertex(avids) && AABB::is_face(bvids)) {
            for (size_t i = 0; i < 3; i++) {
                data[tid].v0s[i] = V0[avids.x + i * n_vertices];
                data[tid].v1s[i] = V0[bvids.x + i * n_vertices];
                data[tid].v2s[i] = V0[bvids.y + i * n_vertices];
                data[tid].v3s[i] = V0[bvids.z + i * n_vertices];
                data[tid].v0e[i] = V1[avids.x + i * n_vertices];
                data[tid].v1e[i] = V1[bvids.x + i * n_vertices];
                data[tid].v2e[i] = V1[bvids.y + i * n_vertices];
                data[tid].v3e[i] = V1[bvids.z + i * n_vertices];
            }
        } else if (AABB::is_edge(avids) && AABB::is_edge(bvids)) {
            for (size_t i = 0; i < 3; i++) {
                data[tid].v0s[i] = V0[avids.x + i * n_vertices];
                data[tid].v1s[i] = V0[avids.y + i * n_vertices];
                data[tid].v2s[i] = V0[bvids.x + i * n_vertices];
                data[tid].v3s[i] = V0[bvids.y + i * n_vertices];
                data[tid].v0e[i] = V1[avids.x + i * n_vertices];
                data[tid].v1e[i] = V1[avids.y + i * n_vertices];
                data[tid].v2e[i] = V1[bvids.x + i * n_vertices];
                data[tid].v3e[i] = V1[bvids.y + i * n_vertices];
            }
        } else {
            assert(false);
        }
    }

#ifdef SCALABLE_CCD_TOI_PER_QUERY
    struct is_collision {
        __host__ __device__ bool operator()(const CCDData& data)
        {
            return data.toi < 1;
        }
    };

    void copy_out_collisions(
        const thrust::device_vector<CCDData>& d_ccd_data,
        std::vector<std::tuple<int, int, Scalar>>& collisions)
    {
        // Filter only the collisions on the device
        thrust::device_vector<CCDData> d_filtered_ccd_data(d_ccd_data.size());
        auto itr = thrust::copy_if(
            d_ccd_data.begin(), d_ccd_data.end(), d_filtered_ccd_data.begin(),
            is_collision());
        d_filtered_ccd_data.resize(
            thrust::distance(d_filtered_ccd_data.begin(), itr));

        // Copy the filtered collisions to the host
        thrust::host_vector<CCDData> filtered_ccd_data = d_filtered_ccd_data;

        // Transform the filtered collisions to the output format
        for (const auto& data : filtered_ccd_data) {
            collisions.emplace_back(data.aid, data.bid, data.toi);
        }
    }
#endif

} // namespace

void narrow_phase(
    const DeviceMatrix<Scalar>& d_vertices_t0,
    const DeviceMatrix<Scalar>& d_vertices_t1,
    const thrust::device_vector<AABB>& d_boxes,
    const thrust::device_vector<int2>& d_overlaps,
    const int threads,
    const int max_iter,
    const Scalar tol,
    const Scalar ms,
    const bool allow_zero_toi,
    std::shared_ptr<MemoryHandler> memory_handler,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<std::tuple<int, int, Scalar>>& collisions,
#endif
    Scalar& toi)
{
    assert(toi >= 0);

    const bool use_ms = ms > 0;

    size_t start_id = 0;
    size_t size = d_overlaps.size();
    memory_handler->MAX_QUERIES = size;

    size_t remaining_queries;
#ifndef SCALABLE_CCD_TOI_PER_QUERY
    while ((remaining_queries = size - start_id) > 0 && toi > 0) {
#else
    while ((remaining_queries = size - start_id) > 0) {
#endif
        logger().trace("Remaining queries to process: {:d}", remaining_queries);

        bool overflowed = false;
        size_t n_queries_to_process;

        thrust::device_vector<CCDData> d_vf_data_list, d_ee_data_list;
        do {
            n_queries_to_process =
                std::min(remaining_queries, memory_handler->MAX_QUERIES);

            if (!overflowed) { // only true in first iteration
                memory_handler->handleNarrowPhase(n_queries_to_process);
            } else {
                memory_handler->handleOverflow(n_queries_to_process);
            }
            assert(n_queries_to_process > 0);
            assert(n_queries_to_process <= d_overlaps.size());

            {
                // Allocate enough space for the worst case
                DeviceBuffer<int2> d_vf_overlaps(n_queries_to_process);
                DeviceBuffer<int2> d_ee_overlaps(n_queries_to_process);

                {
                    SCALABLE_CCD_GPU_PROFILE_POINT("split_overlaps");

                    split_overlaps<<<
                        n_queries_to_process / threads + 1, threads>>>(
                        thrust::raw_pointer_cast(d_boxes.data()),
                        thrust::raw_pointer_cast(d_overlaps.data()) + start_id,
                        n_queries_to_process, d_vf_overlaps, d_ee_overlaps);

                    gpuErrchk(cudaDeviceSynchronize());
                }

                logger().trace(
                    "# FV queries: {:d}; # EE queries: {:d}",
                    d_vf_overlaps.size(), d_ee_overlaps.size());

                {
                    SCALABLE_CCD_GPU_PROFILE_POINT("create_ccd_data");

                    d_vf_data_list.resize(d_vf_overlaps.size());
                    add_data<<<d_vf_data_list.size() / threads + 1, threads>>>(
                        d_vertices_t0.data(), d_vertices_t1.data(),
                        d_vertices_t0.rows(),
                        thrust::raw_pointer_cast(d_boxes.data()), d_vf_overlaps,
                        ms, thrust::raw_pointer_cast(d_vf_data_list.data()));
                    gpuErrchk(cudaDeviceSynchronize());

                    d_ee_data_list.resize(d_ee_overlaps.size());
                    add_data<<<d_ee_data_list.size() / threads + 1, threads>>>(
                        d_vertices_t0.data(), d_vertices_t1.data(),
                        d_vertices_t0.rows(),
                        thrust::raw_pointer_cast(d_boxes.data()), d_ee_overlaps,
                        ms, thrust::raw_pointer_cast(d_ee_data_list.data()));
                    gpuErrchk(cudaDeviceSynchronize());
                }
            }

            logger().trace(
                "Narrow phase CCD data size: {:g} GB",
                (d_vf_data_list.size() + d_ee_data_list.size())
                    * sizeof(CCDData) / 1e9);

            constexpr int parallel = 64;
            logger().trace(
                "Running memory-pooled CCD using {:d} threads", parallel);
            {
                SCALABLE_CCD_GPU_PROFILE_POINT("FV CCD");

                overflowed = ccd</*is_vf=*/true>(
                    d_vf_data_list, memory_handler, parallel, max_iter, tol,
                    use_ms, allow_zero_toi, toi);

                gpuErrchk(cudaDeviceSynchronize());
            }

            if (overflowed) // rerun
            {
                logger().debug(
                    "Narrow-phase: overflowed upon face-vertex; reducing parallel count");
                continue;
            }

            logger().debug("ToI after FV: {:e}", toi);

            {
                SCALABLE_CCD_GPU_PROFILE_POINT("EE CCD");

                overflowed = ccd</*is_vf=*/false>(
                    d_ee_data_list, memory_handler, parallel, max_iter, tol,
                    use_ms, allow_zero_toi, toi);

                gpuErrchk(cudaDeviceSynchronize());
            }

            if (overflowed) {
                logger().debug(
                    "Narrow-phase: overflowed upon edge-edge; reducing parallel count");
                continue;
            }

            logger().debug("ToI after EE: {:e}", toi);
        } while (overflowed);

#ifdef SCALABLE_CCD_TOI_PER_QUERY
        {
            SCALABLE_CCD_GPU_PROFILE_POINT("copy_out_collisions");
            copy_out_collisions(d_vf_data_list, collisions);
            copy_out_collisions(d_ee_data_list, collisions);
        }
#endif

        start_id += n_queries_to_process;
    }
}

} // namespace scalable_ccd::cuda