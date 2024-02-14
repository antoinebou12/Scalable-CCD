#include "ccd.cuh"

#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/narrow_phase/narrow_phase.cuh>
#include <scalable_ccd/cuda/utils/assert.cuh>
#include <scalable_ccd/cuda/utils/device_matrix.cuh>
#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>

namespace scalable_ccd::cuda {

double
ccd(const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    const Scalar minimum_separation_distance,
    const int max_iterations,
    const Scalar tolerance,
    const bool allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<std::tuple<int, int, Scalar>>& collisions,
#endif
    const int memory_limit_GB)
{
    assert(vertices_t0.rows() == vertices_t1.rows());
    assert(vertices_t0.cols() == vertices_t1.cols());

    constexpr int bp_threads = 32;
    constexpr int np_threads = 1024;

    // --- Copy vertices to device --------------------------------------------
    logger().trace("Copying vertices");

    const DeviceMatrix<Scalar> d_vertices_t0(vertices_t0);
    const DeviceMatrix<Scalar> d_vertices_t1(vertices_t1);

    // --- Copy boxes to device -----------------------------------------------
    logger().trace("Building broad phase");

    std::shared_ptr<MemoryHandler> memory_handler =
        std::make_shared<MemoryHandler>();
    if (memory_limit_GB) {
        logger().trace("Setting memory limit to {:d} GB", memory_limit_GB);
        memory_handler->memory_limit_GB = memory_limit_GB;
    }

    BroadPhase broad_phase(memory_handler);
    broad_phase.threads_per_block = bp_threads;

    std::vector<AABB> boxes;
    broad_phase.build(vertices_t0, vertices_t1, edges, faces, boxes);

    // --- Run broad + narrow phase -------------------------------------------

    Scalar toi = 1;

    while (!broad_phase.is_complete()) {
        logger().trace("Running broad phase");
        {
            SCALABLE_CCD_GPU_PROFILE_POINT("Broad Phase");

            broad_phase.detect_overlaps_partial();

            gpuErrchk(cudaDeviceSynchronize());
        }
        const thrust::device_vector<int2>& d_overlaps = broad_phase.overlaps();

        logger().debug("Running narrow phase");
        {
            SCALABLE_CCD_GPU_PROFILE_POINT("Narrow Phase");

            narrow_phase(
                d_vertices_t0, d_vertices_t1, broad_phase.boxes(), d_overlaps,
                np_threads, max_iterations, tolerance,
                minimum_separation_distance, allow_zero_toi, memory_handler,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
                collisions,
#endif
                toi);

            gpuErrchk(cudaDeviceSynchronize());
        }
    }

    return toi;
}

} // namespace scalable_ccd::cuda