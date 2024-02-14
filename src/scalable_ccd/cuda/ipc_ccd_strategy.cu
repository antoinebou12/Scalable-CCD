#include "ipc_ccd_strategy.hpp"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/narrow_phase/narrow_phase.cuh>
#include <scalable_ccd/cuda/utils/device_matrix.cuh>
#include <scalable_ccd/cuda/utils/assert.cuh>

namespace scalable_ccd::cuda {

Scalar ipc_ccd_strategy(
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double min_distance,
    const int max_iter,
    const double tolerance)
{
    assert(V0.rows() == V1.rows() && V0.cols() == V1.cols());
    assert(V0.cols() == 3);

    constexpr int npthreads = 1024;

    // --- Copy vertices to device --------------------------------------------
    logger().trace("Copying vertices");

    const DeviceMatrix<Scalar> d_vertices_t0(V0), d_vertices_t1(V1);

    // --- Construct boxes ----------------------------------------------------
    logger().trace("Constructing boxes");

    std::vector<AABB> boxes;
    constructBoxes(V0, V1, E, F, boxes);

    // --- Copy boxes to device -----------------------------------------------
    logger().trace("Building broad phase");

    std::shared_ptr<MemoryHandler> memory_handler =
        std::make_shared<MemoryHandler>();

    BroadPhase broad_phase(memory_handler);
    broad_phase.build(boxes);

    // --- Run broad + narrow phase -------------------------------------------

    Scalar earliest_toi = 1.0;
    while (!broad_phase.is_complete()) {
        logger().trace("Running broad phase");

        const thrust::device_vector<int2>& d_overlaps =
            broad_phase.detect_overlaps_partial();

        logger().debug("Running narrow phase");
        const Scalar earliest_toi_before = earliest_toi;
#ifdef SCALABLE_CCD_TOI_PER_QUERY
        std::vector<std::tuple<int, int, Scalar>> _collisions;
#endif
        narrow_phase(
            d_vertices_t0, d_vertices_t1, broad_phase.boxes(), d_overlaps,
            npthreads, /*max_iter=*/max_iter, /*tol=*/tolerance,
            /*ms=*/min_distance, /*allow_zero_toi=*/true, memory_handler,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            _collisions,
#endif
            earliest_toi);

        if (earliest_toi < 1e-6) {
            logger().debug(
                "Running narrow phase again (earliest_toi={:g})", earliest_toi);
            earliest_toi = earliest_toi_before;
            narrow_phase(
                d_vertices_t0, d_vertices_t1, broad_phase.boxes(), d_overlaps,
                npthreads, /*max_iter=*/-1, /*tol=*/tolerance, /*ms=*/0.0,
                /*allow_zero_toi=*/false, memory_handler,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
                _collisions,
#endif
                earliest_toi);
            earliest_toi *= 0.8;
        }

        gpuErrchk(cudaDeviceSynchronize());
    }

    logger().debug("Earliest toi: {:g}", earliest_toi);

    return earliest_toi;
}

} // namespace scalable_ccd::cuda