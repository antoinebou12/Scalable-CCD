#include "ipc_ccd_strategy.hpp"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/stq/broadphase.cuh>
#include <scalable_ccd/cuda/tight_inclusion/helper.cuh>
#include <scalable_ccd/cuda/utils/device_matrix.cuh>

namespace scalable_ccd::cuda {

using namespace stq;

double ipc_ccd_strategy(
    const Eigen::MatrixXd& _V0,
    const Eigen::MatrixXd& _V1,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const int max_iter,
    const double min_distance,
    const double tolerance)
{
    assert(_V0.rows() == _V1.rows() && _V0.cols() == _V1.cols());

    constexpr int npthreads = 1024;

    // --- Copy vertices to device --------------------------------------------
    logger().trace("Copying vertices");

#ifdef SCALABLE_CCD_USE_DOUBLE
    const Eigen::MatrixXd& V0 = _V0;
    const Eigen::MatrixXd& V1 = _V1;
#else
    const Eigen::MatrixXf V0 = _V0.cast<float>();
    const Eigen::MatrixXf V1 = _V1.cast<float>();
#endif

    const DeviceMatrix<Scalar> d_vertices_t0(V0), d_vertices_t1(V1);

    // --- Construct boxe -----------------------------------------------------
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
        std::vector<int> _result_list; // unused
        run_narrow_phase(
            d_vertices_t0, d_vertices_t1, broad_phase.boxes(), d_overlaps,
            npthreads, /*max_iter=*/max_iter, /*tol=*/tolerance,
            /*ms=*/min_distance, /*allow_zero_toi=*/true, memory_handler,
            _result_list, earliest_toi);

        if (earliest_toi < 1e-6) {
            logger().debug("Running narrow phase again (earliest_toi={:g})");
            run_narrow_phase(
                d_vertices_t0, d_vertices_t1, broad_phase.boxes(), d_overlaps,
                npthreads, /*max_iter=*/-1, /*tol=*/tolerance, /*ms=*/0.0,
                /*allow_zero_toi=*/false, memory_handler, _result_list,
                earliest_toi);
            earliest_toi *= 0.8;
        }

        gpuErrchk(cudaDeviceSynchronize());
    }

    return earliest_toi;
}

} // namespace scalable_ccd::cuda