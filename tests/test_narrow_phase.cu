#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_approx.hpp>

#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/ccd.cuh>
#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>

#include <filesystem>
namespace fs = std::filesystem;

TEST_CASE("Test CUDA narrow phase", "[gpu][cuda][narrow_phase]")
{
    using namespace scalable_ccd;
    using namespace scalable_ccd::cuda;

    const fs::path data(SCALABLE_CCD_DATA_DIR);

    const fs::path file_t0 =
        data / "cloth-ball" / "frames" / "cloth_ball92.ply";
    const fs::path file_t1 =
        data / "cloth-ball" / "frames" / "cloth_ball93.ply";

    const fs::path vf_ground_truth =
        data / "cloth-ball" / "boxes" / "92vf.json";
    const fs::path ee_ground_truth =
        data / "cloth-ball" / "boxes" / "92ee.json";

    // ------------------------------------------------------------------------
    // Load meshes

    Eigen::MatrixXd vertices_t0, vertices_t1;
    Eigen::MatrixXi edges, faces;
    parse_mesh(file_t0, file_t1, vertices_t0, vertices_t1, faces, edges);

    constexpr bool allow_zero_toi = true;
    constexpr Scalar min_distance = 0;
    constexpr int max_iterations = -1;
    constexpr Scalar tolerance = 1e-6;
    constexpr Scalar memory_limit_GB = 0;

#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<int> result_list;
#endif

    Scalar toi =
        ccd(vertices_t0, vertices_t1, edges, faces, min_distance,
            max_iterations, tolerance, allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            result_list,
#endif
            memory_limit_GB);

    CHECK(toi == Catch::Approx(3.814697265625e-06));

#ifdef SCALABLE_CCD_WITH_PROFILER
    profiler().data()["memory_limit_GB"] = memory_limit_GB;
    profiler().data()["toi"] = toi;
    profiler().print();
#endif
}