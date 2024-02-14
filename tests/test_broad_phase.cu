#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/broad_phase/utils.cuh>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/utils/pca.hpp>
#include <scalable_ccd/utils/logger.hpp>

#include <igl/write_triangle_mesh.h>

#include <filesystem>
namespace fs = std::filesystem;

TEST_CASE("Test CUDA broad phase", "[gpu][cuda][broad_phase]")
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

    const bool pca = GENERATE(false, true);
    if (pca) {
        nipals_pca(vertices_t0, vertices_t1);
    }

    // ------------------------------------------------------------------------
    // Run

    BroadPhase broad_phase;

    std::vector<scalable_ccd::cuda::AABB> boxes; // output
    broad_phase.build(vertices_t0, vertices_t1, edges, faces, boxes);

    CHECK(boxes.size() == (vertices_t0.rows() + edges.rows() + faces.rows()));
    CHECK(broad_phase.boxes().size() == boxes.size());

    std::vector<std::pair<int, int>> overlaps = broad_phase.detect_overlaps();

    logger().trace("Final CPU overlaps size: {:d}", overlaps.size());
    const size_t expected_overlap_size = pca ? 6'954'911 : 6'852'873;
    CHECK(overlaps.size() == expected_overlap_size);

    // The ground truth is stored as VF, so swap the values if a is F and b is
    // V. This also sorts the EE pairs to be consistent with the ground truth.
    for (auto& [a, b] : overlaps) {
        if (a > b) {
            std::swap(a, b);
        }
    }

    compare_mathematica(overlaps, vf_ground_truth);
    compare_mathematica(overlaps, ee_ground_truth);
}
