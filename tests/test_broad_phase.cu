#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/broad_phase/utils.cuh>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/utils/pca.hpp>
#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>

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

    std::vector<scalable_ccd::cuda::AABB> vertex_boxes, edge_boxes, face_boxes;
    build_vertex_boxes(vertices_t0, vertices_t1, vertex_boxes);
    build_edge_boxes(vertex_boxes, edges, edge_boxes);
    build_face_boxes(vertex_boxes, faces, face_boxes);

    BroadPhase broad_phase;

    broad_phase.build(
        std::make_shared<DeviceAABBs>(vertex_boxes),
        std::make_shared<DeviceAABBs>(face_boxes));
    std::vector<std::pair<int, int>> vf_overlaps =
        broad_phase.detect_overlaps();

    broad_phase.build(std::make_shared<DeviceAABBs>(edge_boxes));
    std::vector<std::pair<int, int>> ee_overlaps =
        broad_phase.detect_overlaps();

    const size_t expected_overlap_size = pca ? 6'954'911 : 6'852'873;
    CHECK(vf_overlaps.size() + ee_overlaps.size() == expected_overlap_size);

    // Offset the boxes to match the way ground truth was originally generated.
    int offset = vertex_boxes.size();
    for (auto& [a, b] : ee_overlaps) {
        a += offset;
        b += offset;
    }
    offset += edge_boxes.size();
    for (auto& [v, f] : vf_overlaps) {
        f += offset;
    }

    compare_mathematica(vf_overlaps, vf_ground_truth);
    compare_mathematica(ee_overlaps, ee_ground_truth);

#ifdef SCALABLE_CCD_WITH_PROFILER
    profiler().print();
#endif
}
