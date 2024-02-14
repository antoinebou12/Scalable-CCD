#include "io.hpp"

#include <catch2/catch_test_macros.hpp>

#include <igl/edges.h>
#include <igl/read_triangle_mesh.h>

namespace scalable_ccd {

void parse_mesh(
    const std::filesystem::path& file_t0,
    const std::filesystem::path& file_t1,
    Eigen::MatrixXd& V0,
    Eigen::MatrixXd& V1,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& E)
{
    CAPTURE(file_t0, file_t1);
    REQUIRE(igl::read_triangle_mesh(file_t0.string(), V0, F));
    REQUIRE(igl::read_triangle_mesh(file_t1.string(), V1, F));
    igl::edges(F, E);
}

void parse_mesh(
    const std::filesystem::path& file_t0,
    const std::filesystem::path& file_t1,
    std::vector<AABB>& vertex_boxes,
    std::vector<AABB>& edge_boxes,
    std::vector<AABB>& face_boxes)
{
    Eigen::MatrixXd V0, V1;
    Eigen::MatrixXi F;
    Eigen::MatrixXi E;
    parse_mesh(file_t0, file_t1, V0, V1, F, E);
    build_vertex_boxes(V0, V1, vertex_boxes);
    build_edge_boxes(vertex_boxes, E, edge_boxes);
    build_face_boxes(vertex_boxes, F, face_boxes);
}

} // namespace scalable_ccd