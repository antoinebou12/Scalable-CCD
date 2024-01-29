#pragma once

#include <scalable_ccd/scalar.hpp>

#include <array>
#include <vector>

#include <Eigen/Core>

namespace scalable_ccd {

using ArrayMax3 =
    Eigen::Array<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1>;

class AABB {
public:
    AABB() = default;

    AABB(
        const ArrayMax3& _min,
        const ArrayMax3& _max,
        const std::array<long, 3>& _vertex_ids)
        : min(_min)
        , max(_max)
        , vertex_ids(_vertex_ids)
    {
    }

public:
    /// @brief Minimum corner of the AABB.
    ArrayMax3 min;
    /// @brief Maximum corner of the AABB.
    ArrayMax3 max;
    /// @brief Vertex IDs attached to the AABB.
    std::array<long, 3> vertex_ids;
    /// @brief Element ID attached to the AABB.
    long id;
};

/// @brief Build one AABB per vertex position moving linearly from t=0 to t=1.
/// @param vertices_t0 Vertex positions at t=0 (rowwise).
/// @param vertices_t1 Vertex positions at t=1 (rowwise).
/// @param vertex_boxes Vertex AABBs.
/// @param inflation_radius Radius of a capsule around the temporal edges which the AABBs enclose.
void build_vertex_boxes(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    std::vector<AABB>& vertex_boxes,
    double inflation_radius = 0);

/// @brief Build one AABB per edge.
/// @param vertex_boxes Vertex AABBs.
/// @param edges Edges (rowwise).
/// @param edge_boxes Edge AABBs.
void build_edge_boxes(
    const std::vector<AABB>& vertex_boxes,
    const Eigen::MatrixXi& edges,
    std::vector<AABB>& edge_boxes);

/// @brief Build one AABB per face.
/// @param vertex_boxes Vertex AABBs.
/// @param faces Faces (rowwise).
/// @param face_boxes Face AABBs.
void build_face_boxes(
    const std::vector<AABB>& vertex_boxes,
    const Eigen::MatrixXi& faces,
    std::vector<AABB>& face_boxes);

} // namespace scalable_ccd