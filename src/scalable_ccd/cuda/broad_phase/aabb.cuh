#pragma once

#include <scalable_ccd/cuda/scalar.cuh>

#include <Eigen/Core>

#include <vector>
#include <thrust/device_vector.h>

namespace scalable_ccd::cuda {

class AABB {
public:
    AABB() = default;

    AABB(const Scalar3& min, const Scalar3& max);

    AABB(const AABB& a, const AABB& b)
        : AABB(
            make_Scalar3(
                std::min(a.min.x, b.min.x),
                std::min(a.min.y, b.min.y),
                std::min(a.min.z, b.min.z)),
            make_Scalar3(
                std::max(a.max.x, b.max.x),
                std::max(a.max.y, b.max.y),
                std::max(a.max.z, b.max.z)))
    {
    }

    AABB(const AABB& a, const AABB& b, const AABB& c)
        : AABB(
            make_Scalar3(
                std::min({ a.min.x, b.min.x, c.min.x }),
                std::min({ a.min.y, b.min.y, c.min.y }),
                std::min({ a.min.z, b.min.z, c.min.z })),
            make_Scalar3(
                std::max({ a.max.x, b.max.x, c.max.x }),
                std::max({ a.max.y, b.max.y, c.max.y }),
                std::max({ a.max.z, b.max.z, c.max.z })))
    {
    }

    /// @brief Construct an AABB for a static point.
    /// @param p The point's position.
    /// @param inflation_radius Radius of a sphere around the point which the AABB encloses.
    /// @return The constructed AABB.
    static AABB from_point(const Scalar3& p, const double inflation_radius = 0);

    /// @brief Construct an AABB for a moving point (i.e. temporal edge).
    /// @param p_t0 The point's position at time t=0.
    /// @param p_t1 The point's position at time t=1.
    /// @param inflation_radius Radius of a capsule around the temporal edge which the AABB encloses.
    /// @return The constructed AABB.
    static AABB from_point(
        const Scalar3& p_t0,
        const Scalar3& p_t1,
        const double inflation_radius = 0)
    {
        return AABB(
            from_point(p_t0, inflation_radius),
            from_point(p_t1, inflation_radius));
    }

    /// @brief Check if another AABB intersects with this one.
    /// @param other The other AABB.
    /// @return If the two AABBs intersect.
    __device__ bool intersects(const AABB& other) const
    {
        return max.x >= other.min.x && min.x <= other.max.x
            && max.y >= other.min.y && min.y <= other.max.y
            && max.z >= other.min.z && min.z <= other.max.z;
    }

    /// @brief Compute a conservative inflation of the AABB.
    static void conservative_inflation(
        Scalar3& min, Scalar3& max, const double inflation_radius);

    // ------------------------------------------------------------------------

public:
    /// @brief Minimum corner of the AABB.
    Scalar3 min;
    /// @brief Maximum corner of the AABB.
    Scalar3 max;
    /// @brief Vertex IDs attached to the AABB.
    int3 vertex_ids;
    /// @brief Element ID attached to the AABB.
    /// We require this because the boxes will be sorted and we need to know
    /// which element they belong to upstream.
    /// This is in [0, |X| - 1] where X is the set of vertices, edges, or faces.
    int element_id;
};

/// @brief AABB with only y,z coordinates.
struct MiniBox {
    /// @brief Check if another AABB intersects with this one.
    /// @param other The other AABB.
    /// @return If the two AABBs intersect.
    __device__ bool intersects(const MiniBox& other) const
    {
        return max.x >= other.min.x && min.x <= other.max.x
            && max.y >= other.min.y && min.y <= other.max.y;
    }

    /// @brief Minimum corner of the mini-box.
    Scalar2 min;

    /// @brief Maximum corner of the mini-box.
    Scalar2 max;

    /// @brief Vertex IDs attached to the mini-box.
    /// @see AABB::vertex_ids
    int3 vertex_ids;

    /// @brief Box ID attached to the mini-box.
    /// @see AABB::element_id
    int element_id;
};

/// @brief A struct to store the sorted major axis intervals and the sorted mini boxes.
struct DeviceAABBs {
    DeviceAABBs() = default;

    /// @brief Construct the sorted major intervals and mini boxes from a vector of AABBs.
    /// @param boxes Boxes on the host to be copied to the device.
    DeviceAABBs(const std::vector<AABB>& boxes);

    /// @brief Clear the sorted major intervals and mini boxes.
    void clear()
    {
        sorted_major_intervals.clear();
        mini_boxes.clear();
    }

    /// @brief Shrink the sorted major intervals and mini boxes to fit the number of elements.
    void shrink_to_fit()
    {
        sorted_major_intervals.shrink_to_fit();
        mini_boxes.shrink_to_fit();
    }

    /// @brief Get the number of boxes.
    size_t size() const { return sorted_major_intervals.size(); }

    /// @brief Sorted major axis intervals of d_boxes
    thrust::device_vector<Scalar2> sorted_major_intervals;
    /// @brief Sorted min and max values of the non-major axes and vertex information to check for simplex matching and covertices
    thrust::device_vector<MiniBox> mini_boxes;
};

/// @brief Build one AABB per vertex position (row of V).
/// @param[in] vertices Vertex positions (rowwise).
/// @param[out] vertex_boxes Vertex AABBs.
/// @param[in] inflation_radius Radius of a sphere around the points which the AABBs enclose.
void build_vertex_boxes(
    const Eigen::MatrixXd& vertices,
    std::vector<AABB>& vertex_boxes,
    const double inflation_radius = 0);

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

} // namespace scalable_ccd::cuda