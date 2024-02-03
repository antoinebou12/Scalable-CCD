#pragma once

#include <scalable_ccd/cuda/scalar.cuh>

#include <Eigen/Core>

#include <vector>

namespace scalable_ccd::cuda {

class AABB {
public:
    AABB() = default;

    AABB(const Scalar3& _min, const Scalar3& _max);

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

    /// @brief Compute a conservative inflation of the AABB.
    static void conservative_inflation(
        Scalar3& min, Scalar3& max, const double inflation_radius);

    // ------------------------------------------------------------------------

    __host__ __device__ bool is_vertex() const;
    __host__ __device__ bool is_edge() const;
    __host__ __device__ bool is_face() const;

    __host__ __device__ static bool is_vertex(const int3& vids);
    __host__ __device__ static bool is_edge(const int3& vids);
    __host__ __device__ static bool is_face(const int3& vids);

    __host__ __device__ static bool is_valid_pair(const AABB& x, const AABB& y);
    __host__ __device__ static bool is_valid_pair(const int3& a, const int3& b);

public:
    /// @brief Minimum corner of the AABB.
    Scalar3 min;
    /// @brief Maximum corner of the AABB.
    Scalar3 max;
    /// @brief Vertex IDs attached to the AABB.
    int3 vertex_ids;
    /// @brief Box ID attached to the AABB.
    int box_id;
    /// @brief Element ID attached to the AABB.
    int element_id;
};

struct MiniBox {
    MiniBox() = default;

    __device__
    MiniBox(Scalar* tempmin, Scalar* tempmax, int3 vids, int assignid)
    {
        min = make_Scalar2(tempmin[0], tempmin[1]);
        max = make_Scalar2(tempmax[0], tempmax[1]);
        vertex_ids = vids;
        box_id = assignid;
    }

    Scalar2 min; // only y,z coord
    Scalar2 max;
    int3 vertex_ids;
    int box_id;
};

void constructBoxes(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    std::vector<AABB>& boxes,
    Scalar inflation_radius = 0);

} // namespace scalable_ccd::cuda