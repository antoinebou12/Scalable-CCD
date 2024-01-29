#include "aabb.hpp"

#include <scalable_ccd/config.hpp>

#include <tbb/parallel_for.h>

namespace scalable_ccd {

#ifndef SCALABLE_CCD_WITH_DOUBLE
namespace {
    float nextafter_up(float x)
    {
        return nextafterf(x, x + std::numeric_limits<float>::max());
    }

    float nextafter_down(float x)
    {
        return nextafterf(x, x - std::numeric_limits<float>::max());
    }
} // namespace
#endif

void build_vertex_boxes(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    std::vector<AABB>& vertex_boxes,
    const double inflation_radius)
{
    vertex_boxes.resize(vertices_t0.rows());

    tbb::parallel_for(
        tbb::blocked_range<long>(0, vertices_t0.rows()),
        [&](const tbb::blocked_range<long>& r) {
            for (long i = r.begin(); i < r.end(); i++) {
                vertex_boxes[i].id = i;
                vertex_boxes[i].vertex_ids = { { i, -i - 1, -i - 1 } };

                const ArrayMax3 vertex_t0 = vertices_t0.row(i).cast<Scalar>();
                const ArrayMax3 vertex_t1 = vertices_t1.row(i).cast<Scalar>();
#ifdef SCALABLE_CCD_WITH_DOUBLE
                vertex_boxes[i].min =
                    vertex_t0.min(vertex_t1) - inflation_radius;
                vertex_boxes[i].max =
                    vertex_t0.max(vertex_t1) + inflation_radius;
#else
                vertex_boxes[i].min =
                    vertex_t0.min(vertex_t1).unaryExpr(&nextafter_down)
                    - nextafter_up(inflation_radius);
                vertex_boxes[i].max =
                    vertex_t0.max(vertex_t1).unaryExpr(&nextafter_up)
                    + nextafter_up(inflation_radius);
#endif
            }
        });
}

void build_edge_boxes(
    const std::vector<AABB>& vertex_boxes,
    const Eigen::MatrixXi& edges,
    std::vector<AABB>& edge_boxes)
{
    edge_boxes.resize(edges.rows());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, edges.rows()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                edge_boxes[i].id = i;
                edge_boxes[i].vertex_ids = { { edges(i, 0), edges(i, 1),
                                               -edges(i, 0) - 1 } };

                const AABB& v0_box = vertex_boxes[edges(i, 0)];
                const AABB& v1_box = vertex_boxes[edges(i, 1)];

                edge_boxes[i].min = v0_box.min.min(v1_box.min);
                edge_boxes[i].max = v0_box.max.max(v1_box.max);
            }
        });
}

void build_face_boxes(
    const std::vector<AABB>& vertex_boxes,
    const Eigen::MatrixXi& faces,
    std::vector<AABB>& face_boxes)
{
    face_boxes.resize(faces.rows());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, faces.rows()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                face_boxes[i].id = i;
                face_boxes[i].vertex_ids = { { faces(i, 0), faces(i, 1),
                                               faces(i, 2) } };

                const AABB& v0_box = vertex_boxes[faces(i, 0)];
                const AABB& v1_box = vertex_boxes[faces(i, 1)];
                const AABB& v2_box = vertex_boxes[faces(i, 2)];

                face_boxes[i].min = v0_box.min.min(v1_box.min).min(v2_box.min);
                face_boxes[i].max = v0_box.max.max(v1_box.max).max(v2_box.max);
            }
        });
}

} // namespace scalable_ccd
