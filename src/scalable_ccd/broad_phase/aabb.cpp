#include "aabb.hpp"

#include <scalable_ccd/config.hpp>

#include <tbb/parallel_for.h>

#include <cassert>

namespace scalable_ccd {

AABB::AABB(const ArrayMax3& _min, const ArrayMax3& _max) : min(_min), max(_max)
{
    assert(min.size() == max.size());
    assert((min <= max).all());
}

AABB AABB::from_point(const ArrayMax3& p, const double inflation_radius)
{
    ArrayMax3 min = p, max = p;
    conservative_inflation(min, max, inflation_radius);
    return AABB(min, max);
}

bool AABB::intersects(const AABB& other) const
{
    assert(this->min.size() == other.max.size());
    assert(this->max.size() == other.min.size());
    return (this->min <= other.max).all() && (other.min <= this->max).all();
}

void AABB::conservative_inflation(
    ArrayMax3& min, ArrayMax3& max, const double inflation_radius)
{
    min = min.unaryExpr(&nextafter_down) - nextafter_up(inflation_radius);
    max = max.unaryExpr(&nextafter_up) + nextafter_up(inflation_radius);
}

void build_vertex_boxes(
    const Eigen::MatrixXd& _vertices,
    std::vector<AABB>& vertex_boxes,
    const double inflation_radius)
{
#ifdef SCALABLE_CCD_USE_DOUBLE
    const Eigen::MatrixXd& vertices = _vertices;
#else
    const Eigen::MatrixXf vertices = _vertices.cast<float>();
#endif

    vertex_boxes.resize(vertices.rows());

    tbb::parallel_for(
        tbb::blocked_range<long>(0, vertices.rows()),
        [&](const tbb::blocked_range<long>& r) {
            for (long i = r.begin(); i < r.end(); i++) {
                vertex_boxes[i] =
                    AABB::from_point(vertices.row(i), inflation_radius);
                vertex_boxes[i].vertex_ids = { { i, -i - 1, -i - 1 } };
                vertex_boxes[i].element_id = i;
            }
        });
}

void build_vertex_boxes(
    const Eigen::MatrixXd& _vertices_t0,
    const Eigen::MatrixXd& _vertices_t1,
    std::vector<AABB>& vertex_boxes,
    const double inflation_radius)
{
    assert(_vertices_t0.rows() == _vertices_t1.rows());
    assert(_vertices_t0.cols() == _vertices_t1.cols());

#ifdef SCALABLE_CCD_USE_DOUBLE
    const Eigen::MatrixXd& vertices_t0 = _vertices_t0;
    const Eigen::MatrixXd& vertices_t1 = _vertices_t1;
#else
    const Eigen::MatrixXf vertices_t0 = _vertices_t0.cast<float>();
    const Eigen::MatrixXf vertices_t1 = _vertices_t1.cast<float>();
#endif

    vertex_boxes.resize(vertices_t0.rows());

    tbb::parallel_for(
        tbb::blocked_range<long>(0, vertices_t0.rows()),
        [&](const tbb::blocked_range<long>& r) {
            for (long i = r.begin(); i < r.end(); i++) {
                vertex_boxes[i] = AABB::from_point(
                    vertices_t0.row(i), vertices_t1.row(i), inflation_radius);
                vertex_boxes[i].vertex_ids = { { i, -i - 1, -i - 1 } };
                vertex_boxes[i].element_id = i;
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
                edge_boxes[i] =
                    AABB(vertex_boxes[edges(i, 0)], vertex_boxes[edges(i, 1)]);
                edge_boxes[i].vertex_ids = { { edges(i, 0), edges(i, 1),
                                               -edges(i, 0) - 1 } };
                edge_boxes[i].element_id = i;
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
                face_boxes[i] = AABB(
                    vertex_boxes[faces(i, 0)], vertex_boxes[faces(i, 1)],
                    vertex_boxes[faces(i, 2)]);
                face_boxes[i].vertex_ids = { { faces(i, 0), faces(i, 1),
                                               faces(i, 2) } };
                face_boxes[i].element_id = i;
            }
        });
}

} // namespace scalable_ccd
