#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>

#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

namespace scalable_ccd::cuda {

AABB::AABB(const Scalar3& _min, const Scalar3& _max) : min(_min), max(_max)
{
    assert(min.x <= max.x && min.y <= max.y && min.z <= max.z);
}

AABB AABB::from_point(const Scalar3& p, const double inflation_radius)
{
    Scalar3 min = p, max = p;
    conservative_inflation(min, max, inflation_radius);
    return AABB(min, max);
}

void AABB::conservative_inflation(
    Scalar3& min, Scalar3& max, const double inflation_radius)
{
    min.x = nextafter_down(min.x) - nextafter_up(inflation_radius);
    min.y = nextafter_down(min.y) - nextafter_up(inflation_radius);
    min.z = nextafter_down(min.z) - nextafter_up(inflation_radius);
    max.x = nextafter_up(max.x) + nextafter_up(inflation_radius);
    max.y = nextafter_up(max.y) + nextafter_up(inflation_radius);
    max.z = nextafter_up(max.z) + nextafter_up(inflation_radius);
}

// ----------------------------------------------------------------------------

bool AABB::is_vertex(const int3& vids) { return vids.z < 0 && vids.y < 0; }
bool AABB::is_edge(const int3& vids) { return vids.z < 0 && vids.y >= 0; }
bool AABB::is_face(const int3& vids) { return vids.z >= 0; }

bool AABB::is_vertex() const { return AABB::is_vertex(vertex_ids); }
bool AABB::is_edge() const { return AABB::is_edge(vertex_ids); }
bool AABB::is_face() const { return AABB::is_face(vertex_ids); }

bool AABB::is_valid_pair(const int3& a, const int3& b)
{
    return (is_vertex(a) && is_face(b)) || (is_face(a) && is_vertex(b))
        || (is_edge(a) && is_edge(b));
}

bool AABB::is_valid_pair(const AABB& a, const AABB& b)
{
    return AABB::is_valid_pair(a.vertex_ids, b.vertex_ids);
}

// ----------------------------------------------------------------------------
namespace {

    void build_vertex_boxes(
        const Eigen::MatrixXd& _vertices_t0,
        const Eigen::MatrixXd& _vertices_t1,
        std::vector<AABB>& boxes,
        const double inflation_radius)
    {
        assert(_vertices_t0.rows() == _vertices_t1.rows());
        assert(_vertices_t0.cols() == _vertices_t1.cols());
        assert(_vertices_t0.cols() == 3);
        assert(boxes.size() >= _vertices_t0.rows());

#ifdef SCALABLE_CCD_USE_DOUBLE
        const Eigen::MatrixXd& vertices_t0 = _vertices_t0;
        const Eigen::MatrixXd& vertices_t1 = _vertices_t1;
#else
        const Eigen::MatrixXf vertices_t0 = _vertices_t0.cast<float>();
        const Eigen::MatrixXf vertices_t1 = _vertices_t1.cast<float>();
#endif

        tbb::parallel_for(
            tbb::blocked_range<long>(0, vertices_t0.rows()),
            [&](const tbb::blocked_range<long>& r) {
                for (long i = r.begin(); i < r.end(); i++) {
                    boxes[i] = AABB::from_point(
                        make_Scalar3(
                            vertices_t0(i, 0), vertices_t0(i, 1),
                            vertices_t0(i, 2)),
                        make_Scalar3(
                            vertices_t1(i, 0), vertices_t1(i, 1),
                            vertices_t1(i, 2)),
                        inflation_radius);
                    boxes[i].vertex_ids = make_int3(i, -i - 1, -i - 1);
                    boxes[i].box_id = i;
                    boxes[i].element_id = i;
                }
            });
    }

    void build_edge_boxes(
        const Eigen::MatrixXi& edges,
        const size_t edge_offset,
        std::vector<AABB>& boxes)
    {
        assert(boxes.size() >= edges.rows() + edge_offset);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, edges.rows()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    boxes[edge_offset + i] =
                        AABB(boxes[edges(i, 0)], boxes[edges(i, 1)]);
                    boxes[edge_offset + i].vertex_ids =
                        make_int3(edges(i, 0), edges(i, 1), -edges(i, 0) - 1);
                    boxes[edge_offset + i].box_id = edge_offset + i;
                    boxes[edge_offset + i].element_id = i;
                }
            });
    }

    void build_face_boxes(
        const Eigen::MatrixXi& faces,
        const size_t face_offset,
        std::vector<AABB>& boxes)
    {
        assert(boxes.size() >= faces.rows() + face_offset);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, faces.rows()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    boxes[face_offset + i] = AABB(
                        boxes[faces(i, 0)], boxes[faces(i, 1)],
                        boxes[faces(i, 2)]);
                    boxes[face_offset + i].vertex_ids =
                        make_int3(faces(i, 0), faces(i, 1), faces(i, 2));
                    boxes[face_offset + i].box_id = face_offset + i;
                    boxes[face_offset + i].element_id = i;
                }
            });
    }
} // namespace

void constructBoxes(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    std::vector<AABB>& boxes,
    Scalar inflation_radius)
{
    boxes.resize(vertices_t0.rows() + edges.rows() + faces.rows());
    build_vertex_boxes(vertices_t0, vertices_t1, boxes, inflation_radius);
    build_edge_boxes(edges, vertices_t0.rows(), boxes);
    build_face_boxes(faces, vertices_t0.rows() + edges.rows(), boxes);
}

} // namespace scalable_ccd::cuda