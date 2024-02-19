#include "aabb.cuh"

#include <scalable_ccd/cuda/broad_phase/utils.cuh>
#include <scalable_ccd/cuda/utils/assert.cuh>
#include <scalable_ccd/utils/profiler.hpp>

#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

#include <thrust/sort.h>

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

namespace {
    __global__ void split_boxes(
        const AABB* const boxes,
        Scalar2* sortedmin,
        MiniBox* mini,
        const int num_boxes,
        const Dimension axis)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid >= num_boxes)
            return;

        switch (axis) {
        case x:
            sortedmin[tid] = make_Scalar2(boxes[tid].min.x, boxes[tid].max.x);
            mini[tid].min = make_Scalar2(boxes[tid].min.y, boxes[tid].min.z);
            mini[tid].max = make_Scalar2(boxes[tid].max.y, boxes[tid].max.z);
            break;
        case y:
            sortedmin[tid] = make_Scalar2(boxes[tid].min.y, boxes[tid].max.y);
            mini[tid].min = make_Scalar2(boxes[tid].min.x, boxes[tid].min.z);
            mini[tid].max = make_Scalar2(boxes[tid].max.x, boxes[tid].max.z);
            break;
        case z:
            sortedmin[tid] = make_Scalar2(boxes[tid].min.z, boxes[tid].max.z);
            mini[tid].min = make_Scalar2(boxes[tid].min.x, boxes[tid].min.y);
            mini[tid].max = make_Scalar2(boxes[tid].max.x, boxes[tid].max.y);
            break;
        }

        mini[tid].vertex_ids = boxes[tid].vertex_ids;
        mini[tid].element_id = boxes[tid].element_id;
    }
} // namespace

DeviceAABBs::DeviceAABBs(const std::vector<AABB>& boxes)
{
    SCALABLE_CCD_GPU_PROFILE_POINT("DeviceAABBs::DeviceAABBs");

    thrust::device_vector<AABB> d_boxes_tmp;
    {
        SCALABLE_CCD_GPU_PROFILE_POINT("copy_boxes_to_device");
        d_boxes_tmp = boxes; // copy to device
    }

    // const Dimension axis = calc_sort_dimension();
    const Dimension axis = x;

    // Initialize d_sm and d_mini
    {
        SCALABLE_CCD_GPU_PROFILE_POINT("split_boxes");

        sorted_major_intervals.resize(boxes.size());
        mini_boxes.resize(boxes.size());

        constexpr int threads_per_block = 32;
        split_boxes<<<
            boxes.size() / threads_per_block + 1, threads_per_block>>>(
            thrust::raw_pointer_cast(d_boxes_tmp.data()),
            thrust::raw_pointer_cast(sorted_major_intervals.data()),
            thrust::raw_pointer_cast(mini_boxes.data()), boxes.size(), axis);

        gpuErrchk(cudaDeviceSynchronize());
    }

    {
        SCALABLE_CCD_GPU_PROFILE_POINT("sorting_boxes");
        thrust::sort_by_key(
            thrust::device, sorted_major_intervals.begin(),
            sorted_major_intervals.end(), mini_boxes.begin(), SortIntervals());
    }
}

// ----------------------------------------------------------------------------

void build_vertex_boxes(
    const Eigen::MatrixXd& _vertices,
    std::vector<AABB>& vertex_boxes,
    const double inflation_radius)
{
    SCALABLE_CCD_CPU_PROFILE_POINT("build_vertex_boxes");

    assert(_vertices.cols() == 3);

#ifdef SCALABLE_CCD_USE_DOUBLE
    const Eigen::MatrixXd& vertices = _vertices;
#else
    const Eigen::MatrixXf vertices = _vertices.cast<float>();
#endif

    vertex_boxes.resize(vertices.rows());

    tbb::parallel_for(
        tbb::blocked_range<int>(0, vertices.rows()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                vertex_boxes[i] = AABB::from_point(
                    make_Scalar3(
                        vertices(i, 0), vertices(i, 1), vertices(i, 2)),
                    inflation_radius);
                vertex_boxes[i].vertex_ids = make_int3(i, -i - 1, -i - 1);
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
    SCALABLE_CCD_CPU_PROFILE_POINT("build_vertex_boxes");

    assert(_vertices_t0.rows() == _vertices_t1.rows());
    assert(_vertices_t0.cols() == _vertices_t1.cols());
    assert(_vertices_t0.cols() == 3);

#ifdef SCALABLE_CCD_USE_DOUBLE
    const Eigen::MatrixXd& vertices_t0 = _vertices_t0;
    const Eigen::MatrixXd& vertices_t1 = _vertices_t1;
#else
    const Eigen::MatrixXf vertices_t0 = _vertices_t0.cast<float>();
    const Eigen::MatrixXf vertices_t1 = _vertices_t1.cast<float>();
#endif

    vertex_boxes.resize(vertices_t0.rows());

    tbb::parallel_for(
        tbb::blocked_range<int>(0, vertices_t0.rows()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                vertex_boxes[i] = AABB::from_point(
                    make_Scalar3(
                        vertices_t0(i, 0), vertices_t0(i, 1),
                        vertices_t0(i, 2)),
                    make_Scalar3(
                        vertices_t1(i, 0), vertices_t1(i, 1),
                        vertices_t1(i, 2)),
                    inflation_radius);
                vertex_boxes[i].vertex_ids = make_int3(i, -i - 1, -i - 1);
                vertex_boxes[i].element_id = i;
            }
        });
}

void build_edge_boxes(
    const std::vector<AABB>& vertex_boxes,
    const Eigen::MatrixXi& edges,
    std::vector<AABB>& edge_boxes)
{
    SCALABLE_CCD_CPU_PROFILE_POINT("build_edge_boxes");

    edge_boxes.resize(edges.rows());

    tbb::parallel_for(
        tbb::blocked_range<int>(0, edges.rows()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                edge_boxes[i] =
                    AABB(vertex_boxes[edges(i, 0)], vertex_boxes[edges(i, 1)]);
                edge_boxes[i].vertex_ids =
                    make_int3(edges(i, 0), edges(i, 1), -edges(i, 0) - 1);
                edge_boxes[i].element_id = i;
            }
        });
}

void build_face_boxes(
    const std::vector<AABB>& vertex_boxes,
    const Eigen::MatrixXi& faces,
    std::vector<AABB>& face_boxes)
{
    SCALABLE_CCD_CPU_PROFILE_POINT("build_face_boxes");

    face_boxes.resize(faces.rows());

    tbb::parallel_for(
        tbb::blocked_range<int>(0, faces.rows()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                face_boxes[i] = AABB(
                    vertex_boxes[faces(i, 0)], vertex_boxes[faces(i, 1)],
                    vertex_boxes[faces(i, 2)]);
                face_boxes[i].vertex_ids =
                    make_int3(faces(i, 0), faces(i, 1), faces(i, 2));
                face_boxes[i].element_id = i;
            }
        });
}

} // namespace scalable_ccd::cuda