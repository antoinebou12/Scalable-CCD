#pragma once

#include <scalable_ccd/cuda/memory_handler.cuh>

#include <thrust/device_vector.h>

#include <array>
#include <vector>

namespace scalable_ccd::cuda {

__global__ void initialize_memory_pool(MP_unit* units, int query_size);

__global__ void compute_vf_tolerance_memory_pool(
    CCDData* data, CCDConfig* config, const int query_size);

__global__ void compute_ee_tolerance_memory_pool(
    CCDData* data, CCDConfig* config, const int query_size);

__global__ void shift_queue_pointers(CCDConfig* config);

__global__ void vf_ccd_memory_pool(
    MP_unit* units, int query_size, CCDData* data, CCDConfig* config);

__global__ void ee_ccd_memory_pool(
    MP_unit* units, int query_size, CCDData* data, CCDConfig* config);

__global__ void compute_ee_tolerance_memory_pool(
    CCDData* data, CCDConfig* config, const int query_size);

bool run_memory_pool_ccd(
    thrust::device_vector<CCDData>& d_data_list,
    std::shared_ptr<MemoryHandler> memory_handler,
    const bool is_edge,
    std::vector<int>& result_list,
    const int parallel_nbr,
    const int max_iter,
    const Scalar tol,
    const bool use_ms,
    const bool allow_zero_toi,
    Scalar& toi);

// get the filter of ccd. the inputs are the vertices of the bounding box of
// the simulation scene this function is directly copied from
// https://github.com/Continuous-Collision-Detection/Tight-Inclusion/
std::array<Scalar, 3> get_numerical_error(
    const std::vector<std::array<Scalar, 3>>& vertices,
    const bool& check_vf,
    const bool using_minimum_separation);

} // namespace scalable_ccd::cuda