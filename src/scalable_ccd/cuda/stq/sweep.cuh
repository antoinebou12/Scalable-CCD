#pragma once

#include <scalable_ccd/cuda/types.cuh>
#include <scalable_ccd/cuda/memory_handler.cuh>
#include <scalable_ccd/cuda/stq/collision.cuh>
#include <scalable_ccd/cuda/utils/device_buffer.cuh>

namespace scalable_ccd::cuda::stq {

/// @brief
/// @param boxes
/// @param count
/// @param overlaps
/// @param num_boxes
/// @param guess
/// @param nbox
/// @param start
/// @param end
/// @return
__global__ void retrieve_collision_pairs(
    const AABB* const boxes,
    int* count,
    int2* overlaps,
    int num_boxes,
    int guess,
    int nbox,
    int start = 0,
    int end = INT_MAX);

/// @brief Calculate the mean of the box centers
/// @param boxes Pointer to the boxes
/// @param num_boxes Number of boxes
/// @param mean Output array for the mean
__global__ void
calc_mean(const AABB* const boxes, const int num_boxes, Scalar3* mean);

/// @brief Calculate the variance of the box centers
/// @param boxes Pointer to the boxes
/// @param num_boxes Number of boxes
/// @param mean Mean of the box centers
/// @param var Output array for the variance
__global__ void calc_variance(
    const AABB* const boxes,
    const int num_boxes,
    const Scalar3* const mean,
    Scalar3* var);

/// @brief Splits each box into 2 different objects to improve performance
/// @param boxes Boxes to be rearranged
/// @param sortedmin Contains the sorted min and max values of the non-major axes
/// @param mini Contains the vertex information of the boxes
/// @param num_boxes Number of boxes
/// @param axis Major axis
__global__ void splitBoxes(
    const AABB* const boxes,
    Scalar2* sortedmin,
    MiniBox* mini,
    const int num_boxes,
    const Dimension axis);

/// @brief Runs the sweep and prune tiniest queue (STQ) algorithm
/// @param sortedMajorAxis Contains the sorted min and max values of the major axis
/// @param boxVerts Contains the sorted min and max values of the non-major axes and vertex information to check for simplex matching and covertices
/// @param num_boxes Number of boxes
/// @param start_box_id Starting box index
/// @param overlaps Final output array of colliding box pairs
/// @param memory_handler Memory handler
__global__ void runSTQ(
    const Scalar2* const sortedMajorAxis,
    const MiniBox* const boxVerts,
    const int num_boxes,
    const int start_box_id,
    RawDeviceBuffer<int2> overlaps,
    MemoryHandler* memory_handler);

/// @brief Runs the sweep and prune (SAP) algorithm
/// @param sortedMajorAxis Contains the sorted min and max values of the major axis
/// @param boxVerts Contains the sorted min and max values of the non-major axes and vertex information to check for simplex matching and covertices
/// @param num_boxes Number of boxes
/// @param start_box_id Starting box index
/// @param overlaps Final output array of colliding box pairs
/// @param memory_handler Memory handler
/// @return
__global__ void runSAP(
    const Scalar2* const sortedMajorAxis,
    const MiniBox* const boxVerts,
    const int num_boxes,
    const int start_box_id,
    RawDeviceBuffer<int2> overlaps,
    MemoryHandler* memory_handler);

} // namespace scalable_ccd::cuda::stq