#pragma once

#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/collision.cuh>
#include <scalable_ccd/cuda/broad_phase/utils.cuh>
#include <scalable_ccd/cuda/utils/device_buffer.cuh>

namespace scalable_ccd::cuda {

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
__global__ void sweep_and_tiniest_queue(
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
__global__ void
sweep_and_prune(const Scalar2* const sortedMajorAxis,
    const MiniBox* const boxVerts,
    const int num_boxes,
    const int start_box_id,
    RawDeviceBuffer<int2> overlaps,
    MemoryHandler* memory_handler);

} // namespace scalable_ccd::cuda