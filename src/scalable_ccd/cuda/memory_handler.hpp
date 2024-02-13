#pragma once

#include <scalable_ccd/cuda/narrow_phase/ccd_data.cuh>

namespace scalable_ccd::cuda {

struct MemoryHandler {
    /// @brief Maximum number of boxes to process in a single kernel launch.
    size_t MAX_OVERLAP_CUTOFF = 0;
    /// @brief Maximum number of overlaps to store in a single kernel launch.
    size_t MAX_OVERLAP_SIZE = 0;
    /// @brief
    size_t MAX_UNIT_SIZE = 0;
    /// @brief
    size_t MAX_QUERIES = 0;
    /// @brief The size of the memory allocated for each overlap.
    /// The default value accounts for Broad+Narrow Phase.
    size_t per_overlap_memory_size = sizeof(CCDData) + 3 * sizeof(int2);
    /// @brief The real number of overlaps found in the last kernel launch.
    int real_count = 0;
    /// @brief
    int memory_limit_GB = 0;

    size_t __getAllocatable();

    size_t __getLargestOverlap(const size_t allocatable);

    void setOverlapSize();

    void setMemoryLimitForBroadPhaseOnly();

    void setMemoryLimitForNarrowAndBroadPhase();

    void handleBroadPhaseOverflow(int desired_count);

    void handleNarrowPhase(size_t& nbr);

    void handleOverflow(size_t& nbr);
};

} // namespace scalable_ccd::cuda