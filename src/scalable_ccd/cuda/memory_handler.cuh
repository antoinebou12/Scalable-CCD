#pragma once

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/types.cuh>
#include <scalable_ccd/cuda/utils/assert.cuh>

namespace scalable_ccd::cuda {

__device__ __host__ struct MemoryHandler {
    /// @brief Maximum number of boxes to process in a single kernel launch.
    size_t MAX_OVERLAP_CUTOFF = 0;
    /// @brief Maximum number of overlaps to store in a single kernel launch.
    size_t MAX_OVERLAP_SIZE = 0;
    /// @brief
    size_t MAX_UNIT_SIZE = 0;
    /// @brief
    size_t MAX_QUERIES = 0;
    /// @brief The real number of overlaps found in the last kernel launch.
    int real_count = 0;
    /// @brief
    int limitGB = 0;

    size_t __getAllocatable()
    {
        size_t free;
        size_t total;
        gpuErrchk(cudaMemGetInfo(&free, &total));

        const size_t used = total - free;

        const size_t default_allocatable = static_cast<size_t>(0.95 * free);
        const size_t limit = limitGB << 30; // convert to bytes

        const size_t user_allocatable =
            limit > used ? (limit - used) : default_allocatable;

        logger().trace(
            "Can allocate {:g} of {:g} GB ({:.2f}%) memory",
            user_allocatable / 1e9, total / 1e9,
            100 * static_cast<double>(user_allocatable) / total);

        return std::min(default_allocatable, user_allocatable);
    }

    size_t __getLargestOverlap(const size_t allocatable)
    {
        constexpr size_t constant_memory_size =
            sizeof(MemoryHandler) + 2 * sizeof(int) + sizeof(CCDConfig);
        // extra space for shrink_to_fit
        constexpr size_t per_overlap_memory_size =
            sizeof(CCDData) + 3 * sizeof(int2);
        return (allocatable - constant_memory_size) / per_overlap_memory_size;
    }

    void setOverlapSize()
    {
        MAX_OVERLAP_SIZE = __getLargestOverlap(__getAllocatable());
    }

    void handleBroadPhaseOverflow(int desired_count)
    {
        const size_t allocatable = __getAllocatable();
        const size_t largest_overlap_size = __getLargestOverlap(allocatable);

        MAX_OVERLAP_SIZE =
            std::min(largest_overlap_size, static_cast<size_t>(desired_count));

        if (MAX_OVERLAP_SIZE < desired_count) {
            MAX_OVERLAP_CUTOFF >>= 1; // ÷ 2
            logger().debug(
                "Insufficient memory to increase overlap size; shrinking cutoff by half to {:d}.",
                MAX_OVERLAP_CUTOFF);
        } else {
            logger().debug(
                "Setting MAX_OVERLAP_SIZE to {:d} ({:.2f}% of allocatable memory)",
                MAX_OVERLAP_SIZE,
                100 * double(MAX_OVERLAP_SIZE * sizeof(int2)) / allocatable);
        }
    }

    void handleNarrowPhase(int& nbr)
    {
        size_t allocatable = 0;
        size_t constraint = 0;
        allocatable = __getAllocatable();
        nbr = std::min(static_cast<size_t>(nbr), MAX_QUERIES);
        constraint = sizeof(CCDData) * nbr
            + sizeof(CCDConfig); //+ nbr * sizeof(int2) * 2 +
                                 //+ sizeof(CCDData) * nbr;
        if (allocatable <= constraint) {
            MAX_QUERIES = (allocatable - sizeof(CCDConfig)) / sizeof(CCDData);
            logger().debug(
                "Insufficient memory for queries, shrinking queries to {:d}",
                MAX_QUERIES);
            nbr = std::min(static_cast<size_t>(nbr), MAX_QUERIES);
            return;
        }
        constraint =
            sizeof(MP_unit) * nbr + sizeof(CCDConfig) + sizeof(CCDData) * nbr;

        if (allocatable <= constraint) {
            MAX_UNIT_SIZE = (allocatable - sizeof(CCDConfig))
                / (sizeof(CCDData) + sizeof(MP_unit));
            logger().debug(
                "[MEM INITIAL ISSUE]:  unit size, shrinking unit size to {:d}",
                MAX_UNIT_SIZE);
            nbr = std::min(static_cast<size_t>(nbr), MAX_UNIT_SIZE);
            return;
        }
        // we are ok if we made it here

        size_t available_units = allocatable - constraint;
        available_units /= sizeof(MP_unit);
        // size_t default_units = MAX_UNIT_SIZE ? 2 * MAX_UNIT_SIZE : 2 *
        // size_t default_units = 2 * MAX_QUERIES;
        // if we havent set max_unit_size, set it
        logger().debug(
            "unit options: available {:d} or overlap mulitplier {:d}",
            available_units, MAX_UNIT_SIZE);
        size_t default_units = 2 * nbr;
        MAX_UNIT_SIZE = std::min(available_units, default_units);
        logger().debug("[MEM INITIAL OK]: MAX_UNIT_SIZE={:d}", MAX_UNIT_SIZE);
        nbr = std::min(static_cast<size_t>(nbr), MAX_UNIT_SIZE);
        return;
    }

    void handleOverflow(int& nbr)
    {
        size_t constraint = sizeof(MP_unit) * 2 * MAX_UNIT_SIZE
            + sizeof(CCDConfig) //+ tmp_nbr * sizeof(int2) * 2 +
            + sizeof(CCDData) * nbr;

        size_t allocatable = __getAllocatable();
        if (allocatable > constraint) {
            MAX_UNIT_SIZE *= 2;
            logger().debug(
                "Overflow: Doubling unit_size to {:d}", MAX_UNIT_SIZE);
        } else {
            while (allocatable <= constraint) {
                MAX_QUERIES /= 2;
                nbr = std::min(static_cast<size_t>(nbr), MAX_QUERIES);
                logger().debug(
                    "Overflow: Halving queries to {:d}", MAX_QUERIES);
                constraint = sizeof(MP_unit) * MAX_UNIT_SIZE
                    + sizeof(CCDConfig) //+ tmp_nbr * sizeof(int2) * 2 +
                    + sizeof(CCDData) * nbr;
            }
        }
        nbr = std::min(
            MAX_UNIT_SIZE, std::min(static_cast<size_t>(nbr), MAX_QUERIES));
    }
};

} // namespace scalable_ccd::cuda