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
    /// @brief The size of the memory allocated for each overlap.
    /// The default value accounts for Broad+Narrow Phase.
    size_t per_overlap_memory_size = sizeof(CCDData) + 3 * sizeof(int2);
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
        return (allocatable - constant_memory_size) / per_overlap_memory_size;
    }

    void setOverlapSize()
    {
        MAX_OVERLAP_SIZE = __getLargestOverlap(__getAllocatable());
    }

    void setMemoryLimitForBroadPhaseOnly()
    {
        per_overlap_memory_size = 3 * sizeof(int2);
    }

    void setMemoryLimitForNarrowAndBroadPhase()
    {
        per_overlap_memory_size = sizeof(CCDData) + 3 * sizeof(int2);
    }

    void handleBroadPhaseOverflow(int desired_count)
    {
        const size_t allocatable = __getAllocatable();
        const size_t largest_overlap_size = __getLargestOverlap(allocatable);

        MAX_OVERLAP_SIZE =
            std::min(largest_overlap_size, static_cast<size_t>(desired_count));

        if (MAX_OVERLAP_SIZE < desired_count) {
            MAX_OVERLAP_CUTOFF >>= 1; // ÷ 2
            if (MAX_OVERLAP_CUTOFF < 1) {
                throw std::runtime_error(
                    "Insufficient memory to increase overlap size; "
                    "cannot allocate even a single box's overlaps.");
            }
            logger().debug(
                "Insufficient memory to increase overlap size; shrinking box cutoff to {:d}.",
                MAX_OVERLAP_CUTOFF);
        } else {
            logger().debug(
                "Setting MAX_OVERLAP_SIZE to {:d} ({:.2f}% of allocatable memory)",
                MAX_OVERLAP_SIZE,
                100 * double(MAX_OVERLAP_SIZE * sizeof(int2)) / allocatable);
        }
    }

    void handleNarrowPhase(size_t& nbr)
    {
        const size_t allocatable = __getAllocatable();

        nbr = std::min(nbr, MAX_QUERIES);

        size_t constraint = sizeof(CCDData) * nbr + sizeof(CCDConfig);
        if (allocatable <= constraint) {
            MAX_QUERIES = (allocatable - sizeof(CCDConfig)) / sizeof(CCDData);
            logger().debug(
                "Insufficient memory for queries (requires {:g} GB); shrinking max queries to {:d}",
                constraint / 1e9, MAX_QUERIES);
            nbr = std::min(nbr, MAX_QUERIES);
            return;
        }

        constraint += sizeof(MP_unit) * nbr;
        if (allocatable <= constraint) {
            MAX_UNIT_SIZE = (allocatable - sizeof(CCDConfig))
                / (sizeof(CCDData) + sizeof(MP_unit));
            logger().warn(
                "Insufficient memory for MP units (requires {:g} GB); shrinking max unit size to {:d}",
                constraint / 1e9, MAX_UNIT_SIZE);
            nbr = std::min(nbr, MAX_UNIT_SIZE);
            return;
        }

        // we are ok if we made it here

        const size_t available_units =
            (allocatable - constraint) / sizeof(MP_unit);
        logger().trace(
            "Can allocate {:d} ({:g} GB) units", available_units,
            available_units * sizeof(MP_unit) / 1e9);

        MAX_UNIT_SIZE = std::min(available_units, 2 * nbr);
        logger().trace("Setting a max unit size to {:d}", MAX_UNIT_SIZE);

        nbr = std::min(nbr, MAX_UNIT_SIZE);

        return;
    }

    void handleOverflow(size_t& nbr)
    {
        const size_t allocatable = __getAllocatable();
        size_t constraint = 2 * sizeof(MP_unit) * MAX_UNIT_SIZE
            + sizeof(CCDData) * nbr + sizeof(CCDConfig);

        if (allocatable > constraint) {
            MAX_UNIT_SIZE <<= 2;
            logger().debug(
                "Overflow: increasing unit_size to {:d}", MAX_UNIT_SIZE);
        } else {
            while (allocatable <= constraint) {
                MAX_QUERIES >>= 2;
                nbr = std::min(nbr, MAX_QUERIES);
                constraint = sizeof(MP_unit) * MAX_UNIT_SIZE
                    + sizeof(CCDData) * nbr + sizeof(CCDConfig);
            }
            logger().debug(
                "Overflow: reducing # of queries to {:d}", MAX_QUERIES);
        }

        nbr = std::min({ nbr, MAX_UNIT_SIZE, MAX_QUERIES });
    }
};

} // namespace scalable_ccd::cuda