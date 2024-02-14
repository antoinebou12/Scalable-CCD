#pragma once

#include <scalable_ccd/cuda/scalar.cuh>

namespace scalable_ccd::cuda {

struct Interval {
    __device__ Interval() = default;

    __device__ Interval(const Scalar& l, const Scalar& u) : lower(l), upper(u)
    {
    }

    Scalar lower;
    Scalar upper;
};

struct SplitInterval {
    __device__ SplitInterval(const Interval& interval)
    {
        const Scalar mid = (interval.lower + interval.upper) / 2;
        first = Interval(interval.lower, mid);
        second = Interval(mid, interval.upper);
    }

    Interval first;
    Interval second;
};

class CCDDomain {
public:
    __device__ void init(int i)
    {
        tuv[0] = Interval(0, 1);
        tuv[1] = Interval(0, 1);
        tuv[2] = Interval(0, 1);
        query_id = i;
    }

    /// @brief The intervals for the t, u, and v parameters
    Interval tuv[3];
    /// @brief The query id
    int query_id;
};

// this is to calculate the vertices of the inclusion function
struct DomainCorner {
    /// @brief Update the t, u, and v parameters based on the corner.
    /// @param domain Domain intervals
    /// @param corner The corner to use (the first bit is for the t parameter, second for the u parameter, and third for the v parameter)
    __device__ void update_tuv(const CCDDomain& domain, const uint8_t corner)
    {
        t = (corner & 1) ? domain.tuv[0].upper : domain.tuv[0].lower;
        u = (corner & 2) ? domain.tuv[1].upper : domain.tuv[1].lower;
        v = (corner & 4) ? domain.tuv[2].upper : domain.tuv[2].lower;
    }

    Scalar t; ///< @brief The t parameter at the corner
    Scalar u; ///< @brief The u parameter at the corner
    Scalar v; ///< @brief The v parameter at the corner
};

} // namespace scalable_ccd::cuda