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

    Interval tuv[3];
    int query_id;
};

// this is to calculate the vertices of the inclusion function
struct BoxPrimatives {
    bool b[3];

    int dim;

    Scalar t;
    Scalar u;
    Scalar v;

    __device__ void calculate_tuv(const CCDDomain& domain)
    {
        t = b[0] ? domain.tuv[0].upper : domain.tuv[0].lower;
        u = b[1] ? domain.tuv[1].upper : domain.tuv[1].lower;
        v = b[2] ? domain.tuv[2].upper : domain.tuv[2].lower;
    }
};

} // namespace scalable_ccd::cuda