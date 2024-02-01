#pragma once

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/scalar.cuh>

#include <cuda/semaphore>

namespace scalable_ccd::cuda {

enum Dimension { x, y, z };

struct Interval {
    __device__ Interval() {};

    __device__ Interval(const Scalar& f, const Scalar& s)
    {
        first = f;
        second = s;
    }

    __device__ Interval& operator=(const Interval& x)
    {
        if (this == &x)
            return *this;
        first = x.first;
        second = x.second;
        return *this;
    }

    Scalar first;
    Scalar second;
};

struct IntervalPair {
    __device__ IntervalPair() = default;
    __device__ IntervalPair(const Interval& itv);

    Interval first;
    Interval second;
};

class MP_unit {
public:
    __device__ __host__ void init(int i)
    {
        itv[0].first = 0;
        itv[0].second = 1;
        itv[1].first = 0;
        itv[1].second = 1;
        itv[2].first = 0;
        itv[2].second = 1;
        query_id = i;
        // box_in = true; // same result if true or false
    }

    __device__ MP_unit& operator=(const MP_unit& x)
    {
        if (this == &x)
            return *this;
        itv[0] = x.itv[0];
        itv[1] = x.itv[1];
        itv[2] = x.itv[2];
        query_id = x.query_id;
        // box_in = x.box_in;
        // true_tol = x.true_tol;
        return *this;
    }

    Interval itv[3];
    int query_id;
    // Scalar true_tol;
    // bool box_in;
};

class CCDData {
public:
    __host__ __device__ CCDData() = default;

    // CCDData(const std::array<std::array<Scalar,3>,8>&input);

    __device__ __host__ CCDData& operator=(const CCDData& x)
    {
        if (this == &x)
            return *this;
        for (int i = 0; i < 3; i++) {
            v0s[i] = x.v0s[i];
            v1s[i] = x.v1s[i];
            v2s[i] = x.v2s[i];
            v3s[i] = x.v3s[i];
            v0e[i] = x.v0e[i];
            v1e[i] = x.v1e[i];
            v2e[i] = x.v2e[i];
            v3e[i] = x.v3e[i];
            err[i] = x.err[i];
            tol[i] = x.tol[i];
        }
        ms = x.ms;
        return *this;
    }

    Scalar v0s[3];
    Scalar v1s[3];
    Scalar v2s[3];
    Scalar v3s[3];
    Scalar v0e[3];
    Scalar v1e[3];
    Scalar v2e[3];
    Scalar v3e[3];
    Scalar err[3]; // error bound of each query, calculated from each scene
    Scalar tol[3]; // domain tolerance to help decide which dimension to split
    Scalar ms;     // minimum separation
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    Scalar toi;
    int aid;
    int bid;
#endif
    int nbr_checks = 0;
};

// the initialized error input, solve tolerance, time interval upper bound, etc.
struct CCDConfig {
    /// @brief the input error bound calculate from the AABB of the whole mesh.
    // Scalar err_in[3];

    /// @brief Tolerance of the co-domain.
    Scalar co_domain_tolerance;

    /// @brief The upper bound of the time interval.
    // Scalar max_t;

    /// @brief
    unsigned int mp_start;

    /// @brief
    unsigned int mp_end;

    /// @brief
    int mp_remaining;

    /// @brief
    long unit_size;

    /// @brief Time of impact.
    Scalar toi;

    /// @brief
    ::cuda::binary_semaphore<::cuda::thread_scope_device> mutex;

    /// @brief If true, use a minimum separation.
    bool use_ms;

    /// @brief If true, allow for zero time of impact.
    bool allow_zero_toi;

    /// @brief The maximum number of iterations.
    int max_iter;

    /// @brief Error overflow flag.
    int overflow_flag;
};

// this is to calculate the vertices of the inclusion function
struct BoxPrimatives {
    bool b[3];

    int dim;

    Scalar t;
    Scalar u;
    Scalar v;

    __device__ void calculate_tuv(const MP_unit& unit);
};

} // namespace scalable_ccd::cuda