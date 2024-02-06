#pragma once

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/scalar.cuh>

namespace scalable_ccd::cuda {

class CCDData {
public:
    __host__ __device__ CCDData() = default;

    __device__ __host__ CCDData& operator=(const CCDData& x)
    {
        if (this == &x)
            return *this;
        v0s = x.v0s;
        v1s = x.v1s;
        v2s = x.v2s;
        v3s = x.v3s;
        v0e = x.v0e;
        v1e = x.v1e;
        v2e = x.v2e;
        v3e = x.v3e;
        err = x.err;
        tol = x.tol;
        ms = x.ms;
        // nbr_checks = x.nbr_checks; ???
        return *this;
    }

    Vector3 v0s;
    Vector3 v1s;
    Vector3 v2s;
    Vector3 v3s;
    Vector3 v0e;
    Vector3 v1e;
    Vector3 v2e;
    Vector3 v3e;
    Array3 err; // error bound of each query, calculated from each scene
    Array3 tol; // domain tolerance to help decide which dimension to split
    Scalar ms;  // minimum separation
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    Scalar toi;
    int aid;
    int bid;
#endif
    int nbr_checks = 0;
};

} // namespace scalable_ccd::cuda