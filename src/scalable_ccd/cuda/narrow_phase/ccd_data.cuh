#pragma once

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/scalar.cuh>

namespace scalable_ccd::cuda {

struct CCDData {
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