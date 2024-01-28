#include "scalar.cuh"

#include <cuda_fp16.h>

namespace scalable_ccd::cuda {

__device__ Scalar3 operator+(const Scalar3& a, const Scalar3& b)
{
    return make_Scalar3(
        ::__fadd_rz(a.x, b.x), ::__fadd_rz(a.y, b.y), ::__fadd_rz(a.z, b.z));
}

__device__ Scalar3 operator-(const Scalar3& a, const Scalar3& b)
{
    return make_Scalar3(
        ::__fsub_rz(a.x, b.x), ::__fsub_rz(a.y, b.y), ::__fsub_rz(a.z, b.z));
}

__device__ Scalar3 __fdividef(const Scalar3& a, const Scalar b)
{
    return make_Scalar3(
        ::__fdividef(a.x, b), ::__fdividef(a.y, b), ::__fdividef(a.z, b));
}

__device__ Scalar3 __powf(const Scalar3& a, const Scalar b)
{
    return make_Scalar3(::__powf(a.x, b), ::__powf(a.y, b), ::__powf(a.z, b));
}

__device__ Scalar3 abs(const Scalar3& a)
{
    return make_Scalar3(::__habs(a.x), ::__habs(a.y), ::__habs(a.z));
}

} // namespace scalable_ccd::cuda