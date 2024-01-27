#pragma once

#include <scalable_ccd/scalar.hpp>

// #include <cuda.h>
// #include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace scalable_ccd {

#ifdef SCALABLE_CCD_WITH_DOUBLE

using Scalar2 = double2;
using Scalar3 = double3;

__host__ __device__ inline Scalar2
make_Scalar2(const Scalar& a, const Scalar& b)
{
    return make_double2(a, b);
}

__host__ __device__ inline Scalar3
make_Scalar3(const Scalar& a, const Scalar& b, const Scalar& c)
{
    return make_double3(a, b, c);
}

#else

using Scalar2 = float2;
using Scalar3 = float3;

__host__ __device__ inline Scalar2
make_Scalar2(const Scalar& a, const Scalar& b)
{
    return make_float2(a, b);
}

__host__ __device__ inline Scalar3
make_Scalar3(const Scalar& a, const Scalar& b, const Scalar& c)
{
    return make_float3(a, b, c);
}

#endif

// __host__ __device__ struct half3 {
//     __half x;
//     __half y;
//     __half z;
// };

// __host__ __device__ inline half3 make_half3(__half x, __half y, __half z)
// {
//     half3 t;
//     t.x = x;
//     t.y = y;
//     t.z = z;
//     return t;
// }

// __host__ __device__ inline half3 make_half3(float x, float y, float z)
// {
//     half3 t;
//     t.x = __float2half(x);
//     t.y = __float2half(y);
//     t.z = __float2half(z);
//     return t;
// }

__device__ inline Scalar3 operator+(const Scalar3& a, const Scalar3& b)
{
    return make_Scalar3(
        ::__fadd_rz(a.x, b.x), ::__fadd_rz(a.y, b.y), ::__fadd_rz(a.z, b.z));
}

__device__ inline Scalar3 operator-(const Scalar3& a, const Scalar3& b)
{
    return make_Scalar3(
        ::__fsub_rz(a.x, b.x), ::__fsub_rz(a.y, b.y), ::__fsub_rz(a.z, b.z));
}

__device__ inline Scalar3 __fdividef(const Scalar3& a, const Scalar b)
{
    return make_Scalar3(
        ::__fdividef(a.x, b), ::__fdividef(a.y, b), ::__fdividef(a.z, b));
}

__device__ inline Scalar3 __powf(const Scalar3& a, const Scalar b)
{
    return make_Scalar3(::__powf(a.x, b), ::__powf(a.y, b), ::__powf(a.z, b));
}

__device__ inline Scalar3 abs(const Scalar3& a)
{
    return make_Scalar3(::__habs(a.x), ::__habs(a.y), ::__habs(a.z));
}

} // namespace scalable_ccd