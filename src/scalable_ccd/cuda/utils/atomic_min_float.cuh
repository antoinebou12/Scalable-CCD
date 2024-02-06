#pragma once

namespace scalable_ccd::cuda {

// https://stackoverflow.com/a/51549250/13206140
__device__ __forceinline__ float atomicMin(float* addr, float value)
{
    float old;
    old = (value >= 0)
        ? __int_as_float(::atomicMin((int*)addr, __float_as_int(value)))
        : __uint_as_float(
            ::atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ double atomicMin(double* addr, double value)
{
    assert(value >= 0); // __ulonglong_as_double is not a thing
    float old;
    // old = (value >= 0)
    //     ? __longlong_as_double(
    //         ::atomicMin((long long*)addr, __double_as_longlong(value)))
    //     : __ulonglong_as_double(::atomicMax(
    //         (unsigned long long*)addr, __double_as_longlong(value)));
    old = __longlong_as_double(
        ::atomicMin((long long*)addr, __double_as_longlong(value)));
    return old;
}

} // namespace scalable_ccd::cuda