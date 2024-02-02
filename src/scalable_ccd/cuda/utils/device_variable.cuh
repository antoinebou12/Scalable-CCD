#pragma once

#include <scalable_ccd/cuda/utils/assert.cuh>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

namespace scalable_ccd::cuda {

template <typename T> class DeviceVariable {
public:
    DeviceVariable() { d_ptr = thrust::device_malloc<T>(1); }

    DeviceVariable(const T& value)
    {
        d_ptr = thrust::device_malloc<T>(1);
        *d_ptr = value;
    }

    // not copyable
    DeviceVariable(const DeviceVariable&) = delete;
    DeviceVariable& operator=(const DeviceVariable&) = delete;

    ~DeviceVariable() { thrust::device_free(d_ptr); }

    void operator=(T value) { *d_ptr = value; }

    /// @brief Get the pointer to the device memory.
    /// @note This does not return the address of this object, but the address of the device memory.
    T* operator&() { return thrust::raw_pointer_cast(d_ptr); }
    const T* operator&() const { return thrust::raw_pointer_cast(d_ptr); }

    operator T() const { return *d_ptr; }

private:
    thrust::device_ptr<T> d_ptr;
};

} // namespace scalable_ccd::cuda