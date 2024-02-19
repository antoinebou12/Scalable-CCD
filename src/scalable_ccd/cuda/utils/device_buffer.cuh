#pragma once

#include <scalable_ccd/cuda/utils/assert.cuh>
#include <scalable_ccd/cuda/utils/device_variable.cuh>

#include <thrust/device_vector.h>

namespace scalable_ccd::cuda {

/// @brief A buffer of stored data on the device.
/// @tparam T The type of the data stored in the buffer.
template <typename T> class DeviceBuffer {
public:
    DeviceBuffer() : DeviceBuffer(0) { }

    DeviceBuffer(const size_t capacity) : m_storage(capacity), m_size(0) { }

    // not copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    int size() const { return m_size; }
    size_t capacity() const { return m_storage.size(); }

    T* data() { return thrust::raw_pointer_cast(m_storage.data()); }
    const T* data() const { return thrust::raw_pointer_cast(m_storage.data()); }

    const T* begin() const { return data(); }
    const T* end() const { return data() + size(); }

    void reserve(const size_t n) { m_storage.resize(n); }

    /// @brief Reset the size of the buffer to zero.
    void clear() { m_size = 0; }

    template <typename> friend class RawDeviceBuffer;

private:
    thrust::device_vector<T> m_storage;
    DeviceVariable<int> m_size;
};

/// @brief A raw buffer of stored data on the device.
template <typename T> struct RawDeviceBuffer {
    RawDeviceBuffer(DeviceBuffer<T>& buffer)
        : data(buffer.data())
        , size(&buffer.m_size)
        , capacity(buffer.capacity())
    {
    }

    __device__ T& operator[](const int i) { return data[i]; }

    __device__ const T& operator[](const int i) const { return data[i]; }

    __device__ void push(const T& value)
    {
        const int i = atomicAdd(size, 1);
        assert(i < capacity);
        data[i] = value;
    }

    T* const data;
    int* const size;
    const size_t capacity;
};

} // namespace scalable_ccd::cuda