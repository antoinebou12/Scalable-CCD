#pragma once

#include <scalable_ccd/cuda/utils/assert.cuh>

namespace scalable_ccd::cuda::stq {

template <typename T> class device_variable {
public:
    device_variable() { gpuErrchk(cudaMalloc(&d_ptr, sizeof(T))); }

    device_variable(const T& value)
    {
        gpuErrchk(cudaMalloc(&d_ptr, sizeof(T)));
        gpuErrchk(cudaMemcpy(d_ptr, &value, sizeof(T), cudaMemcpyHostToDevice));
    }

    ~device_variable() { gpuErrchk(cudaFree(d_ptr)); }

    void operator=(T value)
    {
        gpuErrchk(cudaMemcpy(d_ptr, &value, sizeof(T), cudaMemcpyHostToDevice));
    }

    operator T() const
    {
        T value;
        gpuErrchk(cudaMemcpy(&value, d_ptr, sizeof(T), cudaMemcpyDeviceToHost));
        return value;
    }

    T* ptr() { return d_ptr; }

private:
    T* d_ptr;
};

} // namespace scalable_ccd::cuda::stq