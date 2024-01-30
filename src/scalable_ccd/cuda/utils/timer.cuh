#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace scalable_ccd::cuda {

class Timer {
public:
    Timer()
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }

    ~Timer()
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }

    void start(cudaStream_t streamid = 0)
    {
        is_running = true;
        cudaEventRecord(m_start, streamid);
    }

    void stop(cudaStream_t streamid = 0)
    {
        if (!is_running)
            return;
        cudaEventRecord(m_stop, streamid);
        is_running = false;
    }

    double getElapsedTimeInMilliSec() const
    {
        float ms;
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&ms, m_start, m_stop);
        return ms;
    }

private:
    cudaEvent_t m_start, m_stop;
    bool is_running = false;
};

} // namespace scalable_ccd::cuda
