#pragma once

#include <vector>

namespace scalable_ccd::cuda {

class Timer {
public:
    Timer()
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
    }

    ~Timer()
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_end);
    }

    void tick(cudaStream_t streamid = 0) { cudaEventRecord(m_start, streamid); }

    void tock(cudaStream_t streamid = 0) { cudaEventRecord(m_end, streamid); }

    float elapsed() const
    {
        float ms;
        cudaEventSynchronize(m_end);
        cudaEventElapsedTime(&ms, m_start, m_end);
        return ms;
    }

    void clear() { m_records.clear(); }

    void record(std::string tag)
    {
        tock();
        m_records.push_back(make_pair(tag, elapsed()));
    }

private:
    cudaEvent_t m_start, m_end;
    std::vector<std::pair<std::string, float>> m_records;
};

} // namespace scalable_ccd::cuda
