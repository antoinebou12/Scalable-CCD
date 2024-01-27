#pragma once

#include <chrono>

namespace scalable_ccd {

class Timer {
private:
    using clock = std::chrono::steady_clock;

public:
    Timer() { }
    virtual ~Timer() { }

    inline void start()
    {
        is_running = true;
        m_start = clock::now();
    }

    inline void stop()
    {
        if (!is_running)
            return;
        m_stop = clock::now();
        is_running = false;
    }

    double getElapsedTimeInSec()
    {
        return std::chrono::duration<double>(m_stop - m_start).count();
    }

    double getElapsedTimeInMilliSec()
    {
        return std::chrono::duration<double, std::milli>(m_stop - m_start)
            .count();
    }

    double getElapsedTimeInMicroSec()
    {
        return std::chrono::duration<double, std::micro>(m_stop - m_start)
            .count();
    }

    double getElapsedTimeInNanoSec()
    {
        return std::chrono::duration<double, std::nano>(m_stop - m_start)
            .count();
    }

private:
    std::chrono::time_point<clock> m_start, m_stop;
    bool is_running = false;
};

} // namespace scalable_ccd