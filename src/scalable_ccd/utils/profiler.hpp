#pragma once

#include <scalable_ccd/config.hpp>

#ifdef SCALABLE_CCD_WITH_PROFILING

#include <scalable_ccd/utils/timer.hpp>
#include <scalable_ccd/cuda/utils/timer.cuh>
#include <scalable_ccd/utils/logger.hpp>

#include <nlohmann/json.hpp>

#define SCALABLE_CCD_CPU_PROFILE_POINT(...)                                    \
    scalable_ccd::CPUProfilePoint __scalable_ccd_profile_point(__VA_ARGS__)
#define SCALABLE_CCD_GPU_PROFILE_POINT(...)                                    \
    scalable_ccd::GPUProfilePoint __scalable_ccd_profile_point(__VA_ARGS__)

namespace scalable_ccd {

class Profiler {
public:
    Profiler() = default;

    // ~Profiler() { print(); }

    void clear() { m_data.clear(); }

    void start(const std::string& name, const bool on_gpu)
    {
        current_scope.push_back(name);
        if (!m_data.contains(current_scope)) {
            m_data[current_scope] = {
                { "on_cpu", !on_gpu },
                { "on_gpu", on_gpu },
                { "time_ms", 0 },
            };
        } else {
            m_data[current_scope]["on_gpu"] =
                on_gpu || m_data[current_scope]["on_gpu"].get<bool>();
            m_data[current_scope]["on_cpu"] =
                !on_gpu || m_data[current_scope]["on_cpu"].get<bool>();
        }
    }

    void stop(const double time_ms)
    {
        logger().trace(
            "Timer stopped for \"{}\": {:.6f} ms", current_scope.to_string(),
            time_ms);

        assert(m_data.contains(current_scope));
        assert(m_data.at(current_scope).contains("time_ms"));
        m_data[current_scope]["time_ms"] =
            m_data[current_scope]["time_ms"].get<double>() + time_ms;
        current_scope.pop_back();
    }

    void print() const { logger().info("{}", m_data.dump(2)); }

    const nlohmann::json& data() const { return m_data; }
    nlohmann::json& data() { return m_data; }

protected:
    nlohmann::json m_data;
    nlohmann::json::json_pointer current_scope;
};

Profiler& profiler();

template <class Timer, bool on_gpu> class ProfilePoint {
public:
    ProfilePoint(const std::string& name) : ProfilePoint(profiler(), name) { }

    ProfilePoint(Profiler& p_profiler, const std::string& name)
        : m_profiler(p_profiler)
    {
        m_profiler.start(name, on_gpu);
        timer.start();
    }

    ~ProfilePoint()
    {
        timer.stop();
        m_profiler.stop(timer.getElapsedTimeInMilliSec());
    }

protected:
    Profiler& m_profiler;
    Timer timer;
};

using CPUProfilePoint = ProfilePoint<scalable_ccd::Timer, false>;
using GPUProfilePoint = ProfilePoint<scalable_ccd::cuda::Timer, true>;

} // namespace scalable_ccd

#else

#define SCALABLE_CCD_CPU_PROFILE_POINT(...)
#define SCALABLE_CCD_GPU_PROFILE_POINT(...)

#endif