#pragma once

#include <scalable_ccd/utils/timer.hpp>

#include <string>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace scalable_ccd::cuda {

// template <typename... Arguments>
// void record(std::string tag, void(*f)(Arguments...), Arguments... args) {
//       Timer timer;
//       timer.start();
//       f(args...);
//       timer.stop();
//       double elapsed = 0;
//       elapsed += timer.getElapsedTimeInMicroSec();
//       spdlog::trace("{} : {:.6f} ms", tag, elapsed);
// };

struct Record {
    Timer timer;
    cudaEvent_t start, stop;
    std::string tag;
    json j_object;
    bool gpu_timer_on = false;

    Record() {
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // gpu_timer_on = false;
    };

    Record(json& jtmp)
    {
        j_object = jtmp;
        Record();
    };

    void Start(const std::string& s, bool gpu = false)
    {
        tag = s;
        if (!gpu)
            timer.start();
        else {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            // spdlog::trace("Starting gpu timer for {}",  s);
            cudaEventRecord(start);
            gpu_timer_on = true;
        }
    }

    void Start(const std::string& s, json& jtmp, bool gpu = false)
    {
        j_object = jtmp;
        Start(s, gpu);
    }

    void Stop()
    {
        float elapsed = 0; // was double
        if (!gpu_timer_on) {
            timer.stop();
            elapsed += timer.getElapsedTimeInMilliSec();
            spdlog::trace("Cpu timer stopped for {}: {:.6f} ms", tag, elapsed);
        } else {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            spdlog::trace("Gpu timer stopped for {}: {:.6f} ms", tag, elapsed);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            gpu_timer_on = false;
        }
        // j_object[tag]=elapsed;
        if (j_object.contains(tag))
            j_object[tag] = (double)j_object[tag] + elapsed;
        else
            j_object[tag] = elapsed;
        spdlog::trace("{} : {:.3f} ms", tag, elapsed);
    }

    void Print() { spdlog::info("{}", j_object.dump()); }

    json Dump() { return j_object; }

    void Clear() { j_object.clear(); }
};

} // namespace scalable_ccd::cuda