#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/timer.hpp>
#include <scalable_ccd/stq/aabb.hpp>
#include <scalable_ccd/stq/sweep.hpp>

#include <CLI/CLI.hpp>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <tbb/global_control.h>
#include <tbb/info.h>

#include <spdlog/spdlog.h>

#include <set>
#include <vector>
#include <array>
#include <fstream>

int main(int argc, char** argv)
{
    using namespace scalable_ccd;

    spdlog::set_level(spdlog::level::trace);

    CLI::App app("STQ CPU");

    std::string file_t0;
    app.add_option("file_t0", file_t0, "Mesh @ t=0")->required();

    std::string file_t1;
    app.add_option("file_t1", file_t1, "Mesh @ t=1")->required();

    std::vector<std::string> compare;
    app.add_option("-c,--compare", compare, "Compare with Mathematica");

    int N = 0;
    app.add_option("-n", N, "Number of boxes to test");

    int nbox = 0;
    app.add_option("-b", nbox, "Number of boxes to test");

    // int parallel = 1;
    // app.add_option("-p", parallel, "Number of threads");

    CLI11_PARSE(app, argc, argv);

    // ------------------------------------------------------------------------
    // Load meshes

    std::vector<stq::Aabb> boxes;
    parse_mesh(file_t0, file_t1, boxes);

    if (N <= 0) {
        N = boxes.size();
    }

    int n = boxes.size();
    std::vector<stq::Aabb> boxes_batching;
    // boxes_batching.resize(n);
    std::copy(boxes.begin(), boxes.end(), std::back_inserter(boxes_batching));

    // ------------------------------------------------------------------------
    // Run

    static const int CPU_THREADS =
        std::min(tbb::info::default_concurrency(), 64);
    tbb::global_control thread_limiter(
        tbb::global_control::max_allowed_parallelism, CPU_THREADS);
    spdlog::trace("Running with {:d} threads", CPU_THREADS);

    std::vector<std::pair<int, int>> overlaps;
    std::size_t count = 0;

    Timer timer;
    timer.start();

    sweep_cpu_single_batch(boxes_batching, n, N, overlaps);
    while (overlaps.size()) {
        count += overlaps.size();
        sweep_cpu_single_batch(boxes_batching, n, N, overlaps);
    }

    timer.stop();
    spdlog::trace("Elapsed time: {:.6f} ms", timer.getElapsedTimeInMilliSec());

    // ------------------------------------------------------------------------
    // Compare

    spdlog::info("Overlaps: {}", overlaps.size());
    spdlog::info("Final count: {}", count);
    for (const std::string& i : compare) {
        compare_mathematica(overlaps, i);
    }

    return 0;
}