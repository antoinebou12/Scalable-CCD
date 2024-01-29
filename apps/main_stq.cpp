#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/utils/timer.hpp>
#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>

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

    int nbox = 0;
    app.add_option("-b", nbox, "Number of boxes to test");

    // int parallel = 1;
    // app.add_option("-p", parallel, "Number of threads");

    CLI11_PARSE(app, argc, argv);

    // ------------------------------------------------------------------------
    // Load meshes

    std::vector<AABB> vertex_boxes, edge_boxes, face_boxes;
    parse_mesh(file_t0, file_t1, vertex_boxes, edge_boxes, face_boxes);

    // ------------------------------------------------------------------------
    // Run

    static const int CPU_THREADS =
        std::min(tbb::info::default_concurrency(), 64);
    tbb::global_control thread_limiter(
        tbb::global_control::max_allowed_parallelism, CPU_THREADS);
    spdlog::trace("Running with {:d} threads", CPU_THREADS);

    Timer timer;
    timer.start();

    int sort_axis = 0;
    std::vector<std::pair<int, int>> fv_overlaps;
    sort_and_sweep(face_boxes, vertex_boxes, sort_axis, fv_overlaps);

    sort_axis = 1;
    std::vector<std::pair<int, int>> ee_overlaps;
    sort_and_sweep(edge_boxes, sort_axis, ee_overlaps);

    timer.stop();
    spdlog::trace("Elapsed time: {:.6f} ms", timer.getElapsedTimeInMilliSec());

    // ------------------------------------------------------------------------
    // Compare

    spdlog::info(
        "FV Overlaps: {}; EE Overlaps: {}", fv_overlaps.size(),
        ee_overlaps.size());
    for (const std::string& i : compare) {
        // TODO: update this to use the new format
        compare_mathematica(fv_overlaps, i);
        compare_mathematica(ee_overlaps, i);
    }

    return 0;
}