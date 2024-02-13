#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/ccd.cuh>
#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>

#include <iostream>
#include <fstream>
#include <unistd.h>

bool file_exists(const char* file_name)
{
    std::ifstream infile(file_name);
    return infile.good();
}

int main(int argc, char** argv)
{
    using namespace scalable_ccd;
    using namespace scalable_ccd::cuda;

    logger().set_level(spdlog::level::trace);

    std::vector<char*> compare;

    char* file_t0;
    char* file_t1;

    file_t0 = argv[1];
    if (file_exists(argv[2])) // CCD
        file_t1 = argv[2];
    else // static CD
        file_t1 = argv[1];

    std::vector<scalable_ccd::cuda::AABB> boxes;
    Eigen::MatrixXd vertices_t0;
    Eigen::MatrixXd vertices_t1;
    Eigen::MatrixXi faces;
    Eigen::MatrixXi edges;

    {
        SCALABLE_CCD_CPU_PROFILE_POINT("parse_mesh");
        parse_mesh(file_t0, file_t1, vertices_t0, vertices_t1, faces, edges);
    }

    {
        SCALABLE_CCD_CPU_PROFILE_POINT("constructBoxes");
        constructBoxes(vertices_t0, vertices_t1, edges, faces, boxes);
    }

    int N = boxes.size();
    int nbox = 0;
    int parallel = 64;
    int devcount = 1;
    int memory_limit_GB = 0;

    int o;
    while ((o = getopt(argc, argv, "c:n:b:p:v:")) != -1) {
        switch (o) {
        case 'c':
            optind--;
            for (; optind < argc && *argv[optind] != '-'; optind++) {
                compare.push_back(argv[optind]);
                // compare_mathematica(overlaps, argv[optind]);
            }
            break;
        case 'n':
            N = atoi(optarg);
            break;
        case 'b':
            nbox = atoi(optarg);
            break;
        case 'p':
            parallel = atoi(optarg);
            break;
        case 'v':
            memory_limit_GB = atoi(optarg);
            break;
        }
    }

    constexpr bool allow_zero_toi = true;
    constexpr Scalar min_distance = 0;

#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<int> result_list;
#endif

    // toi = compute_toi_strategy(
    //     vertices_t0, vertices_t1, edges, faces, 1e6, 0.0, 1e-6);
    // printf("construct_static_collision_candidates\n");
    // boxes.clear();
    // construct_static_collision_candidates(
    //     vertices_t0, edges, faces, overlaps, boxes);

    Scalar toi = ccd(
        vertices_t0, vertices_t1, boxes, // N, nbox, parallel, devcount,
        /*max_iterations=*/-1, /*tolerance=*/1e-6, min_distance, allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
        result_list,
#endif
        memory_limit_GB);

#ifdef SCALABLE_CCD_WITH_PROFILING
    profiler().data()["memory_limit_GB"] = memory_limit_GB;
    profiler().data()["toi"] = toi;
    profiler().print();
#endif
}