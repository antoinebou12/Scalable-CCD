#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/scalar.hpp>
#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/broad_phase/utils.cuh>
#include <scalable_ccd/cuda/narrow_phase/narrow_phase.cuh>
#include <scalable_ccd/utils/profiler.hpp>
#include <scalable_ccd/utils/timer.hpp>
#include <scalable_ccd/utils/logger.hpp>

#include <iostream>
#include <fstream>
#include <unistd.h>

bool file_exists(const char* fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

int main(int argc, char** argv)
{
    using namespace scalable_ccd;
    using namespace scalable_ccd::cuda;

    logger().set_level(spdlog::level::trace);

    MemoryHandler* memhandle = new MemoryHandler();

    std::vector<char*> compare;

    char* filet0;
    char* filet1;

    filet0 = argv[1];
    if (file_exists(argv[2])) // CCD
        filet1 = argv[2];
    else // static CD
        filet1 = argv[1];

    std::vector<scalable_ccd::cuda::AABB> boxes;
    Eigen::MatrixXd vertices_t0;
    Eigen::MatrixXd vertices_t1;
    Eigen::MatrixXi faces;
    Eigen::MatrixXi edges;

    {
        SCALABLE_CCD_CPU_PROFILE_POINT("parse_mesh");
        parse_mesh(filet0, filet1, vertices_t0, vertices_t1, faces, edges);
    }

    {
        SCALABLE_CCD_CPU_PROFILE_POINT("constructBoxes");
        constructBoxes(vertices_t0, vertices_t1, edges, faces, boxes);
    }
    int N = boxes.size();
    int nbox = 0;
    int parallel = 64;
    int devcount = 1;
    int limitGB = 0;

    // std::copy(from_vector.begin(), from_vector.end(), to_vector.begin());

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
            limitGB = atoi(optarg);
            break;
        }
    }
    std::vector<std::pair<int, int>> overlaps;
    std::vector<int> result_list;
    Scalar toi;

    bool allow_zero_toi = true;
    Scalar min_distance = 0;

    // toi = compute_toi_strategy(vertices_t0, vertices_t1, edges, faces, 1e6,
    // 0.0,
    //                            1e-6);
    // printf("construct_static_collision_candidates\n");
    // boxes.clear();
    // construct_static_collision_candidates(vertices_t0, edges, faces,
    // overlaps,
    //                                       boxes);

    run_ccd(
        boxes, memhandle, vertices_t0, vertices_t1, r, N, nbox, parallel,
        devcount, limitGB, overlaps, result_list, allow_zero_toi, min_distance,
        toi);
    r.j_object["limitGB"] = limitGB;
    r.j_object["toi"] = toi;
    r.Print();
    // r.Print();
    // std::cout << r.j_object["run_ccd (narrowphase)"];

    // std::cout << "result_list " << result_list.size() << std::endl;
    // for (int i = 0; i < result_list.size(); i++)
    //   result_list[i] = 1;

    for (auto i : compare) {
        compare_mathematica(overlaps, result_list, i);
    }
    std::cout << std::endl;

    profiler().print();
}