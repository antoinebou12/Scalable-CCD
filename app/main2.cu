#include <assert.h>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <filesystem>
// #include <cuda.h>
// #include <cuda_runtime.h>

// #define CCD_USE_DOUBLE

#include <stq/gpu/groundtruth.cuh>
#include <stq/gpu/broadphase.cuh>
#include <stq/gpu/util.cuh>
#include <stq/gpu/memory.cuh>
#include <stq/gpu/io.cuh>
#include <stq/gpu/pca.cuh>

#include <igl/writeOBJ.h>
#include <igl/writePLY.h>

#include <spdlog/spdlog.h>

using namespace std;
using namespace stq::gpu;

// spdlog::set_level(spdlog::level::trace);

bool is_file_exist(const char* fileName)
{
    ifstream infile(fileName);
    return infile.good();
}

int main(int argc, char** argv)
{
    spdlog::set_level(static_cast<spdlog::level::level_enum>(0));
    vector<char*> compare;

    MemHandler* memhandle = new MemHandler();

    char* filet0;
    char* filet1;

    filet0 = argv[1];
    if (is_file_exist(argv[2]))
        filet1 = argv[2];
    else
        filet1 = argv[1];

    vector<Aabb> boxes;
    Eigen::MatrixXd vertices_t0;
    Eigen::MatrixXd vertices_t1;
    Eigen::MatrixXd pca_vertices_t0;
    Eigen::MatrixXd pca_vertices_t1;
    Eigen::MatrixXi faces;
    Eigen::MatrixXi edges;

    int nbox = 0;
    int parallel = 0;
    bool evenworkload = false;
    int devcount = 1;
    bool pairing = false;
    bool sharedqueue_mgpu = false;
    bool bigworkerqueue = false;
    bool pca = false;

    int memlimit = 0;

    int o;
    while ((o = getopt(argc, argv, "c:n:b:p:d:v:WPQZ")) != -1) {
        switch (o) {
        case 'c':
            optind--;
            for (; optind < argc && *argv[optind] != '-'; optind++) {
                compare.push_back(argv[optind]);
            }
            break;
        // case 'n':
        //   N = atoi(optarg);
        //   break;
        case 'b':
            nbox = atoi(optarg);
            break;
        case 'v':
            memlimit = atoi(optarg);
            break;
        case 'p':
            parallel = stoi(optarg);
            break;
        case 'd':
            devcount = atoi(optarg);
            break;
        case 'P':
            pca = true;
            break;
        }
    }

    parseMesh(filet0, filet1, vertices_t0, vertices_t1, faces, edges);
    spdlog::trace(
        "vertices_t0 : {:d} x {:d}", static_cast<int>(vertices_t0.rows()),
        static_cast<int>(vertices_t0.cols()));
    if (pca) {
        nipalsPCA(vertices_t0, vertices_t1);
        std::string filet0Str(filet0);
        std::filesystem::path p(filet0Str);
        std::filesystem::path filename = p.filename();
        std::string ext = filet0Str.substr(filet0Str.rfind('.') + 1);
        std::filesystem::path current_path = std::filesystem::current_path();
        std::string outname = current_path.parent_path().string() + "/"
            + filename.stem().string() + "_pca." + ext;
        if (ext == "obj")
            igl::writeOBJ(outname, vertices_t0, faces);
        else
            igl::writePLY(outname, vertices_t0, faces);
    }
    constructBoxes(vertices_t0, vertices_t1, edges, faces, boxes);
    size_t N = boxes.size();

    vector<pair<int, int>> overlaps;
    int2* d_overlaps; // device
    int* d_count;     // device
    int tidstart = 0;

    if (devcount == 1)
        runBroadPhase(
            boxes.data(), memhandle, N, nbox, overlaps, d_overlaps, d_count,
            parallel, tidstart, devcount, memlimit);
    else
        runBroadPhaseMultiGPU(
            boxes.data(), N, nbox, overlaps, parallel, devcount);

    spdlog::debug("Final CPU overlaps size : {:d}", overlaps.size());

    for (auto compFile : compare) {
        compare_mathematica(overlaps, compFile);
    }
}
