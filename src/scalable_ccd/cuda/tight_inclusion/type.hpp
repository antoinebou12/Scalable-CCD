#pragma once

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/scalar.cuh>

#include <array>
#include <assert.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#include <cuda/semaphore>
#include <limits>
#include <utility>

namespace scalable_ccd::cuda {
///////////////////////////////
// here are the parameters for the memory pool
// static const int MAX_QUERIES = 1e6;
static const int MAX_CHECKS = 1e6;

///////////////////////////////

// THE FOLLOWING VALUES ARE JUST FOR DEBUGGING
// #define GPUTI_GO_DEAP_HEAP
// static const int TESTING_ID = 219064;
// static const int TEST_SIZE = 1;

// TODO next when spliting time intervals, check if overlaps the current toi,
// then decide if we push it into the heap the reason of considerting it is that
// the limited heap size. token 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

// overflow instructions
static const int NO_OVERFLOW = 0;
static const int BISECTION_OVERFLOW = 1;
static const int HEAP_OVERFLOW = 2;
static const int ITERATION_OVERFLOW = 3;

class interval_pair {
public:
    __device__ interval_pair(const Singleinterval& itv);
    __device__ interval_pair() {};
    Singleinterval first;
    Singleinterval second;
};

void print_vector(Scalar* v, int size);
void print_vector(int* v, int size);

// this is to record the interval related info

// this is to calculate the vertices of the inclusion function
class BoxPrimatives {
public:
    bool b[3];
    int dim;
    Scalar t;
    Scalar u;
    Scalar v;
    __device__ void calculate_tuv(const MP_unit& unit);
};
CCDData array_to_ccd(const std::array<std::array<Scalar, 3>, 8>& a);
__device__ void single_test_wrapper(CCDData* vfdata, bool& result);
__device__ Scalar calculate_ee(const CCDData& data_in, const BoxPrimatives& bp);
} // namespace scalable_ccd::cuda