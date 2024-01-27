#pragma once

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/cuda/scalar.cuh>

#include <algorithm>

#include <stdint.h>
#include <tbb/info.h>

namespace scalable_ccd::cuda::stq {

static const int CPU_THREADS = std::min(tbb::info::default_concurrency(), 64);

typedef enum { x, y, z } Dimension;

typedef unsigned long long int ull;

} // namespace scalable_ccd::cuda::stq
