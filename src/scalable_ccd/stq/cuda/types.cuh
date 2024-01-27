#pragma once

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/common/scalar.cuh>

#include <algorithm>

#include <stdint.h>
#include <tbb/info.h>

namespace scalable_ccd::stq::gpu {

static const int CPU_THREADS = std::min(tbb::info::default_concurrency(), 64);

typedef enum { x, y, z } Dimension;

typedef unsigned long long int ull;

} // namespace scalable_ccd::stq::gpu
