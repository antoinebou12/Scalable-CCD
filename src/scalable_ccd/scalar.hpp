#pragma once

#include <scalable_ccd/config.hpp>

namespace scalable_ccd {

#ifdef SCALABLE_CCD_USE_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif

} // namespace scalable_ccd