#pragma once

#include <scalable_ccd/config.hpp>

#include <limits>
#include <cmath>

namespace scalable_ccd {

#ifdef SCALABLE_CCD_USE_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif

/// @brief Get the next representable floating point number in the direction of negative infinity.
/// @param x Input number.
/// @return The next representable floating point number in the direction of negative infinity.
inline Scalar nextafter_down(const Scalar x)
{
#ifdef SCALABLE_CCD_USE_DOUBLE
    return nextafter(x, -std::numeric_limits<double>::max());
#else
    return nextafterf(x, -std::numeric_limits<float>::max());
#endif
}

/// @brief Get the next representable floating point number in the direction of positive infinity.
/// @param x Input number.
/// @return The next representable floating point number in the direction of positive infinity.
inline Scalar nextafter_up(const Scalar x)
{
#ifdef SCALABLE_CCD_USE_DOUBLE
    return nextafter(x, std::numeric_limits<double>::max());
#else
    return nextafterf(x, std::numeric_limits<float>::max());
#endif
}

} // namespace scalable_ccd