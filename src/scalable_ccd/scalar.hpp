#pragma once

#include <scalable_ccd/config.hpp>

#include <Eigen/Core>

#include <limits>
#include <cmath>
#include <cfloat>

namespace scalable_ccd {

#ifdef SCALABLE_CCD_USE_DOUBLE
using Scalar = double;
constexpr Scalar SCALAR_MAX = DBL_MAX;
#else
using Scalar = float;
constexpr Scalar SCALAR_MAX = FLT_MAX;
#endif

using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Array3 = Eigen::Array<Scalar, 3, 1>;
using VectorMax3 =
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1>;
using ArrayMax3 =
    Eigen::Array<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1>;

/// @brief Get the next representable floating point number in the direction of negative infinity.
/// @param x Input number.
/// @return The next representable floating point number in the direction of negative infinity.
inline Scalar nextafter_down(const Scalar x)
{
#ifdef SCALABLE_CCD_USE_DOUBLE
    return nextafter(x, -SCALAR_MAX);
#else
    return nextafterf(x, -SCALAR_MAX);
#endif
}

/// @brief Get the next representable floating point number in the direction of positive infinity.
/// @param x Input number.
/// @return The next representable floating point number in the direction of positive infinity.
inline Scalar nextafter_up(const Scalar x)
{
#ifdef SCALABLE_CCD_USE_DOUBLE
    return nextafter(x, SCALAR_MAX);
#else
    return nextafterf(x, SCALAR_MAX);
#endif
}

} // namespace scalable_ccd