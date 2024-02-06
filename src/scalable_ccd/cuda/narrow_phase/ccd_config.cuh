#pragma once

#include <scalable_ccd/cuda/scalar.cuh>

namespace scalable_ccd::cuda {

// the initialized error input, solve tolerance, time interval upper bound, etc.
struct CCDConfig {
    /// @brief Tolerance of the co-domain.
    Scalar co_domain_tolerance;

    /// @brief The upper bound of the time interval.
    // Scalar max_t;

    /// @brief
    unsigned int mp_start; // TODO: remove this

    /// @brief
    unsigned int mp_end; // TODO: remove this

    /// @brief
    int mp_remaining; // TODO: remove this

    /// @brief
    long unit_size; // TODO: remove this

    /// @brief Time of impact.
    Scalar toi;

    /// @brief If true, use a minimum separation.
    bool use_ms;

    /// @brief If true, allow for zero time of impact.
    bool allow_zero_toi;

    /// @brief The maximum number of iterations.
    int max_iter;

    /// @brief Error overflow flag.
    int overflow_flag; // TODO: remove this
};

} // namespace scalable_ccd::cuda