#include "profiler.hpp"

namespace scalable_ccd::cuda {

#ifdef SCALABLE_CCD_WITH_PROFILING

Profiler& profiler()
{
    static Profiler instance;
    return instance;
}

#endif

} // namespace scalable_ccd::cuda
