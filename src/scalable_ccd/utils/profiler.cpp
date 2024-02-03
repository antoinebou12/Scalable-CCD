#include "profiler.hpp"

namespace scalable_ccd {

#ifdef SCALABLE_CCD_WITH_PROFILING

Profiler& profiler()
{
    static Profiler instance;
    return instance;
}

#endif

} // namespace scalable_ccd
