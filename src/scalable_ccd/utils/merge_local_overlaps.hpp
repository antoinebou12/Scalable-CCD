#pragma once

#include <tbb/enumerable_thread_specific.h>

#include <vector>

namespace scalable_ccd {

typedef tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>>
    ThreadSpecificOverlaps;

void merge_local_overlaps(
    const ThreadSpecificOverlaps& storages,
    std::vector<std::pair<int, int>>& overlaps);

} // namespace scalable_ccd