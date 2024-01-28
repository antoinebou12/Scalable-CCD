#include "merge_local_overlaps.hpp"

namespace scalable_ccd {

void merge_local_overlaps(
    const ThreadSpecificOverlaps& storages,
    std::vector<std::pair<int, int>>& overlaps)
{
    overlaps.clear();
    size_t num_overlaps = overlaps.size();
    for (const auto& local_overlaps : storages) {
        num_overlaps += local_overlaps.size();
    }
    // serial merge!
    overlaps.reserve(num_overlaps);
    for (const auto& local_overlaps : storages) {
        overlaps.insert(
            overlaps.end(), local_overlaps.begin(), local_overlaps.end());
    }
}

} // namespace scalable_ccd