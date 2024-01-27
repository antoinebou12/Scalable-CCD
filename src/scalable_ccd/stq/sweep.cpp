#include "sweep.hpp"

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <algorithm> // std::sort
#include <vector>    // std::vector

#include <spdlog/spdlog.h>

namespace scalable_ccd::stq {

namespace {
    // typedef StructAlignment(32) std::array<_simd, 6> SimdObject;

    /// @brief Determine if two boxes are overlapping.
    /// @param a The first box.
    /// @param b The second box.
    /// @param skip_axis The axis to skip.
    /// @return True if the boxes are overlapping, false otherwise.
    bool are_overlapping(const Aabb& a, const Aabb& b, const int skip_axis)
    {
        assert(skip_axis >= 0 && skip_axis <= 2);
        if (skip_axis == 0) {
            return a.max[1] >= b.min[1] && a.min[1] <= b.max[1]
                && a.max[2] >= b.min[2] && a.min[2] <= b.max[2];
        } else if (skip_axis == 1) {
            return a.max[0] >= b.min[0] && a.min[0] <= b.max[0]
                && a.max[2] >= b.min[2] && a.min[2] <= b.max[2];
        } else {
            return a.max[0] >= b.min[0] && a.min[0] <= b.max[0]
                && a.max[1] >= b.min[1] && a.min[1] <= b.max[1];
        }
    }

    /// @brief Determine if two elements share a vertex.
    /// @param a Vertex ids of the first element.
    /// @param b Vertex ids of the second element.
    /// @return True if the elements share a vertex, false otherwise.
    bool
    share_a_vertex(const std::array<int, 3>& a, const std::array<int, 3>& b)
    {
        return a[0] == b[0] || a[0] == b[1] || a[0] == b[2] || a[1] == b[0]
            || a[1] == b[1] || a[1] == b[2] || a[2] == b[0] || a[2] == b[1]
            || a[2] == b[2];
    }

    // https://stackoverflow.com/questions/3909272/sorting-two-corresponding-arrays
    class SortIndices {
    private:
        const std::vector<Aabb>& boxes;
        const int axis;

    public:
        SortIndices(const std::vector<Aabb>& _boxes, const int _axis)
            : boxes(_boxes)
            , axis(_axis)
        {
        }

        bool operator()(int i, int j) const
        {
            return boxes[i].min[axis] < boxes[j].min[axis];
        }
    };

    typedef tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>>
        ThreadSpecificOverlaps;

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

    /// @brief Sweep the boxes along the given axis (without memory check).
    /// @param[in] boxes Boxes to sweep.
    /// @param[in] n Number of boxes.
    /// @param[in,out] sort_axis Axis to sweep along and the next axis to sort along.
    /// @param[out] overlaps Overlaps to populate.
    void batched_sweep(
        const std::vector<Aabb>& boxes,
        const int batch_start,
        const int batch_end,
        int& sort_axis,
        std::vector<std::pair<int, int>>& overlaps)
    {
        ThreadSpecificOverlaps storages;

        tbb::parallel_for(
            tbb::blocked_range<long>(batch_start, batch_end),
            [&](const tbb::blocked_range<long>& r) {
                auto& local_overlaps = storages.local();

                for (long i = r.begin(); i < r.end(); i++) {
                    const Aabb& a = boxes[i];

                    for (long j = i + 1; j < boxes.size(); j++) {
                        const Aabb& b = boxes[j];

                        if (a.max[sort_axis] < b.min[sort_axis]) {
                            break;
                        }

                        if (are_overlapping(a, b, sort_axis)
                            && is_valid_pair(a.vertexIds, b.vertexIds)
                            && !share_a_vertex(a.vertexIds, b.vertexIds)) {
                            local_overlaps.emplace_back(a.id, b.id);
                        }
                    }
                }
            });

        merge_local_overlaps(storages, overlaps);
    }
} // namespace

bool is_valid_pair(const std::array<int, 3>& a, const std::array<int, 3>& b)
{
    return (is_vertex(a) && is_face(b)) || (is_face(a) && is_vertex(b))
        || (is_edge(a) && is_edge(b));
}

void sort_along_axis(const int axis, std::vector<Aabb>& boxes)
{
    // const std::vector<Aabb> orig_boxes = boxes;

    // std::vector<int> box_indices(boxes.size());
    // std::iota(box_indices.begin(), box_indices.end(), 0);
    // tbb::parallel_sort(
    //     box_indices.begin(), box_indices.end(), SortIndices(boxes, axis));

    // for (size_t i = 0; i < boxes.size(); i++) {
    //     boxes[i] = orig_boxes[box_indices[i]];
    // }

    tbb::parallel_sort(
        boxes.begin(), boxes.end(),
        [axis](const Aabb& a, const Aabb& b) -> bool {
            return (a.min[axis] < b.min[axis]);
        });
}

// Adaptively split the boxes if the memory limit is exceeded.
void sweep(
    std::vector<Aabb>& boxes,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps)
{
    overlaps.clear();

    int batch_start = 0, batch_end = boxes.size();
    while (batch_start < boxes.size()) {
        try {
            batched_sweep(boxes, batch_start, batch_end, sort_axis, overlaps);
            // NOTE: This may be too aggressive.
            batch_start = batch_end;
            batch_end = boxes.size();
        } catch (...) {
            batch_end = batch_start + (batch_end - batch_start) / 2;
            if (batch_end == batch_start) {
                throw std::runtime_error(
                    "Unable to sweep boxes: out of memory!");
            } else {
                spdlog::warn(
                    "Out of memory, trying batch size: {:d}",
                    batch_end - batch_start);
            }
        }
    }
}

void sort_and_sweep(
    std::vector<Aabb>& boxes,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps)
{
    overlaps.clear();

    if (boxes.size() == 0) {
        return;
    }

    sort_along_axis(sort_axis, boxes);
    sweep(boxes, sort_axis, overlaps);
}

} // namespace scalable_ccd::stq