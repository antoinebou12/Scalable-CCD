#include "sort_and_sweep.hpp"

#include <scalable_ccd/utils/merge_local_overlaps.hpp>
#include <scalable_ccd/utils/logger.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <algorithm> // std::sort
#include <vector>    // std::vector

namespace scalable_ccd {

namespace {
    inline int flip_id(const int id) { return -id - 1; }

    /// @brief Determine if two boxes are overlapping.
    /// @param a The first box.
    /// @param b The second box.
    /// @param skip_axis The axis to skip.
    /// @return True if the boxes are overlapping, false otherwise.
    bool are_overlapping(const AABB& a, const AABB& b)
    {
        return (a.min <= b.max).all() && (b.min <= a.max).all();
    }

    /// @brief Determine if two elements share a vertex.
    /// @param a Vertex ids of the first element.
    /// @param b Vertex ids of the second element.
    /// @return True if the elements share a vertex, false otherwise.
    bool
    share_a_vertex(const std::array<long, 3>& a, const std::array<long, 3>& b)
    {
        return a[0] == b[0] || a[0] == b[1] || a[0] == b[2] //
            || a[1] == b[0] || a[1] == b[1] || a[1] == b[2] //
            || a[2] == b[0] || a[2] == b[1] || a[2] == b[2];
    }

    // https://stackoverflow.com/questions/3909272/sorting-two-corresponding-arrays
    class SortIndices {
    private:
        const std::vector<AABB>& boxes;
        const int axis;

    public:
        SortIndices(const std::vector<AABB>& _boxes, const int _axis)
            : boxes(_boxes)
            , axis(_axis)
        {
        }

        bool operator()(int i, int j) const
        {
            return boxes[i].min[axis] < boxes[j].min[axis];
        }
    };

    class SortBoxes {
    private:
        const int axis;

    public:
        SortBoxes(const int _axis) : axis(_axis) { }

        bool operator()(const AABB& a, const AABB& b) const
        {
            return a.min[axis] < b.min[axis];
        }
    };

    template <bool is_two_lists>
    inline bool is_valid_pair(const long id_a, const long id_b)
    {
        if constexpr (is_two_lists) {
            return (id_a >= 0 && id_b < 0) || (id_a < 0 && id_b >= 0);
        } else {
            return true;
        }
    }

    /// @brief Sweep the boxes along the given axis (without memory check).
    /// @param[in] boxes Boxes to sweep.
    /// @param[in] n Number of boxes.
    /// @param[in,out] sort_axis Axis to sweep along and the next axis to sort along.
    /// @param[out] overlaps Overlaps to populate.
    template <bool is_two_lists>
    void batched_sweep(
        const std::vector<AABB>& boxes,
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
                    const AABB& a = boxes[i];

                    for (long j = i + 1; j < boxes.size(); j++) {
                        const AABB& b = boxes[j];

                        if (a.max[sort_axis] < b.min[sort_axis]) {
                            break;
                        }

                        if (is_valid_pair<is_two_lists>(a.id, b.id)
                            && are_overlapping(a, b)
                            && !share_a_vertex(a.vertex_ids, b.vertex_ids)) {
                            if constexpr (is_two_lists) {
                                // Negative IDs are from the first list
                                local_overlaps.emplace_back(
                                    a.id < 0 ? flip_id(a.id) : flip_id(b.id),
                                    a.id < 0 ? b.id : a.id);
                            } else {
                                assert(a.id >= 0 && b.id >= 0);
                                local_overlaps.emplace_back(
                                    std::min(a.id, b.id), std::max(a.id, b.id));
                            }
                        }
                    }
                }
            });

        merge_local_overlaps(storages, overlaps);
    }
} // namespace

void sort_along_axis(const int axis, std::vector<AABB>& boxes)
{
    // const std::vector<AABB> orig_boxes = boxes;

    // std::vector<int> box_indices(boxes.size());
    // std::iota(box_indices.begin(), box_indices.end(), 0);
    // tbb::parallel_sort(
    //     box_indices.begin(), box_indices.end(), SortIndices(boxes, axis));

    // for (size_t i = 0; i < boxes.size(); i++) {
    //     boxes[i] = orig_boxes[box_indices[i]];
    // }

    tbb::parallel_sort(boxes.begin(), boxes.end(), SortBoxes(axis));
}

// Adaptively split the boxes if the memory limit is exceeded.
template <bool is_two_lists>
void sweep(
    std::vector<AABB>& boxes,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps)
{
    assert(std::is_sorted(boxes.begin(), boxes.end(), SortBoxes(sort_axis)));

    overlaps.clear();

    int batch_start = 0, batch_end = boxes.size();
    while (batch_start < boxes.size()) {
        try {
            batched_sweep<is_two_lists>(
                boxes, batch_start, batch_end, sort_axis, overlaps);
            // NOTE: This may be too aggressive.
            batch_start = batch_end;
            batch_end = boxes.size();
        } catch (...) {
            batch_end = batch_start + (batch_end - batch_start) / 2;
            if (batch_end == batch_start) {
                throw std::runtime_error(
                    "Unable to sweep boxes: out of memory!");
            } else {
                logger().warn(
                    "Out of memory, trying batch size: {:d}",
                    batch_end - batch_start);
            }
        }
    }

    // Compute the variance of the centers of the boxes.
    ArrayMax3 sum_centers = ArrayMax3::Zero(boxes[0].min.size());
    ArrayMax3 sum_centers_sqr = ArrayMax3::Zero(boxes[0].min.size());
    for (const AABB& box : boxes) {
        const ArrayMax3 center = (box.min + box.max) / 2;
        sum_centers += center;
        sum_centers_sqr += center.square();
    }

    const ArrayMax3 variance =
        sum_centers_sqr - sum_centers.square() / boxes.size();

    // Determine the next axis to sort along based on the variance.
    sort_axis = 0;
    if (variance[1] > variance[0]) {
        sort_axis = 1;
    }
    if (variance.size() == 3 && variance[2] > variance[sort_axis]) {
        sort_axis = 2;
    }
}

void sort_and_sweep(
    std::vector<AABB> boxes,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps)
{
    overlaps.clear();

    if (boxes.size() == 0) {
        return;
    }

    sort_along_axis(sort_axis, boxes);
    sweep</*is_two_lists=*/false>(boxes, sort_axis, overlaps);
}

void sort_and_sweep(
    std::vector<AABB> boxesA,
    std::vector<AABB> boxesB,
    int& sort_axis,
    std::vector<std::pair<int, int>>& overlaps)
{
    overlaps.clear();

    if (boxesA.size() == 0 || boxesB.size() == 0) {
        return;
    }

    sort_along_axis(sort_axis, boxesA);
    sort_along_axis(sort_axis, boxesB);

    // Merge the two lists of boxes (giving the boxA IDs negative values).
    std::vector<AABB> boxes(boxesA.size() + boxesB.size());
    for (size_t i = 0, j = 0, k = 0; k < boxes.size(); k++) {
        if (i < boxesA.size() && j < boxesB.size()) {
            if (boxesA[i].min[sort_axis] < boxesB[j].min[sort_axis]) {
                boxes[k] = boxesA[i];
                boxes[k].id = flip_id(boxesA[i].id); // ∈ [-n, -1]
                i++;
            } else {
                boxes[k] = boxesB[j];
                j++;
            }
        } else if (i < boxesA.size()) {
            boxes[k] = boxesA[i];
            boxes[k].id = flip_id(boxesA[i].id); // ∈ [-n, -1]
            i++;
        } else {
            assert(j < boxesB.size());
            boxes[k] = boxesB[j];
            j++;
        }
    }

    sweep</*is_two_lists=*/true>(boxes, sort_axis, overlaps);
}

// ---------------------------------------------------------------------------
// Explicit template instantiation
template void
sweep<false>(std::vector<AABB>&, int&, std::vector<std::pair<int, int>>&);
template void
sweep<true>(std::vector<AABB>&, int&, std::vector<std::pair<int, int>>&);

} // namespace scalable_ccd