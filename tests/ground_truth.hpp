#pragma once

#include <vector>
#include <utility> // std::pair
#include <filesystem>

namespace scalable_ccd {

void compare_mathematica(
    std::vector<std::pair<int, int>> overlaps,
    const std::filesystem::path& ground_truth_file);

void compare_mathematica(
    std::vector<std::pair<int, int>> overlaps,
    const std::vector<int>& result_list,
    const std::filesystem::path& ground_truth_file);

} // namespace scalable_ccd