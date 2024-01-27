#pragma once

#include <vector>
#include <string>
#include <utility> // std::pair

namespace scalable_ccd {

void compare_mathematica(
    std::vector<std::pair<int, int>> overlaps, const std::string& mma_file);

void compare_mathematica(
    std::vector<std::pair<int, int>> overlaps,
    const std::vector<int>& result_list,
    const std::string& mma_file);

} // namespace scalable_ccd