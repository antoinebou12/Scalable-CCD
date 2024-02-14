#include "ground_truth.hpp"

#include <catch2/catch_test_macros.hpp>

#include <scalable_ccd/utils/logger.hpp>

#include <set>
#include <fstream>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace scalable_ccd {

void compare_mathematica(
    std::vector<std::pair<int, int>> overlaps,
    const std::filesystem::path& ground_truth_file)
{
    // Dummy result list
    std::vector<int> result_list;
    result_list.resize(overlaps.size());
    fill(result_list.begin(), result_list.end(), true);

    compare_mathematica(overlaps, result_list, ground_truth_file);
}

void compare_mathematica(
    std::vector<std::pair<int, int>> overlaps,
    const std::vector<int>& result_list,
    const std::filesystem::path& ground_truth_file)
{
    // Get from file
    std::set<std::pair<long, long>> true_positives;
    {
        std::ifstream in(ground_truth_file);
        REQUIRE(in.good());
        logger().trace(
            "Comparing mathematica file {:s}", ground_truth_file.string());

        const json j = json::parse(in);

        for (auto& arr : j.get<std::vector<std::array<long, 2>>>()) {
            true_positives.emplace(arr[0], arr[1]);
        }
    }

    std::set<std::pair<long, long>> algo_broad_phase;
    for (size_t i = 0; i < overlaps.size(); i++) {
        if (result_list[i]) { // only include actual collisions
            algo_broad_phase.emplace(overlaps[i].first, overlaps[i].second);
        }
    }

    // Get intersection of true positive
    std::vector<std::pair<long, long>> algo_true_positives(
        true_positives.size());
    std::vector<std::pair<long, long>>::iterator it = std::set_intersection(
        true_positives.begin(), true_positives.end(), algo_broad_phase.begin(),
        algo_broad_phase.end(), algo_true_positives.begin());
    algo_true_positives.erase(it, algo_true_positives.end());

    // Check that overlaps contains all the ground truth collisions
    CHECK(algo_true_positives.size() == true_positives.size());
}

} // namespace scalable_ccd