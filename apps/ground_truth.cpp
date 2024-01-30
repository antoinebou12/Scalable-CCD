#include "ground_truth.hpp"

#include <scalable_ccd/utils/logger.hpp>

#include <set>
#include <fstream>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace scalable_ccd {

// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
unsigned long cantor(unsigned long x, unsigned long y)
{
    return (x + y) * (x + y + 1) / 2 + y;
}

struct cmp {
    bool operator()(std::pair<long, long>& a, std::pair<long, long>& b) const
    {
        return a.first == b.first && a.second == b.second;
    };
};

void compare_mathematica(
    std::vector<std::pair<int, int>> overlaps, const std::string& json_path)
{
    std::vector<int> result_list;
    result_list.resize(overlaps.size());
    fill(result_list.begin(), result_list.end(), true);
    compare_mathematica(overlaps, result_list, json_path);
}

void compare_mathematica(
    std::vector<std::pair<int, int>> overlaps,
    const std::vector<int>& result_list,
    const std::string& json_path)
{
    // Get from file
    std::ifstream in(json_path);
    if (in.fail()) {
        logger().trace("{:s} does not exist", json_path);
        return;
    } else {
        logger().trace("Comparing mathematica file {:s}", json_path);
    }

    json j_vec = json::parse(in);

    std::set<std::pair<long, long>> true_positives;
    for (auto& arr : j_vec.get<std::vector<std::array<long, 2>>>()) {
        true_positives.emplace(arr[0], arr[1]);
    }

    std::set<std::pair<long, long>> algo_broad_phase;
    for (size_t i = 0; i < overlaps.size(); i += 1) {
        if (result_list[i]) {
            algo_broad_phase.emplace(overlaps[i].first, overlaps[i].second);
        }
    }

    // Get intersection of true positive
    std::vector<std::pair<long, long>> algo_true_positives(
        true_positives.size());
    std::vector<std::pair<long, long>>::iterator it = std::set_intersection(
        true_positives.begin(), true_positives.end(), algo_broad_phase.begin(),
        algo_broad_phase.end(), algo_true_positives.begin());
    algo_true_positives.resize(it - algo_true_positives.begin());

    logger().trace(
        "Contains {:d}/{:d} TP", algo_true_positives.size(),
        true_positives.size());
    return;
}

} // namespace scalable_ccd