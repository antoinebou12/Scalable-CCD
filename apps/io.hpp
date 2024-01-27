#pragma once

#include <scalable_ccd/stq/aabb.hpp>

#include <vector>
#include <string>

#include <Eigen/Core>

namespace scalable_ccd {

void parse_mesh(
    const std::string& file_t0,
    const std::string& file_t1,
    Eigen::MatrixXd& V0,
    Eigen::MatrixXd& V1,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& E);

void parse_mesh(
    const std::string& file_t0,
    const std::string& file_t1,
    std::vector<stq::Aabb>& boxes);

} // namespace scalable_ccd