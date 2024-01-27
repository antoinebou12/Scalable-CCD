#pragma once

#include <Eigen/Dense>

namespace scalable_ccd::stq::gpu {

void nipalsPCA(Eigen::MatrixXd& vertices_t0, Eigen::MatrixXd& vertices_t1);

} // namespace scalable_ccd::stq::gpu