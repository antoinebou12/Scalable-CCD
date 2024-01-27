#pragma once

#include <Eigen/Dense>

namespace stq::gpu {

void nipalsPCA(Eigen::MatrixXd& vertices_t0, Eigen::MatrixXd& vertices_t1);

} // namespace stq::gpu