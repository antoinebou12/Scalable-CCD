#pragma once

#include <Eigen/Dense>

namespace scalable_ccd {

void nipals_pca(Eigen::MatrixXd& vertices_t0, Eigen::MatrixXd& vertices_t1);

} // namespace scalable_ccd