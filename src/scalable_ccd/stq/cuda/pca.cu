#include <iostream>
#include <Eigen/Dense>
#include <scalable_ccd/stq/cuda/pca.cuh>

namespace stq::gpu {

void nipalsPCA(Eigen::MatrixXd& vertices_t0, Eigen::MatrixXd& vertices_t1)
{
    int n_components = vertices_t0.cols();

    Eigen::MatrixXd X(vertices_t0.rows() + vertices_t1.rows(), n_components);
    X.block(0, 0, vertices_t0.rows(), n_components) = vertices_t0;
    X.block(vertices_t0.rows(), 0, vertices_t1.rows(), n_components) =
        vertices_t1;

    // NIPALS algorithm for PCA

    // m is the origin of the plane
    Eigen::VectorXd m = X.colwise().mean();
    // y is the vectors from the centriod yi=xi-m
    Eigen::MatrixXd vertices_3d = X.rowwise() - m.transpose();

    // Compute the covariance matrix
    Eigen::MatrixXd cov =
        (vertices_3d.adjoint() * vertices_3d) / double(vertices_3d.rows() - 1);

    // Compute the eigenvectors and eigenvalues of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov);
    if (eigensolver.info() != Eigen::Success)
        abort();

    // Get the eigenvectors and eigenvalues
    Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();

    // Sort the eigenvectors and eigenvalues in descending order
    int ncomp = 3;
    Eigen::MatrixXd sorted_eigenvectors =
        eigenvectors.rowwise().reverse().block(
            0, 0, eigenvectors.rows(), ncomp);

    // Transform the original data to the new coordinate system
    Eigen::MatrixXd transformed_vertices = vertices_3d * sorted_eigenvectors;

    vertices_t0 =
        transformed_vertices.block(0, 0, vertices_t0.rows(), n_components);
    vertices_t1 = transformed_vertices.block(
        vertices_t0.rows(), 0, vertices_t1.rows(), n_components);
}
} // namespace stq::gpu
