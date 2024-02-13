#pragma once

#include <thrust/device_vector.h>
#include <Eigen/Core>

namespace scalable_ccd::cuda {

/// @brief A matrix object stored on the device.
/// @tparam T the type of the elements in the matrix.
template <typename T> class DeviceMatrix {
public:
    DeviceMatrix() = default;

    DeviceMatrix(const size_t rows, const size_t cols)
        : m_rows(rows)
        , m_cols(cols)
        , m_data(rows * cols)
    {
    }

    DeviceMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat)
        : m_rows(mat.rows())
        , m_cols(mat.cols())
        , m_data(mat.size())
    {
        thrust::copy(mat.data(), mat.data() + mat.size(), m_data.begin());
    }

    void operator=(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat)
    {
        m_rows = mat.rows();
        m_cols = mat.cols();
        m_data.resize(mat.size());
        thrust::copy(mat.data(), mat.data() + mat.size(), m_data.begin());
    }

#ifndef SCALABLE_CCD_USE_DOUBLE
    DeviceMatrix(const Eigen::MatrixXd& mat) : DeviceMatrix(mat.cast<T>()) { }

    void operator=(const Eigen::MatrixXd& mat)
    {
        return operator=(mat.cast<T>());
    }
#endif

    // TODO: Add function to retrieve the matrix from the device.

    const T* data() const { return thrust::raw_pointer_cast(m_data.data()); }
    size_t size() const { return m_data.size(); }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }

private:
    size_t m_rows = 0;
    size_t m_cols = 0;
    thrust::device_vector<T> m_data;
};

} // namespace scalable_ccd::cuda