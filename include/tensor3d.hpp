/**
 * @file     tensor3d.hpp
 * @brief   Tensor3D<T> - 3D tensor (batch of matrices)
 * @author  Rogov Anatoliy
 * @date    2026-04-13
 *
 * @details Provides 3D tensor operations for attention mechanism
 *
 * Usage Example:
 * @code{.cpp}
 * Tensor3D<float> Q(32, 8, 64);
 * Tensor3D<float> K(32, 16, 64);
 * Tensor3D<float> V(32, 16, 32);
 * std::ofstream log("log.log");
 * Q.dump(log);
 * @endcode
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "matrix.hpp"

/**
 * @class Tensor3D
 * @brief 3-dimensional tensor represented as a batch of matrices
 * @tparam T Type of tensor elements
 */
template <typename T>
class Tensor3D {
   private:
    int batch_ = 0;                ///< Number of matrices in batch
    int rows_ = 0;                 ///< Number of rows in each matrix
    int cols_ = 0;                 ///< Number of columns in each matrix
    std::vector<Matrix<T>> data_;  ///< Batch of matrices

   public:
    Tensor3D() = default;

    /**
     * @brief Main constructor with dimensions
     * @param val Initial value for all elements
     * @throws std::invalid_argument If any dimension <= 0
     */
    Tensor3D(int batch, int rows, int cols, T val = T{})
        : batch_(batch), rows_(rows), cols_(cols) {
        if (batch <= 0 || rows <= 0 || cols <= 0)
            throw std::invalid_argument("Tensor dimensions must be positive");

        data_.reserve(static_cast<size_t>(batch));
        for (int b = 0; b < batch; b++) {
            data_.emplace_back(rows, cols, val);
        }
    }

    /**
     * @brief Constructor from existing matrices
     * @param matrices Vector of matrices
     * @throws std::invalid_argument If matrices have different sizes or empty
     */
    Tensor3D(const std::vector<Matrix<T>>& matrices) {
        if (matrices.empty()) {
            throw std::invalid_argument(
                "Cannot create tensor from empty vector");
        }

        batch_ = static_cast<int>(matrices.size());
        rows_ = matrices[0].nrows();
        cols_ = matrices[0].ncols();

        for (const auto& mat : matrices) {
            if (mat.nrows() != rows_ || mat.ncols() != cols_) {
                throw std::invalid_argument(
                    "All matrices must have same dimensions");
            }
        }

        data_ = matrices;
    }

    /** @brief Get amount of Matrix in Tensor3D */
    int batch() const noexcept { return batch_; }

    /** @brief Get amount of each Matrix rows in Tensor3D */
    int rows() const noexcept { return rows_; }

    /** @brief Get amount of each Matrix columns in Tensor3D*/
    int cols() const noexcept { return cols_; }

    /** @brief Dump tensor to ostream */
    void dump(std::ostream& os = std::cout) const {
        os << "Tensor3D [" << batch_ << " x " << rows_ << " x " << cols_ << "]"
           << std::endl;
        for (int b = 0; b < batch_; b++) {
            os << "Batch " << b << ":" << std::endl;
            data_[static_cast<size_t>(b)].dump(os);
            if (b < batch_ - 1) os << std::endl;
        }
    }
};
