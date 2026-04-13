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
 * std::ofstream log("log.log");
 * Tensor3D<float> Q(8, 3, 3, -100.0f, 100.0f);
 * Tensor3D<float> I = Tensor3D<float>::eye(8, 3);
 * Q.dump(log);
 * Tensor3D<float> A = tensorMul(Q, I, NAIVE);
 * Tensor3D<float> B = tensorMul(Q, I, CACHE_OPTIMIZED);
 * std::cout << (A.isApprox(B) ? "Success!" : "Failed!") << std::endl;
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

    /**
     * @brief Constructor with random initialization for all matrices
     * @param min_val Minimum random value (inclusive)
     * @param max_val Maximum random value (inclusive)
     */
    Tensor3D(int batch, int rows, int cols, T min_val, T max_val)
        : batch_(batch), rows_(rows), cols_(cols) {
        if (batch <= 0 || rows <= 0 || cols <= 0)
            throw std::invalid_argument("Tensor dimensions must be positive");

        if (min_val > max_val)
            throw std::invalid_argument("min_val must be <= max_val");

        data_.reserve(static_cast<size_t>(batch));
        for (int b = 0; b < batch; b++)
            data_.emplace_back(rows, cols, min_val, max_val);
    }

    /**
     * @brief Fill entire tensor with random values
     * @param min_val Minimum random value
     * @param max_val Maximum random value
     */
    void fill_random(T min_val, T max_val) {
        for (auto& mat : data_) mat.fill_random(min_val, max_val);
    }

    /**
     * @brief Create identity tensor (each matrix is identity)
     * @param size Matrix size (rows = cols = size)
     * @return Tensor with identity matrices
     */
    static Tensor3D<T> eye(int batch, int size) {
        Tensor3D<T> tensor(batch, size, size);
        for (int b = 0; b < batch; b++) tensor[b] = Matrix<T>::eye(size);
        return tensor;
    }

    /**
     * @brief Access matrix by batch index
     * @throws std::out_of_range If index is invalid
     */
    Matrix<T>& operator[](int b) {
        if (b < 0 || b >= batch_) {
            throw std::out_of_range("Batch index out of range");
        }
        return data_[static_cast<size_t>(b)];
    }

    /**
     * @brief Const access matrix by batch index
     * @throws std::out_of_range If index is invalid
     */
    const Matrix<T>& operator[](int b) const {
        if (b < 0 || b >= batch_) {
            throw std::out_of_range("Batch index out of range");
        }
        return data_[static_cast<size_t>(b)];
    }

    /**
     * @brief Equality operator for tensors
     * @param other Tensor to compare with
     * @return true if tensors have same dimensions and all matrices equal
     */
    bool operator==(const Tensor3D<T>& other) const {
        if (batch_ != other.batch_ || rows_ != other.rows_ ||
            cols_ != other.cols_)
            return false;

        for (int b = 0; b < batch_; b++) {
            if (data_[static_cast<size_t>(b)] !=
                other.data_[static_cast<size_t>(b)])
                return false;
        }
        return true;
    }

    /** @brief Inequality operator for tensors */
    bool operator!=(const Tensor3D<T>& other) const {
        return !(*this == other);
    }

    /**
     * @brief Equality with tolerance (for floating point types)
     * @param other Tensor to compare with
     * @param eps Tolerance for comparison
     * @return true if tensors are approximately equal within tolerance
     */
    bool isApprox(const Tensor3D<T>& other,
                  T eps = Matrix<T>::default_eps()) const {
        if (batch_ != other.batch_ || rows_ != other.rows_ ||
            cols_ != other.cols_)
            return false;

        for (int b = 0; b < batch_; b++) {
            if (!data_[static_cast<size_t>(b)].isApprox(
                    other.data_[static_cast<size_t>(b)], eps))
                return false;
        }
        return true;
    }

    /** @brief Scalar multiplication */
    Tensor3D<T> operator*(T scalar) const {
        Tensor3D<T> result(batch_, rows_, cols_);
        for (int b = 0; b < batch_; b++) result[b] = data_[b] * scalar;
        return result;
    }

    /** @brief Scalar multiplication and assign*/
    Tensor3D<T>& operator*=(T scalar) {
        for (size_t i = 0; i < data_.size(); i++) data_[i] *= scalar;
        return *this;
    }

    /** @brief Access individual element */
    T& at(int b, int i, int j) { return (*this)[b][i][j]; }

    /** @brief Const access individual element */
    const T& at(int b, int i, int j) const { return (*this)[b][i][j]; }

    /**
     * @brief Transpose tensor in-place
     * @return Reference to this tensor
     */
    Tensor3D<T>& transpose() & {
        for (int b = 0; b < batch_; b++)
            data_[static_cast<size_t>(b)].transpose();
        std::swap(rows_, cols_);
        return *this;
    }

    /**
     * @brief Transpose for const objects
     * @return New transposed tensor
     */
    Tensor3D<T> transpose() const& {
        Tensor3D<T> result = *this;
        result.transpose();
        return result;
    }

    /** @brief Get amount of Matrix in Tensor3D */
    int nbatch() const noexcept { return batch_; }

    /** @brief Get amount of each Matrix rows in Tensor3D */
    int nrows() const noexcept { return rows_; }

    /** @brief Get amount of each Matrix columns in Tensor3D*/
    int ncols() const noexcept { return cols_; }

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

/**
 * @brief Matrix multiplication along last two dimensions
 * @throws std::invalid_argument If batch sizes mismatch or inner dimensions
 * mismatch
 */
template <typename T>
Tensor3D<T> tensorMul(const Tensor3D<T>& A, const Tensor3D<T>& B,
                      MatMulType type = CACHE_OPTIMIZED) {
    if (A.nbatch() != B.nbatch()) {
        throw std::invalid_argument(
            "Batch sizes must match for tensor multiplication");
    }
    if (A.ncols() != B.nrows()) {
        throw std::invalid_argument(
            "Inner dimensions must match for multiplication");
    }

    Tensor3D<T> result(A.nbatch(), A.nrows(), B.ncols());
    std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)> matmul =
        getMatmul<T>(type);
    for (int b = 0; b < A.nbatch(); b++) result[b] = matmul(A[b], B[b]);
    return result;
}
