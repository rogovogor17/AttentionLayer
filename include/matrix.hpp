/**
 * @file     matrix.hpp
 * @brief   Matrix<T>
 * @author  Rogov Anatoliy
 * @date    2026-04-11
 *
 * @details This module provides class Matrix and different operations with them
 *
 * Usage Example:
 * @code{.cpp}
 * Matrix<float> A = Matrix<float>::eye(3);
 * Matrix<float> B(3, 3, 5);
 * Matrix<float> C = naive_multiply(A, B);
 * C[1][1] += C[0][0] * C[2][2];
 * C.dump();
 * @endcode
 */

#pragma once

#include <stddef.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

/**
 * @class Matrix
 * @brief Main class for matrix
 * @tparam T Type of elements that matrix consists of
 */
template <typename T>
class Matrix {
   private:
    int rows_ = 0;         ///< Amount of matrix rows
    int cols_ = 0;         ///< Amount of matrix columns
    std::vector<T> data_;  ///< Matrix templated data

    /**
     * @class ProxyRow
     * @brief Class for redefinition of operator[] for laconic indexing in
     * matrix
     */
    class ProxyRow {
       private:
        T* row_;    ///< Pointer to T type row in matrix
        int size_;  ///< Size of row in matrix

        /**
         * @brief Check position n of element in row
         * @throws std::out_of_range In case of going beyond the row boundary in
         * the matrix
         */
        T& at(int n) const {
            if (n < 0 || n >= size_)
                throw std::out_of_range("Invalid indexing to matrix column");
            return row_[n];
        }

       public:
        ProxyRow(T* row, int size) : row_(row), size_(size) {};
        const T& operator[](int n) const { return at(n); }
        T& operator[](int n) { return at(n); }
    };

    /**
     * @brief Returning row number n in matrix
     * @throws std::out_of_range In case of access to an invalid row in matrix
     */
    ProxyRow at_row(int n) const {
        if (n < 0 || n >= rows_)
            throw std::out_of_range("Invalid indexing to matrix row");
        return ProxyRow(const_cast<T*>(&data_[static_cast<size_t>(n * cols_)]),
                        cols_);
    }

   public:
    Matrix() = default;

    /**
     * @brief Main Matrix constructor with known dimensions and optional init
     * value.
     * @throws std::invalid_argument In the case of negative dimensions
     * @throws std::length_error In the case of exceeding the maximum number of
     * matrix elements.
     */
    Matrix(int rows, int cols, T val = T{})
        : rows_(rows), cols_(cols), data_() {
        if (rows <= 0 || cols <= 0)
            throw std::invalid_argument("Invalid sizes to create matrix");

        if (static_cast<size_t>(rows_ * cols_) >= data_.max_size())
            throw std::length_error("Matrix sizes are too large");

        data_.assign(static_cast<size_t>(rows_ * cols_), val);
    }

    /**
     * @brief Template matrix constructor with known dimensions and general
     * iterators of memory area for matrix data.
     * @details Memory area with input iterators will be moved to matrix
     * data
     * @tparam It Type of general iterator of memory area
     * @throws std::length_error If the transferred sizes do not match the
     * sizes of the memory area.
     */
    template <typename It>
    Matrix(int rows, int cols, It begin, It end) : Matrix(rows, cols) {
        if (rows * cols != std::distance(begin, end)) {
            throw std::length_error(
                "Matrix sizes are not match with memory area");
        }

        data_.assign(std::make_move_iterator(begin),
                     std::make_move_iterator(end));
    }

    /**
     * @brief Static constructor of Identity Matrix
     * @details Construct Identity Matrix with T{1} value
     */
    static Matrix<T> eye(int n) {
        Matrix<T> I(n, n);
        for (int i = 0; i < n; i++) I[i][i] = T{1};
        return I;
    }

    ProxyRow operator[](int n) { return at_row(n); }
    const ProxyRow operator[](int n) const { return at_row(n); }

    /** @brief Get amount of Matrix columns */
    int ncols() const noexcept { return static_cast<int>(cols_); }

    /** @brief Get amount of Matrix rows */
    int nrows() const noexcept { return static_cast<int>(rows_); }

    /** @brief Transpose Matrix */
    Matrix& transpose() & {
        std::vector<T> cp_data(data_);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                data_[static_cast<size_t>(j * rows_ + i)] =
                    cp_data[static_cast<size_t>(i * cols_ + j)];
            }
        }
        std::swap(rows_, cols_);
        return *this;
    }

    /** @brief Transpose constant Matrix */
    Matrix transpose() const& {
        Matrix transposeMatrix = *this;
        transposeMatrix.transpose();
        return transposeMatrix;
    }

    /** @brief Dump matrix to ostream. Default behavior - dump to
       std::cout*/
    void dump(std::ostream& os = std::cout) const {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                os << data_[static_cast<size_t>(i * cols_ + j)] << " ";
            }
            os << std::endl;
        }
    }
};

/**
 * @brief Performs naive O(n³) matrix multiplication.
 *
 * @tparam T Matrix element type (must support += and * operators)
 * @param A Left matrix (m × k)
 * @param B Right matrix (k × n)
 * @return Matrix<T> Result matrix C (m × n) where C[i][j] = Σ A[i][k] * B[k][j]
 *
 * @throws std::invalid_argument If A.ncols() != B.nrows()
 */
template <typename T>
Matrix<T> naive_multiply(const Matrix<T>& A, const Matrix<T>& B) {
    if (A.ncols() != B.nrows())
        throw std::invalid_argument("Invalid matrixes for multiplying");
    Matrix<T> C(A.nrows(), B.ncols());

    for (int i = 0; i < C.nrows(); i++) {
        for (int j = 0; j < C.ncols(); j++) {
            for (int k = 0; k < A.ncols(); k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

/**
 * @brief Performs cached O(n³) matrix multiplication.
 *
 * @tparam T Matrix element type (must support += and * operators)
 * @param A Left matrix (m × k)
 * @param B Right matrix (k × n)
 * @param bs Size of block that matrix is divided
 * @return Matrix<T> Result matrix C (m × n)
 *
 * @throws std::invalid_argument If A.ncols() != B.nrows()
 */
template <typename T>
Matrix<T> cached_multiply(const Matrix<T>& A, const Matrix<T>& B, int bs = 32) {
    if (A.ncols() != B.nrows())
        throw std::invalid_argument("Invalid matrix sizes");

    int M = A.nrows(), N = B.ncols(), K = A.ncols();
    Matrix<T> C(M, N);
    Matrix<T> Bt = B.transpose();

    for (int ii = 0; ii < M; ii += bs) {
        for (int jj = 0; jj < N; jj += bs) {
            for (int kk = 0; kk < K; kk += bs) {
                for (int i = ii; i < std::min(ii + bs, M); i++) {
                    for (int k = kk; k < std::min(kk + bs, K); k++) {
                        T A_ik = A[i][k];
                        for (int j = jj; j < std::min(jj + bs, N); j++)
                            C[i][j] += A_ik * Bt[j][k];
                    }
                }
            }
        }
    }

    return C;
}
