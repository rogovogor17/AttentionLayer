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
 * std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
 * Matrix<float> A(3, 3, values.begin(), values.end());
 * A.dump();
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
    using index_t = int;
    index_t rows_ = 0;     ///< Amount of matrix rows
    index_t cols_ = 0;     ///< Amount of matrix columns
    std::vector<T> data_;  ///< Matrix templated data

    /**
     * @class ProxyRow
     * @brief Class for redefinition of operator[] for laconic indexing in
     * matrix
     */
    class ProxyRow {
       private:
        T* row_;        ///< Pointer to T type row in matrix
        index_t size_;  ///< Size of row in matrix

        /**
         * @brief Returning element with position n in row
         * @throws std::out_of_range In case of going beyond the row boundary in
         * the matrix
         */
        T& at(int n) const {
            if (n < 0 || n >= size_)
                throw std::out_of_range("Invalid indexing to matrix column");
            return row_[n];
        }

       public:
        ProxyRow(T* row, index_t size) : row_(row), size_(size) {};
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
        return ProxyRow(&data_[n * cols_], cols_);
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
        : rows_(static_cast<index_t>(rows)),
          cols_(static_cast<index_t>(cols)),
          data_() {
        if (rows <= 0 || cols <= 0)
            throw std::invalid_argument("Invalid sizes to create matrix");

        if (rows_ * cols_ >= data_.max_size())
            throw std::length_error("Matrix sizes are too large");

        data_.assign(rows_ * cols_, val);
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

    ProxyRow operator[](int n) { return at_row(n); }
    const ProxyRow operator[](int n) const { return at_row(n); }

    /** @brief Get amount of Matrix columns */
    int ncols() const noexcept { return static_cast<int>(cols_); }

    /** @brief Get amount of Matrix rows */
    int nrows() const noexcept { return static_cast<int>(rows_); }

    /** @brief Dump matrix to ostream. Default behavior - dump to std::cout*/
    void dump(std::ostream& os = std::cout) {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                os << data_[i * cols_ + j] << " ";
            }
            os << std::endl;
        }
    }
};

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
