/**
 * @file     matrix.hpp
 * @brief   Matrix multiplication
 * @author  Rogov Anatoliy
 * @date    2026-04-11
 *
 * @details This module provides class matrix and different operations with them
 *
 * Usage Example:
 * @code{.cpp}
 * std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
 * Matrix<float> A(3, 3, values.begin(), values.end());
 * A.dump();
 * @endcode
 */

#include <stddef.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

/**
 * @class Matrix
 * @brief Main class for matrix
 */
template <typename T>
class Matrix {
   private:
    using index_t = size_t;
    index_t rows_ = 0;     ///< Amount of matrix rows
    index_t cols_ = 0;     ///< Amount of matrix columns
    std::vector<T> data_;  ///< Matrix template data
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

    /** @brief Template matrix constructor with known dimensions and general
     * iterators of memory area for matrix data.
     * @throws std::length_error If the transferred sizes do not match the sizes
     * of the memory area.
     */
    template <typename It>
    Matrix(int rows, int cols, It begin, It end) : Matrix(rows, cols) {
        if (rows * cols != std::distance(begin, end)) {
            throw std::length_error(
                "Matrix sizes are not match with memory area");
        }

        data_.assign(begin, end);
    }

    /** @brief Dump matrix to ostream. Default behavior - dump to std::cout*/
    void dump(std::ostream& os = std::cout) {
        for (size_t i = 0; i < rows_; i++) {
            for (size_t j = 0; j < cols_; j++)
                os << data_[i * cols_ + j] << " ";
            os << std::endl;
        }
    }
};
