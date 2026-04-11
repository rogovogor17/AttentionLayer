/**
 * @file     matmul.hpp
 * @brief   Matrix multiplication
 * @author  Rogov Anatoliy
 * @date    2026-04-11
 *
 * @details This module provides different matrix multiplication operations
 */

#include <iostream>
#include <vector>

/**
 * @class Matrix
 * @brief Main class for matrix
 *
 * @details Class constructor needs reference to vector of data and matrix
 * raws-columns.
 */
template <typename T>
class Matrix {
   private:
    size_t rows_ = 0;     ///< Amount of matrix rows
    size_t columns_ = 0;  ///< Amount of matrix columns
    T* data_;             ///< Matrix data
   public:
    Matrix() = default;

    /** @param[in] data Reference to constant vector of matrix data */
    Matrix(size_t rows, size_t columns, const std::vector<T>& data)
        : rows_(rows), columns_(columns) {
        if (rows_ * columns_ != data.size()) {
            throw std::invalid_argument(
                "Data size is not match with emerging matrix sizes");
        }

        data_ = &data;
    };

    /** @brief Print matrix */
    void print() {
        for (size_t i = 0; i < rows_; i++) {
            for (size_t j = 0; j < columns_; j++)
                std::cout << (*data_)[i * columns_ + j] << " ";
            std::cout << std::endl;
        }
    }
};

template <typename T>
void naiveMatMul(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);
