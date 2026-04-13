/**
 * @file     attention.hpp
 * @brief   Scaled Dot-Product Attention
 * @author  Rogov Anatoliy
 * @date    2026-04-11
 *
 * @details This module provides Attention function with different matmul
 * realizations
 */

#pragma once

#include "tensor3d.hpp"

/**
 * @brief Compute Softmax function with area iterators
 * @return Result of sequence softmax operation: probability distribution
 */
template <typename T>
void softmax_sequence(T* data, int size) {
    if (size <= 0)
        throw std::invalid_argument("Iterators of object sequence incorrect");

    T sum = 0.0;
    for (int i = 0; i < size; i++) {
        data[i] = std::exp(data[i]);
        sum += data[i];
    }

    for (int i = 0; i < size; i++) data[i] /= sum;
}

/**
 * @brief Compute Softmax function on input Tensor3D
 * @return Result of Softmax operation Tensor3D
 */
template <typename T>
Tensor3D<T> softmax(const Tensor3D<T>& A) {
    Tensor3D<T> result = A;
    for (int b = 0; b < result.nbatch(); b++) {
        for (int i = 0; i < result.nrows(); i++)
            softmax_sequence(&result[b][i][0], result.ncols());
    }
    return result;
}

/**
 * @brief Compute Scaled Dot-Product Attention.
 *
 * Mathematical expression:
 * \f[ \text{Attention}(Q, K, V) =
 * \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \f]
 *
 * @param Q Tensor3D Queries.
 * @param K Tensor3D Keys.
 * @param V Tensor3D Values.
 * @return Result of using Attention mechanism.
 */
template <typename T>
Tensor3D<T> attention_with_matmul(const Tensor3D<T>& Q, const Tensor3D<T>& K,
                                  const Tensor3D<T>& V,
                                  MatMulType matmul_type) {}
