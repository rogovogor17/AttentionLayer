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
template <typename T, typename It>
void softmax_sequence(It begin, It end) {
    if (std::distance(begin, end) <= 0)
        throw std::invalid_argument("Iterators of object sequence incorrect");

    T sum = 0.0;
    for (It it = begin; it != end; it++) {
        *it = std::exp(*it);
        sum += *it;
    }

    for (It it = begin; it != end; it++) *it /= sum;
}

/**
 * @brief Compute Softmax function on input Tensor3D
 * @return Result of Softmax operation Tensor3D
 */
template <typename T>
Tensor3D<T> softmax(const Tensor3D<T>& A) {
    Tensor3D<T> result = A;
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
