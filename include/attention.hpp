/**
 * @file     attention.hpp
 * @brief   Scaled Dot-Product Attention
 * @author  Rogov Anatoliy
 * @date    2026-04-11
 *
 * @details This module provides Attention function with different matmul
 * realizations
 *
 * Usage Example:
 * @code{.cpp}
 * std::ofstream log("log.log");
 * int batch = 2, seq_q = 2, seq_k = 1, d_k = 1, d_v = 2;
 * Tensor3D<float> Q(batch, seq_q, d_k, -2.0f, 2.0f);
 * Q.dump(log);
 * Tensor3D<float> K(batch, seq_k, d_k, -2.0f, 2.0f);
 * K.dump(log);
 * Tensor3D<float> V(batch, seq_k, d_v, -2.0f, 2.0f);
 * V.dump(log);
 * Tensor3D<float> result = attention_with_matmul(Q, K, V, CACHE_OPTIMIZED);
 * result.dump(log);
 * @endcode
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
Tensor3D<T>& softmax(Tensor3D<T>& A) {
    for (int b = 0; b < A.nbatch(); b++) {
        for (int i = 0; i < A.nrows(); i++)
            softmax_sequence(&A[b][i][0], A.ncols());
    }
    return A;
}

/**
 * @brief Compute Online Softmax
 * @return softmax(A) * B;
 */
template <typename T>
Tensor3D<T> online_softmax(Tensor3D<T>& A, const Tensor3D<T>& B) {
    if (A.nbatch() != B.nbatch()) {
        throw std::invalid_argument(
            "Batch sizes must match for tensor multiplication");
    }
    if (A.ncols() != B.nrows()) {
        throw std::invalid_argument(
            "Inner dimensions must match for multiplication");
    }

    Tensor3D<T> result(A.nbatch(), A.nrows(), B.ncols());
    for (int b = 0; b < A.nbatch(); b++) {
        for (int i = 0; i < A.nrows(); i++) {
            softmax_sequence(&A[b][i][0], A.ncols());
            for (int k = 0; k < A.ncols(); k++) {
                for (int j = 0; j < B.ncols(); j++)
                    result[b][i][j] += A[b][i][k] * B[b][k][j];
            }
        }
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
                                  MatMulType matmul_type) {
    K.transpose();
    Tensor3D<float> A = tensorMul(Q, K, matmul_type);
    A *= static_cast<T>(1.0 / std::sqrt(Q.ncols()));
    A = online_softmax(A, V);

    return A;
}
