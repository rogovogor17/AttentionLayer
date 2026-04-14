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

    T sum = 0;
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
    Tensor3D<T> A = tensorMul(Q, K, matmul_type);
    A *= static_cast<T>(1.0 / std::sqrt(Q.ncols()));
    return tensorMul(softmax(A), V, matmul_type);
}

/**
 * @brief Compute Scaled Dot-Product Attention.
 * @details Attention without intermediate Tensor
 * instead of softmax
 */
template <typename T>
Tensor3D<T> attention_online(const Tensor3D<T>& Q, const Tensor3D<T>& K,
                             const Tensor3D<T>& V) {
    int batch = Q.nbatch();
    int seq_q = Q.nrows();
    int seq_k = K.nrows();
    int d_k = Q.ncols();
    int d_v = V.ncols();

    Tensor3D<T> result(batch, seq_q, d_v);
    T scale = static_cast<T>(1.0 / std::sqrt(d_k));

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < seq_q; i++) {
            T sum = T{};
            std::vector<T> row(d_v, T{});
            const T* q_row = &Q[b][i][0];
            for (int k = 0; k < seq_k; k++) {
                T score = 0;
                const T* k_row = &K[b][k][0];

                for (int j = 0; j < d_k; j++) score += q_row[j] * k_row[j];

                score = std::exp(score * scale);
                sum += score;
                const T* v_row = &V[b][k][0];

                for (int j = 0; j < d_v; j++) row[j] += score * v_row[j];
            }

            T* result_row = &result[b][i][0];
            for (int j = 0; j < d_v; j++) result_row[j] = row[j] / sum;
        }
    }

    return result;
}
