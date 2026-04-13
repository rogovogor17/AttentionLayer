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

template <typename T>
Tensor3D<T> attention_with_matmul(const Tensor3D<T>& Q, const Tensor3D<T>& K,
                                  const Tensor3D<T>& V,
                                  MatMulType matmul_type) {}
