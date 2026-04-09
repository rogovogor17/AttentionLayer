/**
 * @file     tensor.hpp
 * @brief   Tensors and base operations with them
 * @author  Rogov Anatoliy
 * @date    2026-04-9
 *
 * @details This module provide class Tensor for base
 *			operations with tensors.
 *
 * Usage Example:
 * @code{.cpp}
 * Tensor tnr;
 * @endcode
 */

#pragma once

#include <cstddef>

/**
 * @class Tensor
 * @brief Main class for tensor
 *
 * Provided operations:
 */
class Tensor {
   private:
    size_t rang;  ///< Rang or dimension of tensor

    Tensor add(const Tensor& term) const {
        // Check dimensions
        Tensor result;
        // Addition
        return result;
    }

   public:
    size_t getRang() { return rang; }

    Tensor operator+(const Tensor& term) const { return add(term); }

    Tensor& operator+=(const Tensor& term) {
        // Addition with 'this' changes
        return *this;
    }
};
