/**
 * @file     tensor.hpp
 * @brief   Tensors and base operations with them
 * @author  Rogov Anatoliy
 * @date    2026-04-9
 *
 * @details This module provides class Tensor for base operations with tensors.
 *
 * Usage Example:
 * @code{.cpp}
 * Tensor tnr;
 * @endcode
 */

#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

/**
 * @class   Shape
 * @brief   Class for shape info about tensor
 * @details Construct shape from dimension vector or from lists of each
 * 			dimension value.
 */
class Shape {
   private:
    std::vector<size_t> dims_;  ///< Shape sizes of dimensions
    size_t rang_ = 0;           ///< Shape amount of dimensions
    size_t elemCount_ = 1;      ///< Shape overall elements count

   public:
    Shape() = default;

    /** @param[in] dims Vector of dimensions sizes */
    Shape(std::vector<size_t> dims)
        : dims_(std::move(dims)), rang_(dims_.size()) {
        for (auto dim : dims_) elemCount_ *= dim;
    }

    /** @param[in] dim_values Initializer list of dimensions sizes */
    Shape(std::initializer_list<size_t> dim_values)
        : dims_(dim_values), rang_(dims_.size()) {
        for (auto dim : dims_) elemCount_ *= dim;
    }

    /** @brief Get current shape rang */
    size_t getRang() const { return rang_; }

    /** @brief Get current shape elements count */
    size_t getCount() const { return elemCount_; }

    /** @brief Print rang and size of each dimension from container */
    void print() const {
        std::cout << "rang: " << rang_ << std::endl;
        std::cout << "dims: [ ";
        for (auto dim : dims_) {
            std::cout << dim << " ";
        }
        std::cout << "]" << std::endl;
        std::cout << "count: " << elemCount_ << std::endl;
    }
};

/**
 * @class   Data
 * @brief   Class for data inside tensor
 * @details Template class for work with data inside tensor
 */
template <typename T>
class Data {
   private:
    std::vector<T> data_;

   public:
    Data() = default;

    /** @param[in] size Size of data_ */
    Data(size_t size) : data_(size) {}

    /** @brief Get current size of data */
    size_t size() const { return data_.size(); }

    /** @brief Print data_ in special order */
    void print() const {
        std::cout << "data: { ";
        for (auto elem : data_) {
            std::cout << elem << "; ";
        }
        std::cout << "}" << std::endl;
    }
};

/**
 * @class Tensor
 * @brief Main class for tensor
 *
 * @details Class constructor needs tensor shape dimensions sizes.
 * Provided operations: add
 */
template <typename T>
class Tensor {
   private:
    Shape shape_;   ///< Tensor shape information
    Data<T> data_;  ///< Tensor data

    Tensor add(const Tensor& term) const {
        // Check dimensions
        Tensor result;
        // Addition
        return result;
    }

   public:
    Tensor() = default;
    Tensor(const Shape& shape) : shape_(shape), data_(shape_.getCount()) {}

    Tensor operator+(const Tensor& term) const { return add(term); }
    Tensor& operator+=(const Tensor& term) {
        // Addition with 'this' changes
        return *this;
    }

    /** @brief Set new shape information about tensor */
    void setShape(const Shape& shape) { shape_ = shape; }

    /** @brief Print tensor shape and data information */
    void print() {
        shape_.print();
        data_.print();
    }
};
