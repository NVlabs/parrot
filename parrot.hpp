/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PARROT_HPP
#define PARROT_HPP

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_reference.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <ctime>

#include <cuda/std/iterator>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include "thrustx.hpp"

namespace parrot {

namespace literals {
// User-defined literal for integral_constant to avoid template syntax
// Allows writing 2_ic instead of std::integral_constant<int, 2>{}
template <char... Digits>
constexpr auto operator""_ic() {
    constexpr int value = []() {
        int result = 0;
        ((result = result * 10 + (Digits - '0')), ...);
        return result;
    }();
    return std::integral_constant<int, value>{};
}

}  // namespace literals

// Type trait to extract underlying type from device_reference
template <typename T>
struct extract_value_type {
    using type = T;
};

template <typename T>
struct extract_value_type<thrust::device_reference<T>> {
    using type = T;
};

template <typename T>
using extract_value_type_t = typename extract_value_type<T>::type;

// Sentinel type to indicate no mask
struct no_mask_t {};

// Forward declaration of fusion_array
template <typename Iterator, typename MaskIterator = no_mask_t>
class fusion_array;

// Now define the trait since fusion_array is forward declared
template <typename T>
struct is_fusion_array : std::false_type {};

// Specialization for fusion_array (both with and without mask)
template <typename Iterator>
struct is_fusion_array<fusion_array<Iterator, no_mask_t>> : std::true_type {};

template <typename Iterator, typename MaskIterator>
struct is_fusion_array<fusion_array<Iterator, MaskIterator>> : std::true_type {
};

// Helper variable template
template <typename T>
inline constexpr bool is_fusion_array_v = is_fusion_array<T>::value;

// Forward declare the range function
inline auto range(int end) -> fusion_array<thrust::counting_iterator<int>>;

// Forward declare the stats namespace
namespace stats {
template <typename Iterator, typename MaskIterator>
auto norm_cdf(const fusion_array<Iterator, MaskIterator> &arr);
}

// Type trait to check if a type is a thrust::pair
template <typename T>
struct is_thrust_pair : std::false_type {};

template <typename T1, typename T2>
struct is_thrust_pair<thrust::pair<T1, T2>> : std::true_type {};

template <typename T>
inline constexpr bool is_thrust_pair_v = is_thrust_pair<T>::value;

// Helper function to check if T is a fusion_array (fallback for older
// compilers)
template <typename T>
constexpr auto is_fusion_array_func() -> bool {
    return is_fusion_array<T>::value;
}

// First element of a pair
template <typename T, typename U>
struct fst_functor {
    __host__ __device__ auto operator()(const thrust::pair<T, U> &p) const
      -> T {
        return p.first;
    }
};

// Second element of a pair
template <typename T, typename U>
struct snd_functor {
    __host__ __device__ auto operator()(const thrust::pair<T, U> &p) const
      -> U {
        return p.second;
    }
};

/**
 * @brief Function object to extract the first element of a pair
 */
struct fst {
    template <typename T, typename U>
    __host__ __device__ auto operator()(const thrust::pair<T, U> &p) const
      -> T {
        return p.first;
    }
};

/**
 * @brief Function object to extract the second element of a pair
 */
struct snd {
    template <typename T, typename U>
    __host__ __device__ auto operator()(const thrust::pair<T, U> &p) const
      -> U {
        return p.second;
    }
};

// Function objects for transformations
template <typename T>
struct times_functor {
    T scalar;

    explicit times_functor(T s) : scalar(s) {}

    __host__ __device__ auto operator()(const T &x) const -> T {
        return x * scalar;
    }
};

template <typename T>
struct add_functor {
    T scalar;

    explicit add_functor(T s) : scalar(s) {}

    __host__ __device__ auto operator()(const T &x) const -> T {
        return x + scalar;
    }
};

// Delta functor for adjacent element differences
struct delta {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const
      -> decltype(b - a) {
        return b - a;
    }
};

// Min functor for elementwise minimum
template <typename T>
struct min_functor {
    T scalar;

    explicit min_functor(T s) : scalar(s) {}

    __host__ __device__ auto operator()(const T &x) const -> T {
        return x < scalar ? x : scalar;
    }
};

/**
 * @brief Function object for minimum of two values
 */
struct min {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const
      -> decltype(a < b ? a : b) {
        return a < b ? a : b;
    }
};

/**
 * @brief Function object for maximum of two values
 */
struct max {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const
      -> decltype(a > b ? a : b) {
        return a > b ? a : b;
    }
};

/**
 * @brief Function object for inequality comparison returning 1 if not equal, 0
 * otherwise
 */
struct neq {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const -> int {
        return a != b ? 1 : 0;
    }
};

/**
 * @brief Function object for less-than comparison returning 1 if a < b, 0
 * otherwise
 */
struct lt {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const -> int {
        return a < b ? 1 : 0;
    }
};

/**
 * @brief Function object for greater-than comparison returning 1 if a > b, 0
 * otherwise
 */
struct gt {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const -> int {
        return a > b ? 1 : 0;
    }
};

/**
 * @brief Function object for equality comparison returning 1 if equal, 0
 * otherwise
 */
struct eq {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const -> int {
        return a == b ? 1 : 0;
    }
};

/**
 * @brief Function object for less-than-or-equal comparison returning 1 if a <=
 * b, 0 otherwise
 */
struct lte {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const -> int {
        return a <= b ? 1 : 0;
    }
};

/**
 * @brief Function object for greater-than-or-equal comparison returning 1 if a
 * >= b, 0 otherwise
 */
struct gte {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const -> int {
        return a >= b ? 1 : 0;
    }
};

/**
 * @brief Function object for addition
 */
struct add {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const
      -> decltype(a + b) {
        return a + b;
    }
};

/**
 * @brief Function object for multiplication
 */
struct mul {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const
      -> decltype(a * b) {
        return a * b;
    }
};

/**
 * @brief Function object for division
 */
struct div {
    template <typename T1, typename T2>
    __host__ __device__ auto operator()(const T1 &a, const T2 &b) const
      -> decltype(a / b) {
        return a / b;
    }
};

/**
 * @brief Function object for integer division
 */
struct idiv {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const -> T {
        if (std::is_integral<T>::value) { return a / b; }
        return static_cast<T>(static_cast<int>(a / b));
    }
};

/**
 * @brief Function object for subtraction
 */
struct minus {
    template <typename T>
    __host__ __device__ auto operator()(const T &a, const T &b) const
      -> decltype(a - b) {
        return a - b;
    }
};

// Double functor to multiply elements by 2
template <typename T>
struct double_functor {
    __host__ __device__ auto operator()(const T &x) const -> T { return x * 2; }
};

// Half functor to divide elements by 2
template <typename T>
struct half_functor {
    __host__ __device__ auto operator()(const T &x) const -> T {
        if (std::is_integral<T>::value) {
            // For integer types, perform integer division
            return x / T(2);
        }  // For floating-point types, use regular division
        return x * T(0.5);
    }
};

// Absolute value functor
template <typename T>
struct abs_functor {
    __host__ __device__ auto operator()(const T &x) const -> T {
        return x < T(0) ? -x : x;
    }
};

// Odd check functor (returns 1 for odd numbers, 0 for even)
template <typename T>
struct odd_functor {
    __host__ __device__ auto operator()(const T &x) const -> int {
        return std::is_integral<T>::value ? (x % T(2) != T(0) ? 1 : 0) : 0;
    }
};

// Even check functor (returns 1 for even numbers, 0 for odd)
template <typename T>
struct even_functor {
    __host__ __device__ auto operator()(const T &x) const -> int {
        return std::is_integral<T>::value ? (x % T(2) == T(0) ? 1 : 0) : 0;
    }
};

// Sign functor (returns 1 for positive, 0 for zero, -1 for negative)
template <typename T>
struct sign_functor {
    __host__ __device__ auto operator()(const T &x) const -> int {
        return (x > T(0)) ? 1 : ((x < T(0)) ? -1 : 0);
    }
};

// Square root functor
template <typename T>
struct sqrt_functor {
    __host__ __device__ auto operator()(const T &x) const -> T {
        return std::sqrt(x);
    }
};

// Logarithm functor
template <typename T>
struct log_functor {
    __host__ __device__ auto operator()(const T &x) const -> T {
        return std::log(x);
    }
};

// Exponential functor
template <typename T>
struct exp_functor {
    __host__ __device__ auto operator()(const T &x) const -> T {
        return std::exp(x);
    }
};

// Square functor
template <typename T>
struct sq_functor {
    __host__ __device__ auto operator()(const T &x) const -> T { return x * x; }
};

// Type casting functor
template <typename TargetType, typename SourceType>
struct cast_functor {
    __host__ __device__ auto operator()(const SourceType &x) const
      -> TargetType {
        return static_cast<TargetType>(x);
    }
};

// Random number generator struct
struct rnd {
    float a, b;
    unsigned int seed;

#ifndef __CUDA_ARCH__
    static std::atomic<unsigned int> global_counter;
#endif

    __host__ __device__ explicit rnd(float _a = 0.F, float _b = 1.F)
      : a(_a), b(_b) {
#ifndef __CUDA_ARCH__
        // Use a true entropy source when available
        std::random_device rd;
        // Mix entropy sources: random_device + timestamp + counter + address
        auto timestamp = static_cast<unsigned int>(time(nullptr));
        unsigned int const counter_val = global_counter.fetch_add(
          1, std::memory_order_relaxed);
        // Mix multiple entropy sources using XOR and rotation
        auto tmp = reinterpret_cast<uintptr_t>(this);  // NOLINT
        seed     = rd() ^ (timestamp << 16 | timestamp >> 16) ^
               (counter_val * 2654435761U) ^
               static_cast<unsigned int>(tmp & 0xFFFFFFFF);
#else
        // On device, mix thread/block IDs with a base seed
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        seed             = 42U + tid + clock();
#endif
    }

    __host__ __device__ auto operator()(const unsigned int n) const -> float {
        // Use multiple mixing steps for better statistical properties
        auto h1 = seed + n;
        h1      = ((h1 >> 16) ^ h1) * 0x45d9f3b;
        h1      = ((h1 >> 16) ^ h1) * 0x45d9f3b;
        h1      = (h1 >> 16) ^ h1;

        thrust::default_random_engine rng(
          static_cast<thrust::default_random_engine::result_type>(h1));
        thrust::uniform_real_distribution<float> dist(a, b);
        return dist(rng);
    }
};

#ifndef __CUDA_ARCH__
// Initialize atomic counter
std::atomic<unsigned int> rnd::global_counter(0);  // NOLINT
#endif

// Random value functor (uses element index to generate random number, then
// scales by element value)
template <typename T>
struct rand_functor {
    unsigned int extra_entropy;

    __host__ __device__ rand_functor() {
#ifndef __CUDA_ARCH__
        // Use true entropy source
        std::random_device rd;
        // Mix multiple sources of entropy
        extra_entropy = rd() ^
                        (static_cast<unsigned int>(clock()) * 2654435761U) ^
                        static_cast<unsigned int>(
                          reinterpret_cast<uintptr_t>(this) &  // NOLINT
                          0xFFFFFFFF);
#else
        // On device, use thread ID and clock
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        extra_entropy    = 0x9e3779b9 + tid + clock();
#endif
    }

    __host__ __device__ auto operator()(const thrust::tuple<int, T> &t) const
      -> T {
        // Get the index and value
        int idx{thrust::get<0>(t)};
        T val = thrust::get<1>(t);

        // Improved hash mixing for better distribution
        auto h1 = static_cast<unsigned int>(idx ^ extra_entropy);
        h1      = ((h1 >> 16) ^ h1) * 0x45d9f3b;
        h1      = ((h1 >> 16) ^ h1) * 0x45d9f3b;
        h1      = (h1 >> 16) ^ h1;

        // Generate a random float between 0 and 1 with extra entropy
        thrust::default_random_engine rng(
          static_cast<thrust::default_random_engine::result_type>(h1));
        thrust::uniform_real_distribution<float> dist(0.0F, 1.0F);
        float rand_val = dist(rng);

        // Scale by value and convert to appropriate type
        if (std::is_integral<T>::value) {
            // For integers, return a random integer in [0, val]
            return static_cast<T>(rand_val * val);
        }  // For floating point, return a random float in [0, val)
        return static_cast<T>(rand_val * val);
    }
};

// Adapter to convert binary functors from tuple-based to two-argument calling
// convention
template <typename BinaryFunctor>
struct binary_op_adapter {
    BinaryFunctor binary_op;

    explicit binary_op_adapter(BinaryFunctor op = BinaryFunctor())
      : binary_op(op) {}

    template <typename T1, typename T2>
    __host__ __device__ auto operator()(const thrust::tuple<T1, T2> &t) const
      -> decltype(binary_op(thrust::get<0>(t), thrust::get<1>(t))) {
        return binary_op(thrust::get<0>(t), thrust::get<1>(t));
    }
};

// Add a helper to convert binary functors to unary by binding the second
// operand
template <typename T, typename BinaryFunctor>
struct make_unary_functor {
    T scalar;
    BinaryFunctor binary_op;

    explicit make_unary_functor(T s, BinaryFunctor op = BinaryFunctor())
      : scalar(s), binary_op(op) {}

    __host__ __device__ auto operator()(const auto &x) const
      -> decltype(binary_op(x, scalar)) {
        return binary_op(x, scalar);
    }
};

template <typename T>
using minmax_pair = thrust::pair<T, T>;

template <typename T>
struct minmax_unary_op {
    __host__ __device__ auto operator()(const T &x) const -> minmax_pair<T> {
        return thrust::make_pair(x, x);
    }
};

template <typename T>
struct minmax_binary_op {
    __host__ __device__ auto operator()(const minmax_pair<T> &x,
                                        const minmax_pair<T> &y) const
      -> minmax_pair<T> {
        return thrust::make_pair(thrust::min(x.first, y.first),
                                 thrust::max(x.second, y.second));
    }
};

// Functor to convert a tuple to a pair
template <typename T>
struct tuple_to_pair_functor {
    __host__ __device__ auto operator()(const thrust::tuple<T, int> &t) const
      -> thrust::pair<T, int> {
        return thrust::pair<T, int>(thrust::get<0>(t), thrust::get<1>(t));
    }
};

// Create a functor to make pairs from a tuple
template <typename T1, typename T2>
struct make_pair_functor {
    __host__ __device__ auto operator()(const thrust::tuple<T1, T2> &t) const
      -> thrust::pair<T1, T2> {
        return thrust::make_pair(thrust::get<0>(t), thrust::get<1>(t));
    }
};

// Transpose functor for permutation iterator
struct transpose_functor {
    int num_rows_orig;
    int num_cols_orig;

    transpose_functor(int rows, int cols)  // NOLINT
      : num_rows_orig(rows), num_cols_orig(cols) {}

    __host__ __device__ auto operator()(int idx_new_linear) const -> int {
        return (idx_new_linear % num_rows_orig) * num_cols_orig +
               (idx_new_linear / num_rows_orig);
    }
};

// Pair operation functor for applying binary operations to pairs
template <typename BinaryOp>
struct pair_op_functor {
    BinaryOp binary_op;

    explicit pair_op_functor(BinaryOp op) : binary_op(op) {}

    template <typename T1, typename T2>
    __host__ __device__ auto operator()(const thrust::pair<T1, T2> &pair) const
      -> decltype(binary_op(pair.first, pair.second)) {
        return binary_op(pair.first, pair.second);
    }
};

// Main class for lazy operations - now supports optional mask
template <typename Iterator, typename MaskIterator>
class fusion_array {
   private:
    Iterator _begin;
    Iterator _end;

    // Optional ownership of a device vector
    std::shared_ptr<void> _owned_storage = nullptr;

    // Shape information for the array (always 1D for now)
    std::vector<int> _shape;

    // Optional mask iterators (only used when MaskIterator != no_mask_t)
    [[no_unique_address]] std::conditional_t<
      std::is_same_v<MaskIterator, no_mask_t>,
      std::monostate,
      std::pair<MaskIterator, MaskIterator>>
      _mask_range;

    // Optional storage to keep mask source alive (only for masked arrays)
    [[no_unique_address]] std::conditional_t<
      std::is_same_v<MaskIterator, no_mask_t>,
      std::monostate,
      std::shared_ptr<void>>
      _mask_storage;

    // Helper to check if this is a masked array
    static constexpr bool has_mask = !std::is_same_v<MaskIterator, no_mask_t>;

    // Track if the array is sorted (for optimization purposes)
    bool _is_sorted = false;

    // Helper to apply mask for eager operations
    [[nodiscard]] auto _apply_mask_if_needed() const {
        using value_type = typename cuda::std::iterator_traits<
          Iterator>::value_type;

        if constexpr (!has_mask) {
            // No mask - just return a copy of the data
            int n = size();
            auto
              result_vec = std::make_shared<thrust::device_vector<value_type>>(
                n, thrust::default_init);
            thrust::copy(_begin, _end, result_vec->begin());

            return fusion_array<
              typename thrust::device_vector<value_type>::iterator,
              no_mask_t>(result_vec->begin(), result_vec->end(), result_vec);
        } else {
            // Has mask - apply it
            int n = size();
            auto
              result_vec = std::make_shared<thrust::device_vector<value_type>>(
                n, thrust::default_init);

            auto end = thrust::copy_if(_begin,
                                       _end,
                                       _mask_range.first,
                                       result_vec->begin(),
                                       cuda::std::identity{});

            int result_size = cuda::std::distance(result_vec->begin(), end);

            return fusion_array<
              typename thrust::device_vector<value_type>::iterator,
              no_mask_t>(result_vec->begin(),
                         result_vec->begin() + result_size,
                         result_vec);
        }
    }

   public:
    using value_type = typename cuda::std::iterator_traits<
      Iterator>::value_type;

    // Storage accessor for composite storage management
    [[nodiscard]] const std::shared_ptr<void> &storage() const {
        return _owned_storage;
    }

    // Basic constructor (no mask)
    fusion_array(Iterator begin, Iterator end)
      : _begin(begin),
        _end(end),
        _shape{static_cast<int>(cuda::std::distance(begin, end))},
        _mask_range{},
        _mask_storage{} {
        static_assert(
          std::is_same_v<MaskIterator, no_mask_t>,
          "Use the constructor with mask parameters for masked arrays");
    }

    // Constructor with mask - only enable for non-no_mask_t types
    template <typename MI = MaskIterator>
    fusion_array(Iterator begin,
                 Iterator end,
                 MI mask_begin,
                 MI mask_end,
                 std::shared_ptr<void> mask_storage =
                   nullptr) requires(!std::is_same_v<MI, no_mask_t>)
      : _begin(begin),
        _end(end),
        _shape{static_cast<int>(cuda::std::distance(begin, end))},
        _mask_range{mask_begin, mask_end},
        _mask_storage{std::move(mask_storage)} {}

    // Constructor with storage (no mask)
    fusion_array(Iterator begin, Iterator end, std::shared_ptr<void> storage)
      : _begin(begin),
        _end(end),
        _owned_storage(std::move(storage)),
        _shape{static_cast<int>(cuda::std::distance(begin, end))},
        _mask_range{},
        _mask_storage{} {
        static_assert(
          std::is_same_v<MaskIterator, no_mask_t>,
          "Use the constructor with mask parameters for masked arrays");
    }

    // Constructor with storage and is_sorted flag (no mask)
    fusion_array(Iterator begin,
                 Iterator end,
                 std::shared_ptr<void> storage,
                 bool is_sorted)
      : _begin(begin),
        _end(end),
        _owned_storage(std::move(storage)),
        _shape{static_cast<int>(cuda::std::distance(begin, end))},
        _mask_range{},
        _mask_storage{},
        _is_sorted(is_sorted) {
        static_assert(
          std::is_same_v<MaskIterator, no_mask_t>,
          "Use the constructor with mask parameters for masked arrays");
    }

    // Constructor with storage and mask
    template <typename MI = MaskIterator>
    fusion_array(Iterator begin,
                 Iterator end,
                 std::shared_ptr<void> storage,
                 MI mask_begin,
                 MI mask_end,
                 std::shared_ptr<void> mask_storage =
                   nullptr) requires(!std::is_same_v<MI, no_mask_t>)
      : _begin(begin),
        _end(end),
        _owned_storage(std::move(storage)),
        _shape{static_cast<int>(cuda::std::distance(begin, end))},
        _mask_range{mask_begin, mask_end},
        _mask_storage{std::move(mask_storage)} {}

    // Constructor with storage and shape (no mask)
    fusion_array(Iterator begin,
                 Iterator end,
                 std::shared_ptr<void> storage,
                 const std::vector<int> &shape)
      : _begin(begin),
        _end(end),
        _owned_storage(std::move(storage)),
        _shape(shape),
        _mask_range{},
        _mask_storage{} {
        static_assert(
          std::is_same_v<MaskIterator, no_mask_t>,
          "Use the constructor with mask parameters for masked arrays");
    }

    // Constructor with storage, shape and mask
    template <typename MI = MaskIterator>
    fusion_array(Iterator begin,
                 Iterator end,
                 std::shared_ptr<void> storage,
                 const std::vector<int> &shape,
                 MI mask_begin,
                 MI mask_end,
                 std::shared_ptr<void> mask_storage =
                   nullptr) requires(!std::is_same_v<MI, no_mask_t>)
      : _begin(begin),
        _end(end),
        _owned_storage(std::move(storage)),
        _shape(shape),
        _mask_range{mask_begin, mask_end},
        _mask_storage{std::move(mask_storage)} {}

    // Constructor with initializer_list shape (no mask)
    fusion_array(Iterator begin,
                 Iterator end,
                 std::shared_ptr<void> storage,
                 std::initializer_list<int> shape)
      : _begin(begin),
        _end(end),
        _owned_storage(std::move(storage)),
        _shape(shape.begin(), shape.end()),
        _mask_range{},
        _mask_storage{} {
        static_assert(
          std::is_same_v<MaskIterator, no_mask_t>,
          "Use the constructor with mask parameters for masked arrays");
    }

    // Scalar constructor - only for non-masked arrays
    template <typename T>
    explicit fusion_array(T value)
      : _begin(thrust::make_constant_iterator(value)),
        _end(thrust::make_constant_iterator(value) + 1),
        _shape{},  // Scalar has empty shape (rank 0)
        _mask_range{},
        _mask_storage{} {
        static_assert(std::is_same_v<MaskIterator, no_mask_t>,
                      "Scalar constructor is only for non-masked arrays");
    }

    // Iterator accessors
    [[nodiscard]] auto begin() const -> Iterator { return _begin; }
    [[nodiscard]] auto end() const -> Iterator { return _end; }

    // For masked arrays, provide a method to apply the mask (eager operation)
    [[nodiscard]] auto apply() const {
        static_assert(has_mask, "apply() can only be called on masked arrays");
        return _apply_mask_if_needed();
    }

    // Shape accessor
    [[nodiscard]] auto shape() const -> std::vector<int> {
        if constexpr (has_mask) {
            // For masked arrays, return 1D shape with the masked size
            return {size()};
        } else {
            return _shape;
        }
    }

    /**
     * @brief Get the rank (dimensionality) of the array
     * @return The number of dimensions in the array's shape
     */
    [[nodiscard]] auto rank() const -> int { return _shape.size(); }

    /**
     * @brief Get the number of rows in a 2D array
     * @return The number of rows (first dimension)
     * @throws std::runtime_error if the array rank is not 2
     */
    [[nodiscard]] auto nrows() const -> int {
        if (rank() != 2) {
            throw std::runtime_error(
              "nrows() can only be called on rank-2 arrays");
        }
        return _shape[0];
    }

    /**
     * @brief Get the number of columns in a 2D array
     * @return The number of columns (second dimension)
     * @throws std::runtime_error if the array rank is not 2
     */
    [[nodiscard]] auto ncols() const -> int {
        if (rank() != 2) {
            throw std::runtime_error(
              "ncols() can only be called on rank-2 arrays");
        }
        return _shape[1];
    }

    // Map operation with another fusion_array
    template <typename OtherIterator,
              typename OtherMaskIterator,
              typename BinaryFunctor>
    auto map2(const fusion_array<OtherIterator, OtherMaskIterator> &value,
              BinaryFunctor binary_op) const {
        if (size() != value.size() and rank() != 0 and value.rank() != 0) {
            throw std::invalid_argument(
              "Incompatible shapes for element-wise operations: " +
              std::to_string(size()) + " " + std::to_string(value.size()));
        }

        auto const n     = std::max(size(), value.size());
        auto const shape = size() < value.size() ? value.shape() : _shape;

        auto zip_begin = thrust::make_zip_iterator(
          thrust::make_tuple(_begin, value.begin()));
        auto zip_end = thrust::make_zip_iterator(
          thrust::make_tuple(_begin + n, value.begin() + n));

        // Create an adapter that converts from tuple-based to two-argument
        // calling convention
        auto adapter = binary_op_adapter<BinaryFunctor>(binary_op);

        using result_iterator = thrust::transform_iterator<decltype(adapter),
                                                           decltype(zip_begin)>;

        // Create composite storage to keep both operands alive
        std::shared_ptr<void> composite_storage;
        if (_owned_storage || value.storage()) {
            composite_storage = std::make_shared<
              std::pair<std::shared_ptr<void>, std::shared_ptr<void>>>(
              _owned_storage, value.storage());
        }

        return fusion_array<result_iterator>(
          thrust::make_transform_iterator(zip_begin, adapter),
          thrust::make_transform_iterator(zip_end, adapter),
          composite_storage,
          shape);
    }

    // Map operation with a scalar value
    template <typename T, typename BinaryFunctor>
    auto map2(const T &value, BinaryFunctor binary_op) const -> decltype(auto) {
        if constexpr (has_mask) {
            return apply().map2(value, binary_op);
        } else {
            using unary_functor = make_unary_functor<value_type, BinaryFunctor>;
            using TransformIterator = thrust::transform_iterator<unary_functor,
                                                                 Iterator>;

            return fusion_array<TransformIterator>(
              thrust::make_transform_iterator(_begin,
                                              unary_functor(value, binary_op)),
              thrust::make_transform_iterator(_end,
                                              unary_functor(value, binary_op)),
              _owned_storage,
              _shape);
        }
    }

    /**
     * @brief Apply a unary functor to each element of the array
     * @tparam UnaryFunctor The type of the unary functor
     * @param op The unary operation to apply
     * @return A new fusion_array with the operation applied
     */
    template <typename UnaryFunctor>
    auto map(UnaryFunctor op) const {
        using TransformIterator = thrust::transform_iterator<UnaryFunctor,
                                                             Iterator>;

        return fusion_array<TransformIterator>(
          thrust::make_transform_iterator(_begin, op),
          thrust::make_transform_iterator(_end, op),
          _owned_storage,  // Pass ownership to derived array
          _shape);
    }

    /**
     * @brief Multiply each element by a scalar
     * @param scalar The value to multiply by
     * @return A new fusion_array with the operation applied
     */
    template <typename T = int>
    auto times(T scalar) const {
        return map2(scalar, mul{});
    }

    /**
     * @brief Add a scalar or another fusion_array element-wise
     * @param value The scalar or fusion_array to add
     * @return A new fusion_array with the operation applied
     */
    template <typename T>
    auto add(const T &value) const {
        return map2(value, parrot::add{});
    }

    /**
     * @brief Return elementwise minimum with a scalar
     * @param scalar The value to compare with
     * @return A new fusion_array with the operation applied
     */
    template <typename T = int>
    auto min(T scalar) const {
        return map2(scalar, parrot::min{});
    }

    /**
     * @brief Return elementwise maximum with a scalar
     * @param scalar The value to compare with
     * @return A new fusion_array with the operation applied
     */
    template <typename T = int>
    auto max(T scalar) const {
        return map2(scalar, parrot::max{});
    }

    /**
     * @brief Divide each element by a scalar or perform element-wise division
     * with another fusion_array.
     * @tparam T The type of the argument, which can be a scalar or another
     * fusion_array.
     * @param arg The scalar value to divide by or the other fusion_array for
     * element-wise division.
     * @return A new fusion_array with the division operation applied.
     * @throws std::invalid_argument if `arg` is a fusion_array and shapes are
     * incompatible for element-wise operations.
     */
    template <typename T>
    auto div(const T &arg) const {
        return map2(arg, parrot::div{});
    }

    /**
     * @brief Integer divide each element by a scalar (truncating any fractional
     * part)
     * @param scalar The value to divide by
     * @return A new fusion_array with the operation applied
     */
    template <typename T = int>
    auto idiv(T scalar) const {
        return map2(scalar, parrot::idiv{});
    }

    /**
     * @brief Subtract a scalar from each element
     * @param scalar The value to subtract
     * @return A new fusion_array with the operation applied
     */
    template <typename T = int>
    auto minus(T scalar) const {
        return map2(scalar, parrot::minus{});
    }

    /**
     * @brief Check if each element is greater than or equal to a scalar
     * @param scalar The value to compare with
     * @return A new fusion_array containing 1 where the condition is true, 0
     * otherwise
     */
    template <typename T = int>
    auto gte(T scalar) const {
        return map2(scalar, parrot::gte{});
    }

    /**
     * @brief Check if each element is less than or equal to a scalar
     * @param scalar The value to compare with
     * @return A new fusion_array containing 1 where the condition is true, 0
     * otherwise
     */
    template <typename T = int>
    auto lte(T scalar) const {
        return map2(scalar, parrot::lte{});
    }

    /**
     * @brief Check if each element is greater than a scalar
     * @param scalar The value to compare with
     * @return A new fusion_array containing 1 where the condition is true, 0
     * otherwise
     */
    template <typename T = int>
    auto gt(T scalar) const {
        return map2(scalar, parrot::gt{});
    }

    /**
     * @brief Check if each element is less than a scalar
     * @param scalar The value to compare with
     * @return A new fusion_array containing 1 where the condition is true, 0
     * otherwise
     */
    template <typename T = int>
    auto lt(T scalar) const {
        return map2(scalar, parrot::lt{});
    }

    /**
     * @brief Check if each element is equal to a scalar
     * @param scalar The value to compare with
     * @return A new fusion_array containing 1 where the condition is true, 0
     * otherwise
     */
    template <typename T = int>
    auto eq(T scalar) const {
        return map2(scalar, parrot::eq{});
    }

    /**
     * @brief Apply a binary operation to adjacent elements
     * @param op The binary operation to apply
     * @return A new fusion_array with the operation applied
     */
    template <typename BinaryOp>
    auto map_adj(BinaryOp op) const -> fusion_array<thrust::transform_iterator<
      thrust::zip_function<BinaryOp>,
      thrust::zip_iterator<thrust::tuple<Iterator, Iterator>>>> {
        auto zip_begin = thrust::make_zip_iterator(
          thrust::make_tuple(_begin, _begin + 1));
        auto transform_begin = thrust::make_transform_iterator(
          zip_begin, thrust::make_zip_function(op));

        using return_type = fusion_array<decltype(transform_begin)>;

        // If we don't have at least 2 elements, return empty array type
        if (size() < 2) {
            return return_type(
              transform_begin, transform_begin, _owned_storage);
        }

        // Return a new array with one fewer element (since we're processing
        // pairs)
        return return_type(
          transform_begin, transform_begin + (size() - 1), _owned_storage);
    }

    /**
     * @brief Check if adjacent elements are different
     * @return A new fusion_array containing 1 where adjacent elements differ,
     * 0 otherwise
     */
    [[nodiscard]] auto differ() const { return map_adj(neq{}); }

    /**
     * @brief Compute differences between adjacent elements
     * @return A new fusion_array containing the differences
     */
    [[nodiscard]] auto deltas() const { return map_adj(delta{}); }

    /**
     * @brief Double each element (multiply by 2)
     * @return A new fusion_array with doubled values
     */
    [[nodiscard]] auto dble() const {
        return map(double_functor<value_type>());
    }

    /**
     * @brief Halve each element (divide by 2)
     * @return A new fusion_array with halved values
     */
    [[nodiscard]] auto half() const { return map(half_functor<value_type>()); }

    /**
     * @brief Calculate absolute value of each element
     * @return A new fusion_array with absolute values
     */
    [[nodiscard]] auto abs() const { return map(abs_functor<value_type>()); }

    /**
     * @brief Calculate natural logarithm of each element
     * @return A new fusion_array with logarithm values
     */
    [[nodiscard]] auto log() const { return map(log_functor<value_type>()); }

    /**
     * @brief Calculate exponential of each element
     * @return A new fusion_array with exponential values
     */
    [[nodiscard]] auto exp() const { return map(exp_functor<value_type>()); }

    /**
     * @brief Check if each element is odd
     * @return A new fusion_array containing 1 for odd elements, 0 for even
     */
    [[nodiscard]] auto neg() const { return map(thrust::negate<value_type>()); }

    /**
     * @brief Check if each element is odd
     * @return A new fusion_array containing 1 for odd elements, 0 for even
     */
    [[nodiscard]] auto odd() const { return map(odd_functor<value_type>()); }

    /**
     * @brief Check if each element is even
     * @return A new fusion_array containing 1 for even elements, 0 for odd
     */
    [[nodiscard]] auto even() const { return map(even_functor<value_type>()); }

    /**
     * @brief Get sign of each element
     * @return A new fusion_array containing 1 for positive, 0 for zero, -1 for
     * negative
     */
    [[nodiscard]] auto sign() const { return map(sign_functor<value_type>()); }

    /**
     * @brief Calculate square root of each element
     * @return A new fusion_array with square root values
     */
    [[nodiscard]] auto sqrt() const { return map(sqrt_functor<value_type>()); }

    /**
     * @brief Square each element
     * @return A new fusion_array with squared values
     */
    [[nodiscard]] auto sq() const { return map(sq_functor<value_type>()); }

    /**
     * @brief Cast each element to a different type
     * @tparam TargetType The target type to cast elements to
     * @return A new fusion_array with elements cast to the target type
     */
    template <typename TargetType>
    [[nodiscard]] auto as() const {
        return map(cast_functor<TargetType, value_type>());
    }

    /**
     * @brief Generate random values between 0 and each element
     * @return A new fusion_array with random values
     */
    [[nodiscard]] auto rand() const {
        // Create a counting iterator for indices
        auto indices = thrust::make_counting_iterator(0);

        // Create a zip iterator combining indices with the array values
        auto zip_begin = thrust::make_zip_iterator(
          thrust::make_tuple(indices, _begin));
        auto zip_end = thrust::make_zip_iterator(
          thrust::make_tuple(indices + size(), _end));

        // Create a transform iterator that applies the random functor
        using RandFunc = rand_functor<value_type>;

        return fusion_array<
          thrust::transform_iterator<RandFunc, decltype(zip_begin)>>(
          thrust::make_transform_iterator(zip_begin, RandFunc()),
          thrust::make_transform_iterator(zip_end, RandFunc()),
          _owned_storage,
          _shape);
    }

    /**
     * @brief Get the size of the array
     * @return The number of elements
     */
    [[nodiscard]] auto size() const -> int {
        if constexpr (has_mask) {
            // For masked arrays, count the number of non-zero mask elements
            auto mask_begin = _mask_range.first;
            auto mask_end   = _mask_range.second;
            return thrust::count_if(
              mask_begin, mask_end, cuda::std::identity{});
        } else {
            return cuda::std::distance(_begin, _end);
        }
    }

    /**
     * @brief Append a value to the array
     * @param value The value to append
     * @return A new fusion_array with the appended value
     */
    template <typename T = int>
    auto append(T value) const {
        if constexpr (has_mask) {
            // Apply mask first, then append
            auto unmasked = _apply_mask_if_needed();
            return unmasked.append(value);
        } else {
            int const n         = size();
            auto appended_begin = thrustx::make_append_iterator(
              _begin, n, value);

            return fusion_array<decltype(appended_begin)>(
              appended_begin,
              appended_begin + n + 1,
              _owned_storage  // Pass ownership to derived array
            );
        }
    }

    /**
     * @brief Prepend a value to the array
     * @param value The value to prepend
     * @return A new fusion_array with the prepended value
     */
    template <typename T = int>
    auto prepend(T value) const {
        if constexpr (has_mask) {
            // Apply mask first, then prepend
            auto unmasked = _apply_mask_if_needed();
            return unmasked.prepend(value);
        } else {
            int const n          = size();
            auto prepended_begin = thrustx::make_prepend_iterator(
              _begin, n, value);

            return fusion_array<decltype(prepended_begin)>(
              prepended_begin,
              prepended_begin + n + 1,
              _owned_storage  // Pass ownership to derived array
            );
        }
    }

    /**
     * @brief Sort the array (eager operation)
     * @return A new fusion_array with sorted elements
     */
    [[nodiscard]] auto sort() const {
        int n = size();

        // Create a new device vector and take ownership of it
        auto sorted_data = std::make_shared<thrust::device_vector<value_type>>(
          n, thrust::default_init);

        thrust::copy(_begin, _end, sorted_data->begin());
        thrust::sort(sorted_data->begin(), sorted_data->end());

        // Return a new fusion_array with ownership of the sorted data
        return fusion_array<
          typename thrust::device_vector<value_type>::iterator>(
          sorted_data->begin(), sorted_data->end(), sorted_data, true);
    }

    /**
     * @brief Sort the array using a custom comparator (eager operation)
     * @param comp The binary comparator to use for sorting
     * @return A new fusion_array with sorted elements
     */
    template <typename BinaryComp>
    auto sort_by(BinaryComp comp) const {
        int n = size();

        // Create a new device vector and take ownership of it
        auto sorted_data = std::make_shared<thrust::device_vector<value_type>>(
          n, thrust::default_init);

        thrust::copy(_begin, _end, sorted_data->begin());
        thrust::sort(sorted_data->begin(), sorted_data->end(), comp);

        // Return a new fusion_array with ownership of the sorted data
        return fusion_array<
          typename thrust::device_vector<value_type>::iterator>(
          sorted_data->begin(), sorted_data->end(), sorted_data, true);
    }

    /**
     * @brief Sort the array based on keys produced by a unary function (eager
     * operation)
     * @param key_func The unary function to extract keys for comparison
     * @return A new fusion_array with sorted elements
     */
    template <typename KeyFunc>
    auto sort_by_key(KeyFunc key_func) const
      -> fusion_array<typename thrust::device_vector<
        typename cuda::std::iterator_traits<Iterator>::value_type>::iterator> {
        int n = size();

        // Create a new device vector and take ownership of it
        auto sorted_data = std::make_shared<thrust::device_vector<value_type>>(
          n, thrust::default_init);

        // Copy the array to the device vector
        thrust::copy(_begin, _end, sorted_data->begin());

        // Create a comparator that uses the key function
        auto key_comp = [key_func] __host__ __device__(const value_type &a,
                                                       const value_type &b) {
            return key_func(a) < key_func(b);
        };

        // Sort using the key comparator
        thrust::sort(sorted_data->begin(), sorted_data->end(), key_comp);

        // Return a new fusion_array with ownership of the sorted data
        return fusion_array<
          typename thrust::device_vector<value_type>::iterator>(
          sorted_data->begin(), sorted_data->end(), sorted_data, true);
    }

    /**
     * @brief Generic reduction with custom binary operation (eager operation)
     * @tparam Axis The axis along which to reduce (0=all elements, 2=row-wise)
     * @param init The initial value for the reduction
     * @param op The binary operation to apply for reduction
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A fusion_array containing the reduction result(s)
     */
    template <int Axis = 0, typename T, typename BinaryOp>
    auto reduce(T init,
                BinaryOp op,
                std::integral_constant<int, Axis> axis = {}) const {
        using value_type = typename std::iterator_traits<Iterator>::value_type;

        if constexpr (Axis == 0) {
            // Default reduction (all elements)
            // Perform the reduction to get a single scalar value
            auto result = thrust::reduce(_begin, _end, init, op);

            // Return a fusion_array with a constant iterator of the result
            return fusion_array<thrust::constant_iterator<decltype(result)>>(
              thrust::make_constant_iterator(result),
              thrust::make_constant_iterator(result) + 1,
              nullptr,
              std::vector<int>{});
        } else if constexpr (Axis == 2) {
            // Row-wise reduction (for 2D arrays)
            if (_shape.size() < 2) {
                throw std::runtime_error(
                  "Cannot perform row-wise reduction on array with rank < 2");
            }

            int num_rows = _shape[0];
            int num_cols = _shape[1];

            auto
              result_vec = std::make_shared<thrust::device_vector<value_type>>(
                num_rows, thrust::default_init);

            // Perform row-wise reduction using reduce_by_key
            auto output = result_vec->begin();

            thrustx::reduce_by_n(_begin, _end, output, num_cols, op, init);

            // Return result as fusion_array
            return fusion_array<
              typename thrust::device_vector<value_type>::iterator>(
              result_vec->begin(),
              result_vec->end(),
              result_vec,
              std::vector<int>{num_rows});
        } else {
            static_assert(Axis == 0 || Axis == 2,
                          "Invalid axis value. Must be 0 or 2.");
            // This will never be reached due to static_assert, but needed for
            // compilation
            return fusion_array<typename thrust::device_vector<T>::iterator>();
        }
    }

    /**
     * @brief Maximum reduction (eager operation)
     * @tparam Axis The axis along which to reduce (0=all elements, 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A fusion_array containing the maximum value(s)
     */
    template <int Axis = 0, typename T = int>
    [[nodiscard]] auto maxr(std::integral_constant<int, Axis> axis = {}) const {
        return reduce<Axis>(std::numeric_limits<value_type>::lowest(),
                            thrust::maximum<value_type>());
    }

    /**
     * @brief Minimum reduction (eager operation)
     * @tparam Axis The axis along which to reduce (0=all elements, 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A fusion_array containing the minimum value(s)
     */
    template <int Axis = 0, typename T = int>
    [[nodiscard]] auto minr(std::integral_constant<int, Axis> axis = {}) const {
        return reduce<Axis>(std::numeric_limits<value_type>::max(),
                            thrust::minimum<value_type>());
    }

    /**
     * @brief Sum reduction (eager operation)
     * @tparam Axis The axis along which to reduce (0=all elements, 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A fusion_array containing the sum of elements
     */
    template <int Axis = 0, typename T = int>
    [[nodiscard]] auto sum(std::integral_constant<int, Axis> axis = {}) const {
        if constexpr (has_mask) {
            // Apply mask first
            auto unmasked = _apply_mask_if_needed();
            return unmasked.template sum<Axis>();
        } else {
            return reduce<Axis>(value_type(0), thrust::plus<value_type>());
        }
    }

    /**
     * @brief Check if any element is non-zero (eager operation)
     * @tparam Axis The axis along which to reduce (0=all elements, 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A fusion_array containing true if any element is non-zero, false
     * otherwise
     */
    template <int Axis = 0, typename T = int>
    [[nodiscard]] auto any(std::integral_constant<int, Axis> axis = {}) const {
        return reduce<Axis>(0, thrust::logical_or<int>());
    }

    /**
     * @brief Check if all elements are non-zero (eager operation)
     * @tparam Axis The axis along which to reduce (0=all elements, 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A fusion_array containing true if all elements are non-zero,
     * false otherwise
     */
    template <int Axis = 0, typename T = int>
    [[nodiscard]] auto all(std::integral_constant<int, Axis> axis = {}) const {
        return reduce<Axis>(1, thrust::logical_and<int>());
    }

    /**
     * @brief Get the first value (useful for extracting reduction results)
     * @return The first value in the array, properly handling thrust pairs
     */
    [[nodiscard]] auto value() const {
        if constexpr (is_thrust_pair_v<value_type>) {
            // For thrust pairs, explicitly copy device_reference to host pair
            using pair_type = typename cuda::std::iterator_traits<
              Iterator>::value_type;
            thrust::pair<typename pair_type::first_type,
                         typename pair_type::second_type>
              host_pair = *_begin;
            return host_pair;
        } else {
            return *_begin;
        }
    }

    /**
     * @brief Get the first element of the array
     * @return The first element in the array
     */
    [[nodiscard]] auto front() const {
        using host_value_type = extract_value_type_t<value_type>;
        if constexpr (has_mask) {
            // Apply mask first
            auto unmasked = _apply_mask_if_needed();
            return static_cast<host_value_type>(unmasked.front());
        }
        return static_cast<host_value_type>(*_begin);
    }

    /**
     * @brief Get the last element of the array
     * @return The last element in the array
     */
    [[nodiscard]] auto back() const {
        using host_value_type = extract_value_type_t<value_type>;
        if constexpr (has_mask) {
            // Apply mask first and convert to host type
            auto unmasked = _apply_mask_if_needed();
            return static_cast<host_value_type>(unmasked.back());
        }
        // Convert to consistent host type
        return static_cast<host_value_type>(*(_end - 1));
    }

    /**
     * @brief Copy elements from device to host memory
     * @return A std::vector containing the host-side values
     */
    [[nodiscard]] auto to_host() const {
        if constexpr (has_mask) {
            // Apply mask first
            auto unmasked = _apply_mask_if_needed();
            return unmasked.to_host();
        } else {
            int n = size();

            // Use the extracted value type to avoid device_reference
            // constructor issues
            using host_value_type = extract_value_type_t<value_type>;
            std::vector<host_value_type> host_vector;
            host_vector.reserve(n);

            for (int i = 0; i < n; ++i) {
                host_vector.push_back(static_cast<host_value_type>(_begin[i]));
            }

            return host_vector;
        }
    }

    /**
     * @brief Product reduction (eager operation)
     * @tparam Axis The axis along which to reduce (0=all elements, 2=row-wise)
     * @return A fusion_array containing the product of elements
     */
    template <int Axis = 0, typename T = int>
    [[nodiscard]] auto prod() const {
        return reduce<Axis>(value_type(1), thrust::multiplies<value_type>());
    }

    /**
     * @brief Get indices where elements are non-zero (lazy operation)
     * @return A masked fusion_array containing the 1-indexed positions where
     * elements are non-zero
     */
    [[nodiscard]] auto where() const { return range(size()).keep(*this); }

    /**
     * @brief Filter elements based on a mask array (lazy operation)
     * @param mask The mask array (elements kept where mask is non-zero)
     * @return A masked fusion_array that will filter elements when evaluated
     */
    template <typename MaskIteratorKeep>
    auto keep(const fusion_array<MaskIteratorKeep> &mask) const {
        int n = size();

        // Size check - the mask must be the same size as the array
        if (n != mask.size()) {
            throw std::invalid_argument("Mask size must match array size");
        }

        // Keep the mask array alive to prevent iterator invalidation
        auto mask_source = std::make_shared<fusion_array<MaskIteratorKeep>>(
          mask);

        return fusion_array<Iterator, MaskIteratorKeep>(
          _begin,
          _end,
          _owned_storage,
          mask_source->begin(),
          mask_source->end(),
          mask_source);  // Pass mask storage to keep it alive
    }

    /**
     * @brief Gather elements from this array using the provided indices
     * @param indices Array of indices to gather from
     * @return A new fusion_array containing the gathered elements
     */
    template <typename IndexIterator>
    auto gather(const fusion_array<IndexIterator> &indices) const {
        auto begin = thrust::make_permutation_iterator(_begin, indices.begin());
        auto end   = thrust::make_permutation_iterator(_begin, indices.end());
        return fusion_array<decltype(begin)>(begin, end);
    }

    /**
     * @brief Check if arrays match element by element (eager operation)
     * @param other The array to compare with
     * @return True if arrays match exactly, false otherwise
     */
    template <typename OtherIterator>
    auto match(const fusion_array<OtherIterator> &other) const -> bool {
        // Check if sizes match
        if (size() != other.size()) { return false; }

        // Compare elements
        return thrust::equal(_begin, _end, other.begin());
    }

    /**
     * @brief Inclusive scan with logical OR (eager operation)
     * @tparam Axis The axis along which to scan (0=all elements, 1=column-wise,
     * 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A new fusion_array containing running OR of elements
     */
    template <int Axis = 0>
    [[nodiscard]] auto anys(std::integral_constant<int, Axis> axis = {}) const {
        return scan<Axis>(thrust::logical_or<value_type>());
    }

    /**
     * @brief Inclusive scan with logical AND (eager operation)
     * @tparam Axis The axis along which to scan (0=all elements, 1=column-wise,
     * 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A new fusion_array containing running AND of elements
     */
    template <int Axis = 0>
    [[nodiscard]] auto alls(std::integral_constant<int, Axis> axis = {}) const {
        return scan<Axis>(thrust::logical_and<value_type>());
    }

    /**
     * @brief Inclusive scan with minimum (eager operation)
     * @tparam Axis The axis along which to scan (0=all elements, 1=column-wise,
     * 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A new fusion_array containing running minimum of elements
     */
    template <int Axis = 0>
    [[nodiscard]] auto mins(std::integral_constant<int, Axis> axis = {}) const {
        return scan<Axis>(thrust::minimum<value_type>());
    }

    /**
     * @brief Inclusive scan with maximum (eager operation)
     * @tparam Axis The axis along which to scan (0=all elements, 1=column-wise,
     * 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A new fusion_array containing running maximum of elements
     */
    template <int Axis = 0>
    [[nodiscard]] auto maxs(std::integral_constant<int, Axis> axis = {}) const {
        return scan<Axis>(thrust::maximum<value_type>())._mark_sorted();
    }

    /**
     * @brief Inclusive scan with addition (eager operation)
     * @tparam Axis The axis along which to scan (0=all elements, 1=column-wise,
     * 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A new fusion_array containing running sum of elements
     */
    template <int Axis = 0>
    [[nodiscard]] auto sums(std::integral_constant<int, Axis> axis = {}) const {
        return scan<Axis>(thrust::plus<value_type>());
    }

    /**
     * @brief Inclusive scan with multiplication (eager operation)
     * @tparam Axis The axis along which to scan (0=all elements, 1=column-wise,
     * 2=row-wise)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A new fusion_array containing running product of elements
     */
    template <int Axis = 0>
    [[nodiscard]] auto prods(
      std::integral_constant<int, Axis> axis = {}) const {
        return scan<Axis>(thrust::multiplies<value_type>());
    }

    /**
     * @brief Generic inclusive scan with custom binary operation (eager
     * operation)
     * @details Allows applying any custom binary scan operation to the array
     * elements. This provides flexibility beyond the predefined scans like
     * sums, prods, mins, maxs.
     * @param op The binary operation to apply for the scan (must be
     * associative)
     * @param axis Optional integral_constant parameter for axis (allows 2_ic
     * syntax)
     * @return A new fusion_array containing the running results of applying
     * the operation
     * @see sums, prods, mins, maxs, anys, alls for common predefined scans
     */
    template <int Axis = 0, typename BinaryOp>
    auto scan(BinaryOp op, std::integral_constant<int, Axis> axis = {}) const {
        if constexpr (Axis == 0) {
            int n = size();
            // Create a device vector to store the scan results
            auto
              result_vec = std::make_shared<thrust::device_vector<value_type>>(
                n, thrust::default_init);

            // Perform inclusive scan with provided binary operation
            thrust::inclusive_scan(_begin, _end, result_vec->begin(), op);

            // Return a new array with the scan results
            return fusion_array<
              typename thrust::device_vector<value_type>::iterator>(
              result_vec->begin(), result_vec->end(), result_vec, _shape);
        } else if constexpr (Axis == 1) {
            // Column-wise scan (for 2D arrays)
            if (rank() != 2) {
                throw std::runtime_error(
                  "Cannot perform column-wise scan on array with rank != 2");
            }
            // Transpose, perform row-wise scan, then transpose back
            auto transposed_array         = this->transpose();
            auto scanned_transposed_array = transposed_array.template scan<2>(
              op);
            return scanned_transposed_array.transpose();
        } else if constexpr (Axis == 2) {
            // Row-wise scan (for 2D arrays)
            if (_shape.size() < 2) {
                throw std::runtime_error(
                  "Cannot perform row-wise scan on array with rank < 2");
            }

            int num_rows = _shape[0];
            int num_cols = _shape[1];

            // Create a counting iterator and transform it to row indices
            auto count_iter  = thrust::make_counting_iterator(0);
            auto row_indices = thrust::make_transform_iterator(
              count_iter, thrust::placeholders::_1 / num_cols);

            // Allocate space for results
            auto
              result_vec = std::make_shared<thrust::device_vector<value_type>>(
                size(), thrust::default_init);

            // Perform row-wise scan using inclusive_scan_by_key
            thrust::inclusive_scan_by_key(row_indices,
                                          row_indices + size(),
                                          _begin,
                                          result_vec->begin(),
                                          thrust::equal_to<int>(),
                                          op);

            // Return result as fusion_array
            return fusion_array<
              typename thrust::device_vector<value_type>::iterator>(
              result_vec->begin(), result_vec->end(), result_vec, _shape);
        } else {
            static_assert(Axis == 0 || Axis == 1 || Axis == 2,
                          "Invalid axis value. Must be 0, 1, or 2.");
            // This will never be reached due to static_assert, but needed for
            // compilation
            return fusion_array<
              typename thrust::device_vector<value_type>::iterator>();
        }
    }

    /**
     * @brief Remove adjacent duplicate elements (lazy operation)
     * @return A masked fusion_array that will filter out adjacent duplicates
     * when evaluated
     */
    [[nodiscard]] auto uniq() const {
        return this->keep(this->differ().prepend(1));
    }

    /**
     * @brief Remove duplicate elements from the array (lazy-ish operation)
     * @details If the array is already sorted (_is_sorted is true), this method
     * applies a unique operation to remove adjacent duplicates. If the array is
     * not sorted, it first sorts the array and then removes duplicates.
     * @return A new fusion_array containing only unique elements
     */
    [[nodiscard]] auto distinct() const {
        if (_is_sorted) { return this->uniq(); }
        return this->sort().uniq();
    }

    [[nodiscard]] auto _mark_sorted() const {
        return fusion_array<Iterator>(_begin, _end, _owned_storage, true);
    }

    /**
     * @brief Run-length encode the array (eager operation)
     * @return A fusion_array of thrust::pairs with value and count
     */
    [[nodiscard]] auto rle() const {
        using pair_type = thrust::pair<value_type, int>;
        int n           = size();

        // Create a result vector to store pairs of (value, count)
        auto result_vec = std::make_shared<thrust::device_vector<pair_type>>(
          n, thrust::default_init);

        // Create a keys vector (values) and a counts vector
        auto keys = std::make_shared<thrust::device_vector<value_type>>(
          n, thrust::default_init);
        auto counts = std::make_shared<thrust::device_vector<int>>(
          n, thrust::default_init);

        // Use constant_iterator for the initial counts (all 1s)
        auto ones = thrust::make_constant_iterator(1);

        // Run-length encode using reduce_by_key
        auto new_end = thrust::reduce_by_key(
          _begin,          // Input keys begin
          _end,            // Input keys end
          ones,            // Input values begin (all 1s)
          keys->begin(),   // Output keys begin
          counts->begin()  // Output values begin
        );

        // Compute the new size after encoding
        int result_size = cuda::std::distance(keys->begin(), new_end.first);

        // Resize the output vectors
        keys->resize(result_size);
        counts->resize(result_size);
        result_vec->resize(result_size);

        // Copy the keys and counts into the result vector as pairs
        thrust::transform(thrust::make_zip_iterator(
                            thrust::make_tuple(keys->begin(), counts->begin())),
                          thrust::make_zip_iterator(
                            thrust::make_tuple(keys->end(), counts->end())),
                          result_vec->begin(),
                          tuple_to_pair_functor<value_type>());

        // Return a new fusion_array with the pairs
        return fusion_array<
          typename thrust::device_vector<pair_type>::iterator>(
          result_vec->begin(), result_vec->end(), result_vec);
    }

    /**
     * @brief Group consecutive elements by a predicate and reduce each group
     * @tparam BinPred The predicate for grouping consecutive elements
     * @tparam BinaryOp The binary operation for reduction within each group
     * @param pred The grouping predicate
     * @param binop The reduction operation
     * @return A fusion_array containing the reduced values for each group
     */
    template <typename BinPred, typename BinaryOp>
    [[nodiscard]] auto chunk_by_reduce(BinPred pred, BinaryOp binop) const {
        int n = size();

        // Create a result vector to store the reduced values
        auto result_vec = std::make_shared<thrust::device_vector<value_type>>(
          n, thrust::default_init);

        // Use reduce_by_key with the provided predicates
        auto new_end = thrust::reduce_by_key(
          _begin,
          _end,
          _begin,  // Values are the same as keys for reduction
          thrust::make_discard_iterator(),
          result_vec->begin(),
          pred,  // Grouping predicate
          binop  // Reduction operation
        );

        // Compute the new size after reduction
        int result_size = cuda::std::distance(result_vec->begin(),
                                              new_end.second);
        result_vec->resize(result_size);

        // Return a new fusion_array with the reduced values
        return fusion_array<
          typename thrust::device_vector<value_type>::iterator>(
          result_vec->begin(), result_vec->end(), result_vec);
    }

    /**
     * @brief Reshape the array to the specified dimensions
     * @param new_shape The new shape as an initializer list of dimensions
     * @return A new fusion_array with the specified shape
     * @details The total size of the new shape must be less than or equal to
     * the current array size. Some data may be truncated if the new size is
     * smaller. For cycling data to fill larger shapes, use cycle() instead.
     * @throws std::invalid_argument if total_size > size() or either size is 0
     * @see cycle
     */
    [[nodiscard]] auto reshape(std::initializer_list<int> new_shape) const {
        int total_size = 1;
        for (auto dim : new_shape) { total_size *= dim; }

        int current_size = size();

        // Throw exception if either size is 0
        if (current_size == 0 || total_size == 0) {
            throw std::invalid_argument(
              "reshape: current_size and total_size must be > 0");
        }

        if (total_size > current_size) {
            throw std::invalid_argument(
              "reshape: total_size must be <= current_size; use cycle() for "
              "larger "
              "shapes");
        }

        // Create a new array with truncated view
        return fusion_array<Iterator>(
          _begin, _begin + total_size, _owned_storage, new_shape);
    }

    /**
     * @brief Cycle the array data to fill the specified shape
     * @param new_shape The new shape as an initializer list of dimensions
     * @return A new fusion_array with the specified shape
     * @details If the new shape's total size is greater than the array size,
     * the current data is cycled to fill the new shape. If the new shape's
     * total size is less than or equal to the array size, behaves like
     * reshape().
     * @throws std::invalid_argument if either size is 0
     * @see reshape
     */
    [[nodiscard]] auto cycle(std::initializer_list<int> new_shape) const {
        int total_size = 1;
        for (auto dim : new_shape) { total_size *= dim; }

        int current_size = size();

        // Throw exception if either size is 0
        if (current_size == 0 || total_size == 0) {
            throw std::invalid_argument(
              "cycle: current_size and total_size must be > 0");
        }

        // Create cycled iterators to repeat the data as needed
        auto cycled_begin = thrustx::make_cycle_iterator(_begin, current_size);
        return fusion_array<decltype(cycled_begin)>(
          cycled_begin, cycled_begin + total_size, _owned_storage, new_shape);
    }

    // ========================================================================
    // Copying Operations (Fused/Lazy)
    // ========================================================================
    // Operations that replicate or copy elements using lazy iterators for
    // efficiency and fusion with subsequent operations.

    /**
     * @brief Repeat a scalar value to create a 1D array of the specified size
     * @param n The number of times to repeat the scalar value
     * @return A new fusion_array with n elements all equal to the scalar value
     * @details This method only works on scalar arrays (rank = 0). It repeats
     * the scalar value n times to create a 1D array. For arrays with rank > 0,
     * use cycle() instead.
     * @throws std::invalid_argument if rank != 0 or n <= 0
     * @see cycle
     */
    [[nodiscard]] auto repeat(int n) const {
        // Check if this is a scalar (rank = 0)
        if (rank() != 0) {
            throw std::invalid_argument(
              "repeat: array must be a scalar (rank = 0)");
        }

        // Throw exception if n is not positive
        if (n <= 0) { throw std::invalid_argument("repeat: n must be > 0"); }

        // Create a constant iterator to repeat the scalar value
        auto value          = *_begin;  // Get the scalar value
        auto repeated_begin = thrust::make_constant_iterator(value);

        return fusion_array<decltype(repeated_begin)>(
          repeated_begin, repeated_begin + n, nullptr);
    }

    /**
     * @brief Repeat each element of the array n times (lazy operation with
     * storage preservation)
     * @param n The number of times to repeat each element
     * @return A new fusion_array with each element repeated n times
     * @details For an array [1, 2, 3] with n=2, returns [1, 1, 2, 2, 3, 3].
     * This operation uses lazy iterators but ensures storage safety by
     * preserving the source array's storage. This is different from repeat()
     * which only works on scalars, and from the mask-based replicate() which
     * materializes.
     * @throws std::invalid_argument if n <= 0
     * @see repeat, replicate(const fusion_array<MaskIterType>&)
     */
    [[nodiscard]] auto replicate(int n) const {
        if (n <= 0) { throw std::invalid_argument("replicate: n must be > 0"); }

        if constexpr (has_mask) {
            // Apply mask first, then replicate
            auto unmasked = _apply_mask_if_needed();
            return unmasked.replicate(n);
        } else {
            int current_size = size();
            int new_size     = current_size * n;

            auto transform_begin = thrustx::make_replicate_iterator(_begin, n);

            return fusion_array<decltype(transform_begin)>(
              transform_begin, transform_begin + new_size, _owned_storage);
        }
    }

    // ========================================================================
    // Copying Operations (Materializing)
    // ========================================================================
    // Operations that replicate or copy elements and require materialization
    // due to complex access patterns that cannot be efficiently represented
    // as lazy iterators.

    /**
     * @brief Repeat each element of the array by the corresponding mask value
     * (materializing operation using gather pattern)
     * @param mask Array of integer values specifying how many times to repeat
     * each element
     * @return A new fusion_array with each element repeated according to the
     * mask
     * @details For an array [1, 2, 3] with mask [2, 1, 3], returns [1, 1, 2, 3,
     * 3, 3]. Each element at index i is repeated mask[i] times.
     * This operation requires materialization due to the variable repetition
     * pattern and uses efficient index generation and permutation iterator
     * pattern. Unlike the scalar replicate(), this cannot be efficiently
     * represented as a lazy iterator.
     * @throws std::invalid_argument if mask size doesn't match array size or
     * any mask value < 0
     * @see replicate(int)
     */
    template <typename MaskIterType>
    auto replicate(const fusion_array<MaskIterType> &mask) const
      -> fusion_array<typename thrust::device_vector<value_type>::iterator,
                      no_mask_t> {
        int current_size = size();

        // Size check - the mask must be the same size as the array
        if (current_size != mask.size()) {
            throw std::invalid_argument(
              "Mask size must match array size for replicate operation");
        }

        if constexpr (has_mask) {
            // Apply mask first, then replicate
            auto unmasked = _apply_mask_if_needed();
            return unmasked.replicate(mask);
        } else {
            // Note: mask values are expected to be non-negative integers
            // For unsigned integer types, this is guaranteed by the type system

            // Check for negative mask values
            if (mask.minr().value() < 0) {
                throw std::invalid_argument(
                  "replicate: mask values must be non-negative");
            }

            // Compute total output size
            int total_size = mask.sum().value();

            if (total_size == 0) {
                // Return empty array with the correct iterator type
                using result_iter = typename thrust::device_vector<
                  value_type>::iterator;
                auto empty_vec = std::make_shared<
                  thrust::device_vector<value_type>>();
                return fusion_array<result_iter, no_mask_t>(
                  empty_vec->end(), empty_vec->end(), empty_vec);
            }

            // Use scatter+scan pattern for efficient index generation
            // Create indices array that maps each output position to input
            // position
            auto indices_vec = std::make_shared<thrust::device_vector<int>>(
              total_size, thrust::default_init);

            // Generate cumulative sum and boundary positions
            auto mask_cumsum = mask.sums();  // [2, 3, 6] for mask [2, 1, 3]
            auto boundaries  = mask_cumsum.prepend(0);  // [0, 2, 3, 6]

            // Initialize indices array to 0
            thrust::fill(indices_vec->begin(), indices_vec->end(), 0);

            // Scatter source indices at boundary positions
            auto counting_iter = thrust::make_counting_iterator(0);
            thrust::scatter_if(
              counting_iter,  // values: [0, 1, 2, ...]
              counting_iter + current_size,
              boundaries.begin(),    // map: boundary positions [0, 2, 3, ...]
              mask.begin(),          // stencil: mask values [2, 1, 3, ...]
              indices_vec->begin(),  // output array
              thrust::placeholders::_1 > 0);

            // Fill gaps using inclusive scan with maximum
            thrust::inclusive_scan(indices_vec->begin(),
                                   indices_vec->end(),
                                   indices_vec->begin(),
                                   thrust::maximum<int>{});

            // Create result vector using permutation iterator and copy
            auto
              result_vec = std::make_shared<thrust::device_vector<value_type>>(
                total_size, thrust::default_init);
            auto perm_begin = thrust::make_permutation_iterator(
              _begin, indices_vec->begin());
            thrust::copy(
              perm_begin, perm_begin + total_size, result_vec->begin());

            using result_iter = typename thrust::device_vector<
              value_type>::iterator;
            using result_type = fusion_array<result_iter, no_mask_t>;
            return result_type{
              result_vec->begin(), result_vec->end(), result_vec};
        }
    }

    /**
     * @brief Compute the cartesian product with another array (lazy operation)
     * @param other The other array to compute the cartesian product with
     * @return A new fusion_array containing pairs representing the cartesian
     * product
     * @details For arrays [1, 2] and [a, b], returns [(1, a), (1, b), (2, a),
     * (2, b)]. The implementation replicates each element of this array by the
     * size of the other array, and cycles the other array to match the total
     * size, then pairs them together.
     * @throws std::invalid_argument if either array is empty
     * @see replicate, cycle, pairs
     */
    template <typename OtherIterator>
    auto cross(const fusion_array<OtherIterator> &other) const {
        int this_size  = size();
        int other_size = other.size();

        if (this_size == 0 || other_size == 0) {
            throw std::invalid_argument("cross: arrays must not be empty");
        }

        return this->replicate(other_size)
          .pairs(other.cycle({this_size * other_size}));
    }

    /**
     * @brief Compute the outer product with another array as a 2D matrix (lazy
     * operation)
     * @param other The other array to compute the outer product with
     * @param binary_op The binary operation to apply to each pair of elements
     * @return A new fusion_array containing the results in a 2D matrix shape
     * @details For arrays [1, 2] and [a, b] with operation `op`, returns a 2×2
     * matrix: [op(1, a), op(1, b)] [op(2, a), op(2, b)] This is much more
     * efficient than the cycle-based approach.
     * @throws std::invalid_argument if either array is empty
     * @see cross, reshape, map
     */
    template <typename OtherIterator, typename BinaryOp>
    auto outer(const fusion_array<OtherIterator> &other,
               BinaryOp binary_op) const {
        int this_size  = size();
        int other_size = other.size();

        if (this_size == 0 || other_size == 0) {
            throw std::invalid_argument("outer: arrays must not be empty");
        }

        auto outer_iter = thrustx::make_outer_iterator(
          _begin, other.begin(), this_size, other_size, binary_op);

        return fusion_array<decltype(outer_iter)>(
          outer_iter,
          outer_iter + (this_size * other_size),
          _owned_storage,
          {this_size, other_size});
    }

    /**
     * @brief Print the array contents to a stream
     * @param os The output stream (defaults to std::cout)
     * @param delimiter The string to print between elements (defaults to space)
     * @return For masked arrays, returns the unmasked array; otherwise returns
     * a reference to this array
     */
    auto print(std::ostream &os      = std::cout,
               const char *delimiter = " ") const {
        if constexpr (has_mask) {
            // Apply mask first then print
            auto unmasked = _apply_mask_if_needed();
            unmasked.print(os, delimiter);
            return unmasked;
        } else {
            int n = size();

            if (_shape.size() <= 1) {
                // 1D array or scalar
                int max_width = 1;
                // Calculate max width first
                for (auto it = _begin; it != _end; ++it) {
                    std::stringstream ss;
                    if constexpr (is_thrust_pair_v<value_type>) {
                        // Explicitly copy device_reference to host pair
                        using pair_type = typename cuda::std::iterator_traits<
                          Iterator>::value_type;
                        thrust::pair<typename pair_type::first_type,
                                     typename pair_type::second_type>
                          host_pair = *it;
                        ss << "(" << host_pair.first << ", " << host_pair.second
                           << ")";
                    } else {
                        ss << *it;  // Implicit copy to host for other types
                    }
                    max_width = std::max(max_width,
                                         static_cast<int>(ss.str().length()));
                }

                // Now print with padding
                if (n > 0) {
                    std::stringstream ss_first;
                    if constexpr (is_thrust_pair_v<value_type>) {
                        // Explicitly copy device_reference to host pair
                        using pair_type = typename cuda::std::iterator_traits<
                          Iterator>::value_type;
                        thrust::pair<typename pair_type::first_type,
                                     typename pair_type::second_type>
                          host_pair = *_begin;
                        ss_first << "(" << host_pair.first << ", "
                                 << host_pair.second << ")";
                    } else {
                        ss_first << *_begin;
                    }
                    os << std::setw(max_width) << ss_first.str();

                    for (auto it = _begin + 1; it != _end; ++it) {
                        std::stringstream ss_rest;
                        if constexpr (is_thrust_pair_v<value_type>) {
                            // Explicitly copy device_reference to host pair
                            using pair_type = typename cuda::std::
                              iterator_traits<Iterator>::value_type;
                            thrust::pair<typename pair_type::first_type,
                                         typename pair_type::second_type>
                              host_pair = *it;
                            ss_rest << "(" << host_pair.first << ", "
                                    << host_pair.second << ")";
                        } else {
                            ss_rest << *it;
                        }
                        os << delimiter << std::setw(max_width)
                           << ss_rest.str();
                    }
                }
                os << '\n';
            } else {
                // Multi-dimensional array
                int outer_dim  = _shape[0];
                int inner_size = 1;
                for (size_t i = 1; i < _shape.size(); i++) {
                    inner_size *= _shape[i];
                }

                // Calculate the maximum width for pretty printing
                int max_width = 1;
                for (auto it = _begin; it != _end; ++it) {
                    std::stringstream ss;
                    if constexpr (is_thrust_pair_v<value_type>) {
                        // Explicitly copy device_reference to host pair
                        using pair_type = typename cuda::std::iterator_traits<
                          Iterator>::value_type;
                        thrust::pair<typename pair_type::first_type,
                                     typename pair_type::second_type>
                          host_pair = *it;
                        ss << "(" << host_pair.first << ", " << host_pair.second
                           << ")";
                    } else {
                        // Handle potential non-string-convertible types
                        // gracefully? For now, assume std::to_string works or
                        // similar stream insertion
                        ss << *it;  // Implicit copy to host for other types
                    }
                    auto width = static_cast<int>(ss.str().length());
                    max_width  = std::max(max_width, width);
                }

                for (int i = 0; i < outer_dim; i++) {
                    // Print each "row"
                    int row_start = i * inner_size;
                    if (inner_size > 0) {
                        std::stringstream ss_first;
                        if constexpr (is_thrust_pair_v<value_type>) {
                            // Explicitly copy device_reference to host pair
                            using pair_type = typename cuda::std::
                              iterator_traits<Iterator>::value_type;
                            thrust::pair<typename pair_type::first_type,
                                         typename pair_type::second_type>
                              host_pair = *(_begin + row_start);
                            ss_first << "(" << host_pair.first << ", "
                                     << host_pair.second << ")";
                        } else {
                            ss_first << *(_begin + row_start);
                        }
                        os << std::setw(max_width) << ss_first.str();

                        for (int j = 1; j < inner_size; j++) {
                            std::stringstream ss_rest;
                            if constexpr (is_thrust_pair_v<value_type>) {
                                // Explicitly copy device_reference to host pair
                                using pair_type = typename cuda::std::
                                  iterator_traits<Iterator>::value_type;
                                thrust::pair<typename pair_type::first_type,
                                             typename pair_type::second_type>
                                  host_pair = *(_begin + row_start + j);
                                ss_rest << "(" << host_pair.first << ", "
                                        << host_pair.second << ")";
                            } else {
                                ss_rest << *(_begin + row_start + j);
                            }
                            os << delimiter << std::setw(max_width)
                               << ss_rest.str();
                        }
                    }
                    os << '\n';
                }
            }

            return *this;
        }
    }

    /**
     * @brief Find the minimum and maximum values in the array (eager
     * operation)
     * @return A fusion_array containing a thrust::pair with first=minimum
     * and second=maximum
     */
    [[nodiscard]] auto minmax() const {
        auto unary_op  = minmax_unary_op<value_type>();
        auto binary_op = minmax_binary_op<value_type>();
        auto init      = unary_op(*_begin);

        return this->map(unary_op).reduce(init, binary_op);
    }

    /**
     * @brief Take the first n elements of the array
     * @param n The number of elements to take
     * @return A new fusion_array containing only the first n elements
     * @throws std::invalid_argument if n > size() or n < 0
     */
    [[nodiscard]] auto take(int n) const {
        if (n < 0 || n > size()) {
            throw std::invalid_argument(
              "take: n must be between 0 and size() inclusive");
        }

        return fusion_array<Iterator>(_begin, _begin + n, _owned_storage);
    }

    /**
     * @brief Drop the first n elements of the array
     * @param n The number of elements to drop
     * @return A new fusion_array containing all elements except the first
     * n
     * @throws std::invalid_argument if n > size() or n < 0
     */
    [[nodiscard]] auto drop(int n) const {
        if (n < 0 || n > size()) {
            throw std::invalid_argument(
              "drop: n must be between 0 and size() inclusive");
        }
        if (n == size()) {  // if dropping all elements, return empty array
            return fusion_array<Iterator>(_end, _end, _owned_storage);
        }
        return fusion_array<Iterator>(_begin + n, _end, _owned_storage);
    }

    /**
     * @brief Flatten the array to rank 1 (modifies shape only, no data copy)
     * @return A new fusion_array with the same data but flattened to 1D
     */
    [[nodiscard]] auto flatten() const {
        std::vector<int> flat_shape = {size()};
        return fusion_array<Iterator>(_begin, _end, _owned_storage, flat_shape);
    }

    /**
     * @brief Get a copy of a specific row (with data copy)
     * @param row_idx The row index to extract
     * @return A fusion_array copy of the specified row
     * @throws std::runtime_error if array rank != 2 or row_idx out of bounds
     */
    [[nodiscard]] auto row(int row_idx) const {
        if (rank() != 2) {
            throw std::runtime_error("row() only works on rank 2 arrays");
        }

        int num_rows = _shape[0];
        int num_cols = _shape[1];

        if (row_idx < 0 || row_idx >= num_rows) {
            throw std::invalid_argument("row index out of bounds");
        }

        return this->flatten().drop(row_idx * num_cols).take(num_cols);
    }

    /**
     * @brief Find the first index of a value in the array (0-based)
     * @param value The value to search for
     * @return The 0-based index of the first occurrence, or -1 if not found
     * @throws std::runtime_error if array rank != 1
     */
    template <typename T>
    auto index_of(const T &value) const -> int {
        if (rank() != 1) {
            throw std::runtime_error("index_of() only works on rank 1 arrays");
        }

        if constexpr (has_mask) {
            // Apply mask first, then search
            auto unmasked = _apply_mask_if_needed();
            return unmasked.index_of(value);
        } else {
            auto it = thrust::find(_begin, _end, value);
            if (it == _end) {
                return -1;  // Not found
            }
            return cuda::std::distance(_begin, it);
        }
    }

    /**
     * @brief Find the last index of a value in the array (0-based)
     * @param value The value to search for
     * @return The 0-based index of the last occurrence, or -1 if not found
     * @throws std::runtime_error if array rank != 1
     */
    template <typename T>
    auto last_index_of(const T &value) const -> int {
        if (rank() != 1) {
            throw std::runtime_error(
              "last_index_of() only works on rank 1 arrays");
        }

        if constexpr (has_mask) {
            // Apply mask first, then search
            auto unmasked = _apply_mask_if_needed();
            return unmasked.last_index_of(value);
        } else {
            // Search from the end using reverse iterators
            auto rbegin = thrust::make_reverse_iterator(_end);
            auto rend   = thrust::make_reverse_iterator(_begin);
            auto it     = thrust::find(rbegin, rend, value);

            if (it == rend) {
                return -1;  // Not found
            }

            // Convert reverse iterator position to forward index
            return cuda::std::distance(_begin, it.base()) - 1;
        }
    }

    /**
     * @brief Find the maximum element based on key extractor
     * @tparam KeyExtractor The type of the key extractor functor
     * @param key_extractor A functor that extracts a key for comparison
     * from each element
     * @return A fusion_array containing the maximum element based on the
     * extracted key
     */
    template <typename KeyExtractor>
    auto max_by_key(KeyExtractor key_extractor) const
      -> fusion_array<typename thrust::device_vector<
        typename cuda::std::iterator_traits<Iterator>::value_type>::iterator> {
        int n = size();

        if (n == 0) {
            // Return an empty array for empty input
            return fusion_array<Iterator>(_begin, _begin, _owned_storage);
        }

        // Create a comparator that uses the key extractor
        auto key_comp = [key_extractor] __host__ __device__(
                          const value_type &a, const value_type &b) {
            return key_extractor(a) < key_extractor(b);
        };

        // Use thrust::max_element with the custom comparator
        auto max_iter = thrust::max_element(_begin, _end, key_comp);

        // Create a device vector with the maximum element directly
        auto result_vec = std::make_shared<thrust::device_vector<value_type>>(
          1, *max_iter);

        // Return a new fusion_array with the maximum element
        return fusion_array<
          typename thrust::device_vector<value_type>::iterator>(
          result_vec->begin(), result_vec->end(), result_vec);
    }

    /**
     * @brief Create pairs from two arrays of the same length
     * @tparam OtherIterator The iterator type of the second array
     * @param other The second array to pair with this array
     * @return A new fusion_array containing pairs of elements from both
     * arrays
     * @throws std::invalid_argument if arrays have different sizes
     */
    template <typename OtherIterator>
    auto pairs(const fusion_array<OtherIterator> &other) const {
        using first_type = typename cuda::std::iterator_traits<
          Iterator>::value_type;
        using second_type = typename cuda::std::iterator_traits<
          OtherIterator>::value_type;
        using pair_type = thrust::pair<first_type, second_type>;

        // Check if arrays have the same size
        if (size() != other.size()) {
            throw std::invalid_argument(
              "Arrays must have the same size for pairs operation");
        }

        // Create a zip iterator combining elements from both arrays
        auto zip_begin = thrust::make_zip_iterator(
          thrust::make_tuple(_begin, other.begin()));
        auto zip_end = thrust::make_zip_iterator(
          thrust::make_tuple(_end, other.end()));

        // Define the functor type for making pairs
        using make_pair_func = make_pair_functor<first_type, second_type>;

        // Transform the zipped iterators into pairs
        return fusion_array<
          thrust::transform_iterator<make_pair_func, decltype(zip_begin)>>(
          thrust::make_transform_iterator(zip_begin, make_pair_func()),
          thrust::make_transform_iterator(zip_end, make_pair_func()),
          _owned_storage,
          _shape);
    }

    /**
     * @brief Create pairs from array elements and their indices (lazy
     * operation)
     * @return A new fusion_array containing pairs of (element, index)
     * @details Creates pairs of (element, index) starting from index 0.
     * For array [a, b, c], returns [(a, 0), (b, 1), (c, 2)].
     * Useful for operations that need to track element positions.
     */
    [[nodiscard]] auto enumerate() const { return this->pairs(range(size())); }

    /**
     * @brief Reverse the array (lazy operation)
     * @return A new fusion_array with elements in reverse order
     */
    [[nodiscard]] auto rev() const {
        auto rbegin = thrust::make_reverse_iterator(_end);
        auto rend   = thrust::make_reverse_iterator(_begin);
        return fusion_array<decltype(rbegin)>(
          rbegin, rend, _owned_storage, _shape);
    }

    /**
     * @brief Filter elements based on a predicate (lazy operation)
     * @param predicate A unary predicate that returns true for elements to
     * keep
     * @return A masked fusion_array that will filter elements when
     * evaluated
     */
    template <typename Predicate>
    auto filter(Predicate predicate) const {
        // Create a transform_iterator that applies the predicate
        auto mask_begin = thrust::make_transform_iterator(_begin, predicate);
        auto mask_end   = thrust::make_transform_iterator(_end, predicate);

        // Use existing keep function with the mask iterator
        return keep(fusion_array<decltype(mask_begin)>(
          mask_begin, mask_end, _owned_storage));
    }

    /**
     * @brief Transpose a 2D array (lazy operation)
     * @return A new fusion_array representing the transpose of the
     * original
     * @throws std::invalid_argument if the array is not rank 2
     */
    [[nodiscard]] auto transpose() const {
        if (rank() != 2) {
            throw std::invalid_argument(
              "transpose: array must be rank 2 (a matrix)");
        }

        int num_rows   = _shape[0];
        int num_cols   = _shape[1];
        int total_size = num_rows * num_cols;

        // Create a counting iterator for the indices of the transposed
        // array
        auto counting_iter = thrust::make_counting_iterator(0);

        // Create a transform iterator that applies the transpose_functor
        // This transform_iterator will yield the indices in the *original*
        // array that correspond to the elements of the transposed array.
        auto index_map_iter = thrust::make_transform_iterator(
          counting_iter, transpose_functor(num_rows, num_cols));

        // Create a permutation iterator
        auto perm_iter_begin = thrust::make_permutation_iterator(
          _begin, index_map_iter);
        auto perm_iter_end = perm_iter_begin + total_size;

        // New shape for the transposed array
        std::vector<int> new_shape = {num_cols, num_rows};

        return fusion_array<decltype(perm_iter_begin)>(
          perm_iter_begin, perm_iter_end, _owned_storage, new_shape);
    }

    // clang-format off
     auto operator/ (auto const &arg) const { return this->div(arg);   }
     auto operator- (auto const &arg) const { return this->minus(arg); }
     auto operator+ (auto const &arg) const { return this->add(arg);   }
     auto operator* (auto const &arg) const { return this->times(arg); }
     auto operator< (auto const &arg) const { return this->lt(arg);    }
     auto operator<=(auto const &arg) const { return this->lte(arg);   }
     auto operator> (auto const &arg) const { return this->gt(arg);    }
     auto operator>=(auto const &arg) const { return this->gte(arg);   }
     auto operator==(auto const &arg) const { return this->eq(arg);    }
     // auto operator%(auto const &arg) const { return this->mod(arg); } // TODO
    // clang-format on
};

/**
 * @brief Create a 1-indexed range array from 1 to n
 * @param end The upper bound of the range (inclusive)
 * @return A fusion_array containing integers from 1 to end
 * @throws std::invalid_argument if end <= 0
 */
inline auto range(int end) -> fusion_array<thrust::counting_iterator<int>> {
    if (end <= 0) { throw std::invalid_argument("range: end must be > 0"); }

    return {thrust::counting_iterator<int>(1),
            thrust::counting_iterator<int>(end + 1)};
}

/**
 * @brief Create a fusion_array from a std::vector
 * @tparam T The element type of the vector
 * @param host_vec The host vector to copy data from
 * @return A fusion_array containing the vector's data on the device
 */
template <typename T>
auto array(const std::vector<T> &host_vec) {
    // Create and own a device vector
    auto device_vec = std::make_shared<thrust::device_vector<T>>(
      host_vec.begin(), host_vec.end());

    return fusion_array<typename thrust::device_vector<T>::iterator>(
      device_vec->begin(), device_vec->end(), device_vec);
    // Note: _shape is already initialized by the constructor to {size}
}

/**
 * @brief Create a fusion_array from an initializer list with automatic type
 * deduction
 * @tparam T The element type (automatically deduced)
 * @param init_list The initializer list containing the array elements
 * @return A fusion_array containing the initializer list's data on the device
 */
template <typename T>
auto array(std::initializer_list<T> init_list) {
    // Convert initializer list to a device vector
    auto device_vec = std::make_shared<thrust::device_vector<T>>(
      init_list.begin(), init_list.end());

    return fusion_array<typename thrust::device_vector<T>::iterator>(
      device_vec->begin(), device_vec->end(), device_vec);
}

/**
 * @brief Create a scalar (rank 0) array containing a single value
 * @tparam T The element type (automatically deduced)
 * @param value The scalar value to store in the array
 * @return A fusion_array with rank 0 containing the single value
 */
template <typename T>
auto scalar(T value) {
    return fusion_array<thrust::constant_iterator<T>>(value);
}

/**
 * @brief Create a multi-dimensional array filled with a constant value
 * @tparam T The element type (automatically deduced)
 * @param value The constant value to fill the matrix with
 * @param shape The shape as an initializer list {rows, cols}
 * @return A fusion_array with the specified shape filled with the constant
 * value
 * @throws std::invalid_argument if shape doesn't have exactly 2 dimensions
 */
template <typename T>
auto matrix(T value, std::initializer_list<int> shape) {
    if (shape.size() != 2) {
        throw std::invalid_argument(
          "matrix: shape must have exactly 2 dimensions");
    }

    auto total_size = 1;
    for (auto dim : shape) { total_size *= dim; }

    return scalar(value).repeat(total_size).reshape(shape);
}

/**
 * @brief Create a 2D matrix from a nested initializer list
 * @tparam T The element type (automatically deduced from the nested list)
 * @param nested_list A nested initializer list where each inner list represents
 * a row
 * @return A fusion_array with shape {rows, cols} containing the matrix data
 * @throws std::invalid_argument if nested_list is empty or inner lists have
 * different lengths
 */
template <typename T>
auto matrix(std::initializer_list<std::initializer_list<T>> nested_list) {
    if (nested_list.size() == 0) {
        throw std::invalid_argument("matrix: nested_list cannot be empty");
    }

    // Get dimensions
    int rows = static_cast<int>(nested_list.size());
    int cols = static_cast<int>(nested_list.begin()->size());

    if (cols == 0) {
        throw std::invalid_argument("matrix: inner lists cannot be empty");
    }

    // Validate that all rows have the same length
    for (const auto &row : nested_list) {
        if (static_cast<int>(row.size()) != cols) {
            throw std::invalid_argument(
              "matrix: all inner lists must have the same length");
        }
    }

    // Flatten the nested data into a single vector (row-major order)
    std::vector<T> flattened_data;
    flattened_data.reserve(rows * cols);

    for (const auto &row : nested_list) {
        for (const auto &element : row) { flattened_data.push_back(element); }
    }

    // Create the array and reshape it to the matrix dimensions
    return array(flattened_data).reshape({rows, cols});
}

// Define the stats namespace implementation
namespace stats {
// Normal CDF functor
template <typename T>
struct norm_cdf_functor {
    __host__ __device__ auto operator()(const T &x) const -> T {
        return normcdff(x);
    }
};

// Standard normal cumulative distribution function
template <typename Iterator, typename MaskIterator>
auto norm_cdf(const fusion_array<Iterator, MaskIterator> &arr) {
    using value_type = typename cuda::std::iterator_traits<
      Iterator>::value_type;
    using NormCDFFunc = norm_cdf_functor<value_type>;
    return arr.map(NormCDFFunc());
}

// Statistical mode function - returns the most frequently occurring value
template <typename Iterator, typename MaskIterator>
auto mode(const fusion_array<Iterator, MaskIterator> &arr) {
    using value_type = typename cuda::std::iterator_traits<
      Iterator>::value_type;

    // Find the mode by sorting, run-length encoding, and finding max by count
    auto mode_value = arr.sort().rle().max_by_key(snd()).value().first;

    // Return as a scalar fusion_array
    return fusion_array<thrust::constant_iterator<value_type>>(mode_value);
}
}  // namespace stats

}  // namespace parrot

#endif  // PARROT_HPP
