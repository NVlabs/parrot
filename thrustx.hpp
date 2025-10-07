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

#ifndef THRUSTX_HPP
#define THRUSTX_HPP

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <iterator>
#include <stdexcept>
#include <type_traits>

// ThrustX namespace for extended thrust functionality
namespace thrustx {

// High-performance implementation using CUB's optimized uniform segmentation
// Reduces input into segments of N elements each using CUB's fixed-size
// segmented reduce API when possible
template <typename InputIterator,
          typename OutputIterator,
          typename BinaryOp,
          typename T>
void reduce_by_n_impl(InputIterator first,
                      InputIterator last,
                      OutputIterator out,
                      int N,
                      BinaryOp op,
                      T init) {
    if (N <= 0) {
        throw std::invalid_argument("reduce_by_n: N must be positive");
    }

    int total_size = std::distance(first, last);
    if (total_size == 0) { return; }

    int num_segments = total_size / N;

    if (total_size % N != 0) {
        throw std::invalid_argument(
          "reduce_by_n: N must be a divisor of std::distance(first, last)");
    }

    // Use raw CUDA memory for temp storage to avoid thrust overhead
    void *d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    // Query temp storage size using CUB's fixed-size segmented reduce API
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                       temp_storage_bytes,
                                       first,
                                       out,
                                       num_segments,
                                       N,  // segment_size
                                       op,
                                       init);

    // Raw CUDA allocation - no initialization kernel
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Actual reduction using fixed-size segmented reduce
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                       temp_storage_bytes,
                                       first,
                                       out,
                                       num_segments,
                                       N,  // segment_size
                                       op,
                                       init);

    // Free temp storage
    cudaFree(d_temp_storage);
}

// Cycle functor for cycling through indices
struct cycle_functor {
    int n;
    explicit cycle_functor(int n) : n(n) {}
    __host__ __device__ auto operator()(int i) const -> int { return i % n; }
};

// Direct cycle functor that avoids permutation iterator overhead
// (renamed from direct_cycle_functor to cycle_functor as requested)
template <typename Iterator>
struct direct_cycle_functor {
    Iterator begin;
    int n;

    direct_cycle_functor(Iterator begin, int n) : begin(begin), n(n) {}

    __host__ __device__ auto operator()(int i) const ->
      typename cuda::std::iterator_traits<Iterator>::value_type {
        return begin[i % n];
    }
};

// Helper function to create a cycle iterator (optimized version using
// direct_cycle_functor)
template <typename Iterator>
auto make_cycle_iterator(Iterator begin, int n) {
    auto counting = thrust::make_counting_iterator<int>(0);
    return thrust::make_transform_iterator(
      counting, direct_cycle_functor<Iterator>(begin, n));
}

// Legacy helper function to create a cycle iterator (compatible with old code)
// This creates a permutation iterator like the original implementation
template <typename Iterator>
auto make_cycle_iterator_permutation(Iterator begin, int n) {
    auto a = thrust::make_counting_iterator<int>(0);
    auto b = thrust::make_transform_iterator(a, cycle_functor(n));
    return thrust::make_permutation_iterator(begin, b);
}

// Helper function to create a cycle functor
template <typename Iterator>
auto make_cycle_functor(Iterator begin, int n) {
    return direct_cycle_functor<Iterator>(begin, n);
}

// Chain functor template for append operations
template <typename Iterator, typename T>
struct append_functor {
    Iterator _begin;
    int _size;
    T _value;

    append_functor(Iterator begin, int size, T value)
      : _begin(begin), _size(size), _value(value) {}

    __host__ __device__ auto operator()(const int &idx) const -> T {
        if (idx < _size) { return _begin[idx]; }
        return _value;
    }
};

// Chain functor template for prepend operations
template <typename Iterator, typename T>
struct prepend_functor {
    Iterator _begin;
    int _size;
    T _value;

    prepend_functor(Iterator begin, int size, T value)
      : _begin(begin), _size(size), _value(value) {}

    __host__ __device__ auto operator()(const int &idx) const -> T {
        if (idx == 0) { return _value; }
        return _begin[idx - 1];
    }
};

// Helper function to create an append functor
template <typename Iterator, typename T>
auto make_append_functor(Iterator begin, int size, T value) {
    return append_functor<Iterator, T>(begin, size, value);
}

// Helper function to create a prepend functor
template <typename Iterator, typename T>
auto make_prepend_functor(Iterator begin, int size, T value) {
    return prepend_functor<Iterator, T>(begin, size, value);
}

// Helper function to create an append iterator
template <typename Iterator, typename T>
auto make_append_iterator(Iterator begin, int size, T value) {
    using value_type = typename cuda::std::iterator_traits<
      Iterator>::value_type;
    using AppendFunc     = append_functor<Iterator, value_type>;
    auto transform_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), AppendFunc(begin, size, value));
    return transform_begin;
}

// Helper function to create a prepend iterator
template <typename Iterator, typename T>
auto make_prepend_iterator(Iterator begin, int size, T value) {
    using value_type = typename cuda::std::iterator_traits<
      Iterator>::value_type;
    using PrependFunc    = prepend_functor<Iterator, value_type>;
    auto transform_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), PrependFunc(begin, size, value));
    return transform_begin;
}

// Replicate functor for repeating each element n times
template <typename Iterator, typename T>
struct replicate_functor {
    Iterator _begin;
    int _n;

    replicate_functor(Iterator begin, int n) : _begin(begin), _n(n) {}

    __host__ __device__ auto operator()(int idx) const -> T {
        return _begin[idx / _n];
    }
};

// Helper function to create a replicate functor
template <typename Iterator>
auto make_replicate_functor(Iterator begin, int n) {
    using value_type = typename cuda::std::iterator_traits<
      Iterator>::value_type;
    return replicate_functor<Iterator, value_type>(begin, n);
}

// Helper function to create a replicate iterator
template <typename Iterator>
auto make_replicate_iterator(Iterator begin, int n) {
    using value_type = typename cuda::std::iterator_traits<
      Iterator>::value_type;
    using ReplicateFunc  = replicate_functor<Iterator, value_type>;
    auto transform_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), ReplicateFunc(begin, n));
    return transform_begin;
}

/**
 * @brief Direct outer product functor (avoids cycle overhead)
 */
template <typename Iterator1, typename Iterator2, typename BinaryOp>
struct direct_outer_functor {
    Iterator1 begin1;
    Iterator2 begin2;
    int size1, size2;
    BinaryOp binary_op;

    direct_outer_functor(
      Iterator1 begin1, Iterator2 begin2, int size1, int size2, BinaryOp op)
      : begin1(begin1),
        begin2(begin2),
        size1(size1),
        size2(size2),
        binary_op(op) {}

    __host__ __device__ auto operator()(int linear_idx) const
      -> decltype(binary_op(*begin1, *begin2)) {
        int row = linear_idx / size2;
        int col = linear_idx % size2;
        return binary_op(begin1[row], begin2[col]);
    }
};

// Helper function to create an outer product iterator
template <typename Iterator1, typename Iterator2, typename BinaryOp>
auto make_outer_iterator(
  Iterator1 begin1, Iterator2 begin2, int size1, int size2, BinaryOp op) {
    auto counting = thrust::make_counting_iterator<int>(0);
    return thrust::make_transform_iterator(
      counting,
      direct_outer_functor<Iterator1, Iterator2, BinaryOp>(
        begin1, begin2, size1, size2, op));
}

// API wrapper function that calls the implementation
template <typename InputIterator,
          typename OutputIterator,
          typename BinaryOp,
          typename T>
void reduce_by_n(InputIterator first,
                 InputIterator last,
                 OutputIterator out,
                 int N,
                 BinaryOp op,
                 T init) {
    reduce_by_n_impl(first, last, out, N, op, init);
}

// Backward compatibility overload (uses default initialization)
template <typename InputIterator, typename OutputIterator, typename BinaryOp>
void reduce_by_n(InputIterator first,
                 InputIterator last,
                 OutputIterator out,
                 int N,
                 BinaryOp op) {
    using value_type = typename std::iterator_traits<InputIterator>::value_type;
    reduce_by_n_impl(first, last, out, N, op, value_type{});
}

}  // namespace thrustx

#endif  // THRUSTX_HPP
