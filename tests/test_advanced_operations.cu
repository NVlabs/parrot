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

#include <stdexcept>
#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_common.hpp"

TEST_CASE("Repeated filters retain masks and short circuit rejected values") {
    test_detail::evaluation_counters counters;
    auto values    = parrot::array({0, 2, 0, 4});
    auto mask      = parrot::array({0, 2, 0, -1})
                       .map(test_detail::observed_mask{counters.data()});
    auto observe   = test_detail::observed_value{counters.data()};
    auto predicate = [observe] __host__ __device__(int value) {
        return observe(value) >= 30;
    };
    auto result = values.keep(mask).filter(predicate).filter(
      [] __host__ __device__(int value) { return value < 3; });

    CHECK_EQ(counters.get(0), 0);
    CHECK_EQ(counters.get(2), 0);
    CHECK(result.storage() == values.storage());
    CHECK(result.to_host() == std::vector<int>{2});
    CHECK_GT(counters.get(0), 0);
    CHECK_EQ(counters.get(1), 0);
    CHECK_EQ(result.rank(), 1);
    CHECK(result.shape() == std::vector<int>{1});
    counters.reset();
    CHECK_EQ(result.sum().value(), 2);
    CHECK_GT(counters.get(0), 0);
    CHECK_EQ(counters.get(1), 0);
    CHECK(result.where().to_host() == std::vector<int>{1});
    CHECK_EQ(counters.get(1), 0);
}

TEST_CASE(
  "Repeated filters retain property and nonbinary selection semantics") {
    auto result = parrot::range(6)
                    .filter(
                      [] __host__ __device__(int x) { return x % 2 == 0; })
                    .filter([] __host__ __device__(int x) { return x < 6; });
    using props = typename decltype(result)::properties_type;
    static_assert(parrot::is_strict_v<props>);
    static_assert(parrot::is_ascending_v<props>);
    static_assert(!parrot::is_nonempty_v<props>);
    CHECK(result.to_host() == std::vector<int>{2, 4});
    CHECK(result.where().to_host() == std::vector<int>{1, 2});

    auto nonbinary = parrot::array({0, 2, 3, 4})
                       .keep(parrot::array({0, 2, -1, 0}))
                       .filter([] __host__ __device__(int x) {
                           return x == 2 ? 0 : -3;
                       });
    CHECK(nonbinary.to_host() == std::vector<int>{3});
    CHECK_EQ(nonbinary.sum().value(), 3);

    auto bytes = parrot::array(std::vector<std::uint8_t>{200, 100, 7})
                   .keep(parrot::array({1, 1, 0}))
                   .map([] __host__ __device__(std::uint8_t x) { return x; })
                   .filter(
                     [] __host__ __device__(std::uint8_t) { return true; });
    static_assert(
      std::is_same_v<typename decltype(bytes.sum())::value_type, int>);
    CHECK_EQ(bytes.sum().value(), 300);
}

TEST_CASE("Repeated filters handle empty inputs and retain temporary owners") {
    test_detail::evaluation_counters counters;
    auto observe   = test_detail::observed_value{counters.data()};
    auto predicate = [observe] __host__ __device__(int x) {
        return observe(x) > 0;
    };
    auto none = parrot::array({0, 0})
                  .keep(parrot::array({0, 0}))
                  .filter(predicate);
    CHECK(none.to_host().empty());
    CHECK_EQ(none.sum().value(), 0);
    CHECK_EQ(counters.get(0), 0);

    auto empty = parrot::array(std::vector<int>{})
                   .keep(parrot::array(std::vector<int>{}))
                   .filter(predicate);
    CHECK(empty.to_host().empty());
    CHECK_EQ(empty.sum().value(), 0);
    auto temporary = [] {
        int threshold = 1;
        return parrot::array({1, 2, 3, 4})
          .filter(
            [threshold] __host__ __device__(int x) { return x > threshold; })
          .filter([] __host__ __device__(int x) { return x % 2 == 0; });
    }();
    CHECK(temporary.to_host() == std::vector<int>{2, 4});
}

TEST_CASE("Filters compose with mapped scalar and reversed selections") {
    auto result = parrot::range(8)
                    .keep(parrot::array({0, 1, 0, 2, 1, 0, -1, 0}))
                    .map([] __host__ __device__(int x) { return x * 2; })
                    .add(1)
                    .filter([] __host__ __device__(int x) { return x > 8; })
                    .rev()
                    .map([] __host__ __device__(int x) { return x - 1; });
    CHECK(result.to_host() == std::vector<int>{14, 10, 8});
    CHECK_EQ(result.sum().value(), 32);
    CHECK_EQ(result.front(), 14);
    CHECK_EQ(result.back(), 8);
    CHECK(result.shape() == std::vector<int>{3});

    auto bytes = parrot::array(std::vector<std::uint8_t>{200, 100, 7})
                   .keep(parrot::array({1, 1, 0}))
                   .map([] __host__ __device__(std::uint8_t x) { return x; })
                   .rev()
                   .filter(
                     [] __host__ __device__(std::uint8_t) { return true; })
                   .map([] __host__ __device__(std::uint8_t x) { return x; });
    static_assert(
      std::is_same_v<typename decltype(bytes.sum())::value_type, std::uint8_t>);
    CHECK_EQ(bytes.sum().value(), 44);
}

// Test where function with all non-zeros
TEST_CASE("ParrotTest - WhereAllTest") {
    auto arr     = parrot::array({1, 1, 1, 1});
    auto indices = arr.where();
    CHECK_EQ(indices.size(), 4);
    auto expected = parrot::array({1, 2, 3, 4});
    CHECK(check_match(indices, expected));  // Use helper for comparison
}

// Test where function with some zeros
TEST_CASE("ParrotTest - WhereSomeTest") {
    auto arr     = parrot::array({0, 1, 0, 1});
    auto indices = arr.where();
    CHECK_EQ(indices.size(), 2);
    auto expected = parrot::array({2, 4});
    CHECK(check_match(indices, expected));  // Use helper for comparison
}

// Test where function with all zeros
TEST_CASE("ParrotTest - WhereNoneTest") {
    auto arr     = parrot::array({0, 0, 0, 0});
    auto indices = arr.where();
    CHECK_EQ(indices.size(), 0);
}

// Test keep function with simple mask
TEST_CASE("ParrotTest - KeepTest") {
    auto arr      = parrot::array({1, 2, 3});
    auto mask     = parrot::array({1, 0, 1});
    auto result   = arr.keep(mask);
    auto expected = parrot::array({1, 3});
    CHECK(check_match(result, expected));
}

// Test keep function with all zeros mask
TEST_CASE("ParrotTest - KeepAllZerosTest") {
    auto arr    = parrot::array({1, 2, 3});
    auto mask   = parrot::array({0, 0, 0});
    auto result = arr.keep(mask);
    CHECK_EQ(result.size(), 0);
}

// Test keep function with all ones mask
TEST_CASE("ParrotTest - KeepAllOnesTest") {
    auto arr    = parrot::array({1, 2, 3});
    auto mask   = parrot::array({1, 1, 1});
    auto result = arr.keep(mask);
    CHECK(check_match(result, arr));
}

// Test match function with identical arrays
TEST_CASE("ParrotTest - MatchIdenticalTest") {
    auto arr1 = parrot::array({1, 2, 3, 4});
    auto arr2 = parrot::array({1, 2, 3, 4});
    CHECK(arr1.match(arr2));
}

// Test match function with different arrays
TEST_CASE("ParrotTest - MatchDifferentTest") {
    auto arr1 = parrot::array({1, 2, 3, 4});
    auto arr2 = parrot::array({1, 2, 3, 5});
    CHECK_FALSE(arr1.match(arr2));
}

// Test match function with different length arrays
TEST_CASE("ParrotTest - MatchDifferentLengthTest") {
    auto arr1 = parrot::array({1, 2, 3});
    auto arr2 = parrot::array({1, 2, 3, 4});
    CHECK_FALSE(arr1.match(arr2));
}

// Test take method with valid size
TEST_CASE("ParrotTest - TakeTest") {
    auto arr    = parrot::array({1, 2, 3, 4, 5});
    auto result = arr.take(3);
    CHECK_EQ(result.size(), 3);
    auto expected = parrot::array({1, 2, 3});
    CHECK(check_match(result, expected));
}

// Test take with full size
TEST_CASE("ParrotTest - TakeFullSizeTest") {
    auto arr    = parrot::array({1, 2, 3, 4, 5});
    auto result = arr.take(5);
    CHECK_EQ(result.size(), 5);
    CHECK(check_match(result, arr));
}

// Test take with zero size
TEST_CASE("ParrotTest - TakeZeroTest") {
    auto arr    = parrot::array({1, 2, 3, 4, 5});
    auto result = arr.take(0);
    CHECK_EQ(result.size(), 0);
}

// Test take with invalid size
TEST_CASE("ParrotTest - TakeInvalidTest") {
    auto arr = parrot::array({1, 2, 3, 4, 5});
    CHECK_THROWS_AS(static_cast<void>(arr.take(6)), std::invalid_argument);
    CHECK_THROWS_AS(static_cast<void>(arr.take(-1)), std::invalid_argument);
}

// Test case for drop method
TEST_CASE("ParrotTest - DropTest") {
    auto arr = parrot::array({1, 2, 3, 4, 5});

    // Drop 2 elements
    auto dropped_arr = arr.drop(2);
    CHECK_EQ(dropped_arr.size(), 3);
    CHECK(check_match(dropped_arr, parrot::array({3, 4, 5})));

    // Drop 0 elements
    auto dropped_zero_arr = arr.drop(0);
    CHECK_EQ(dropped_zero_arr.size(), 5);
    CHECK(check_match(dropped_zero_arr, parrot::array({1, 2, 3, 4, 5})));

    // Drop all elements
    auto dropped_all_arr = arr.drop(5);
    CHECK_EQ(dropped_all_arr.size(), 0);
    CHECK(check_match(dropped_all_arr, parrot::array<int>({})));

    // Drop more than size (should throw)
    CHECK_THROWS_AS([[maybe_unused]] auto x = arr.drop(6),
                    std::invalid_argument);

    // Drop negative (should throw)
    CHECK_THROWS_AS([[maybe_unused]] auto x = arr.drop(-1),
                    std::invalid_argument);

    // Test with an empty array
    auto empty_arr         = parrot::array<int>({});
    auto dropped_empty_arr = empty_arr.drop(0);
    CHECK_EQ(dropped_empty_arr.size(), 0);
    CHECK(check_match(dropped_empty_arr, parrot::array<int>({})));
    CHECK_THROWS_AS([[maybe_unused]] auto x = empty_arr.drop(1),
                    std::invalid_argument);  // Drop from empty
}

// Test filter method with predicate
TEST_CASE("ParrotTest - FilterPredicateTest") {
    auto arr = parrot::array({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    // Filter even numbers
    auto evens = arr.filter(
      [=] __host__ __device__(int x) { return x % 2 == 0; });
    auto expected_evens = parrot::array({2, 4, 6, 8, 10});
    CHECK(check_match(evens, expected_evens));

    // Filter numbers greater than 5
    auto greater_than_five = arr.filter(
      [=] __host__ __device__(int x) { return x > 5; });
    auto expected_greater = parrot::array({6, 7, 8, 9, 10});
    CHECK(check_match(greater_than_five, expected_greater));
}

// Element-wise operations on keep() and filter() results see only the kept
// elements
TEST_CASE("ParrotTest - MaskedElementwiseTest") {
    auto arr  = parrot::array({1, 2, 3, 4});
    auto mask = parrot::array({1, 0, 1, 0});
    auto kept = arr.keep(mask);  // {1, 3}

    CHECK(check_match(kept.sq(), parrot::array({1, 9})));
    CHECK(check_match(kept.dble(), parrot::array({2, 6})));
    CHECK_EQ(kept.sq().sum().value(), 10);
    CHECK(check_match(kept.deltas(), parrot::array({2})));
    CHECK(check_match(kept.add(kept), parrot::array({2, 6})));
    CHECK(
      check_match(kept.add(parrot::array({10, 20})), parrot::array({11, 23})));
    CHECK(
      check_match(parrot::array({10, 20}).add(kept), parrot::array({11, 23})));

    auto evens = arr.filter(
      [=] __host__ __device__(int x) { return x % 2 == 0; });
    CHECK(check_match(evens.sq(), parrot::array({4, 16})));

    // rand(0) is always 0, so only the kept zeros may reach rand()
    auto zeros = parrot::array({0, 1000000, 0, 1000000}).keep(mask);
    CHECK(check_match(zeros.rand(), parrot::array({0, 0})));
}

// Structural operations on keep() results also see only the kept elements,
// whether the masked array is the receiver or the argument
TEST_CASE("ParrotTest - MaskedStructuralTest") {
    auto arr  = parrot::array({1, 2, 3, 4, 5});
    auto mask = parrot::array({1, 0, 1, 1, 0});
    auto kept = arr.keep(mask);  // {1, 3, 4}

    CHECK(check_match(kept.take(2), parrot::array({1, 3})));
    CHECK(check_match(kept.drop(1), parrot::array({3, 4})));
    CHECK(check_match(kept.cycle({5}), parrot::array({1, 3, 4, 1, 3})));
    CHECK(check_match(kept.reshape({3, 1}), parrot::array({1, 3, 4})));
    CHECK(check_match(kept.flatten(), parrot::array({1, 3, 4})));
    CHECK(
      check_match(kept.keep(parrot::array({1, 0, 1})), parrot::array({1, 4})));
    CHECK(
      check_match(kept.filter([] __host__ __device__(int x) { return x > 1; }),
                  parrot::array({3, 4})));
    CHECK(
      check_match(kept.gather(parrot::array({2, 0})), parrot::array({4, 1})));
    CHECK(kept.match(parrot::array({1, 3, 4})));
    CHECK(parrot::array({1, 3, 4}).match(kept));
    CHECK(check_match(kept.enumerate().map(parrot::fst{}),
                      parrot::array({1, 3, 4})));
    CHECK(check_match(parrot::array({7, 8, 9}).pairs(kept).map(parrot::snd{}),
                      parrot::array({1, 3, 4})));
    CHECK(check_match(kept.outer(parrot::array({1, 10}), parrot::mul{}),
                      parrot::array({1, 10, 3, 30, 4, 40})));
    CHECK(check_match(parrot::array({1, 10}).outer(kept, parrot::mul{}),
                      parrot::array({1, 3, 4, 10, 30, 40})));
    CHECK(check_match(parrot::array({1, 2}).cross(kept).map(parrot::snd{}),
                      parrot::array({1, 3, 4, 1, 3, 4})));
    CHECK(check_match(
      parrot::array({10, 20, 30})
        .gather(parrot::array({0, 1, 2}).keep(parrot::array({1, 0, 1}))),
      parrot::array({10, 30})));

    auto runs = parrot::array({2, 9, 2, 3, 3})
                  .keep(parrot::array({1, 0, 1, 1, 1}));  // {2, 2, 3, 3}
    CHECK(check_match(runs.chunk_by_reduce(parrot::eq{}, parrot::add{}),
                      parrot::array({4, 6})));

    auto counts = parrot::array({0, 1, 0, 2, 5})
                    .keep(parrot::array({0, 1, 1, 1, 0}));  // {1, 0, 2}
    CHECK(check_match(parrot::array({1, 2, 3}).replicate(counts),
                      parrot::array({1, 3, 3})));
}

// Lazy results keep temporary operands alive
TEST_CASE("ParrotTest - TemporaryOperandLifetimeTest") {
    auto arr      = parrot::array({1, 2, 3});
    auto gathered = parrot::array({7, 8, 9}).gather(parrot::array({2, 0}));
    auto paired   = arr.pairs(parrot::array({10, 20, 30}));
    auto outer    = arr.outer(parrot::array({10, 20}), parrot::add{});
    auto crossed  = arr.cross(parrot::array({10, 20}));

    CHECK(check_match(gathered, parrot::array({9, 7})));
    CHECK(check_match(paired.map(parrot::snd{}), parrot::array({10, 20, 30})));
    CHECK(check_match(outer, parrot::array({11, 21, 12, 22, 13, 23})));
    CHECK(check_match(crossed.map(parrot::snd{}),
                      parrot::array({10, 20, 10, 20, 10, 20})));
}

// Test rev function
TEST_CASE("ParrotTest - RevTest") {
    auto arr      = parrot::array({1, 2, 3, 4, 5});
    auto result   = arr.rev();
    auto expected = parrot::array({5, 4, 3, 2, 1});
    CHECK(check_match(result, expected));
    CHECK_EQ(result.sum().value(),
             arr.sum().value());  // Sum should be the same
}

// Test rev function with operations
TEST_CASE("ParrotTest - RevChainedTest") {
    auto arr      = parrot::array({1, 2, 3, 4, 5});
    auto result   = arr.rev().times(2);
    auto expected = parrot::array({10, 8, 6, 4, 2});
    CHECK(check_match(result, expected));
}

// Test rev function with operations
TEST_CASE("ParrotTest - IotaReverseTest") {
    auto arr      = parrot::range(5).rev();
    auto expected = parrot::array({5, 4, 3, 2, 1});
    CHECK(check_match(arr, expected));
}

// Test rev function with operations
TEST_CASE("ParrotTest - IotaReverseReverseTest") {
    auto arr      = parrot::range(5).rev().rev();
    auto expected = parrot::array({1, 2, 3, 4, 5});
    CHECK(check_match(arr, expected));
}
