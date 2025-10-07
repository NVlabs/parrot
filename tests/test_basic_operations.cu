/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// Test range function
TEST_CASE("ParrotTest - RangeTest") {
    // Test that range(10) creates a range from 1 to 10
    auto result = parrot::range(10).sum();
    CHECK_EQ(result.value(), 55);  // sum of 1-10 is 55
}

// Test times function
TEST_CASE("ParrotTest - TimesTest") {
    auto result = parrot::range(10).times(2).sum();
    CHECK_EQ(result.value(), 110);  // sum of 2,4,6,8,10,12,14,16,18,20
}

// Test add function
TEST_CASE("ParrotTest - AddTest") {
    auto result = parrot::range(10).add(5).sum();
    CHECK_EQ(result.value(), 105);  // sum of 6,7,8,9,10,11,12,13,14,15
}

// Test chained operations
TEST_CASE("ParrotTest - ChainedOperationsTest") {
    auto result = parrot::range(10).times(2).add(1).sum();
    CHECK_EQ(result.value(), 120);  // sum of 3,5,7,9,11,13,15,17,19,21
}

// Test append function
TEST_CASE("ParrotTest - AppendTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.append(5).sum();
    CHECK_EQ(result.value(), 15);  // sum of 1,2,3,4,5
}

// Test prepend function
TEST_CASE("ParrotTest - PrependTest") {
    auto arr    = parrot::array({2, 3, 4});
    auto result = arr.prepend(1).sum();
    CHECK_EQ(result.value(), 10);  // sum of 1,2,3,4 = 10
}

// Test prepend with non-default type
TEST_CASE("ParrotTest - PrependFloatTest") {
    auto arr    = parrot::array<float>({2.5F, 3.5F, 4.5F});
    auto result = arr.prepend(1.5F).sum();
    CHECK(result.value() ==
          doctest::Approx(12.0F));  // sum of 1.5,2.5,3.5,4.5 = 12.0
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

// Test drop method
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

// Test array function with explicit template parameter
TEST_CASE("ParrotTest - ArrayWithExplicitTemplateTest") {
    auto arr = parrot::array<int>({1, 2, 3, 4});  // Explicit <int>
    CHECK_EQ(arr.size(), 4);
    CHECK_EQ(arr.sum().value(), 10);
}

// Test array function with automatic template parameter deduction
TEST_CASE("ParrotTest - ArrayWithInitializerListTest") {
    auto arr = parrot::array({1, 2, 3, 4});  // Deduces int
    CHECK_EQ(arr.size(), 4);
    CHECK_EQ(arr.sum().value(), 10);

    // Test with floating point
    auto float_arr = parrot::array({1.5F, 2.5F, 3.5F});  // Deduces float
    CHECK_EQ(float_arr.size(), 3);
    CHECK(float_arr.sum().value() == doctest::Approx(7.5));
}

TEST_CASE("ParrotTest - WhereMapPlus0Test") {
    auto arr      = parrot::array({1, 2, 3, 4});
    auto result   = arr.odd().where();
    auto result2  = result.add(0);
    auto expected = parrot::array({1, 3});
    check_match_eq(result, expected);
    check_match_eq(result2, expected);
}
