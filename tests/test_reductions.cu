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

#include <limits>
#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_common.hpp"

// Test deltas function
TEST_CASE("ParrotTest - DeltasTest") {
    auto arr    = parrot::array({1, 3, 6, 10});
    auto result = arr.deltas().sum();
    CHECK_EQ(result.value(), 9);  // sum of 2,3,4
}

// Test maxr function
TEST_CASE("ParrotTest - MaxrTest") {
    auto arr    = parrot::array({1, 5, 3, 2});
    auto result = arr.maxr();
    CHECK_EQ(result.value(), 5);
}

// Test minr function
TEST_CASE("ParrotTest - MinrTest") {
    auto arr    = parrot::array({1, 5, 3, 2});
    auto result = arr.minr();
    CHECK_EQ(result.value(), 1);
}

// Test minr with empty array (using only initial value)
TEST_CASE("ParrotTest - MinrEmptyTest") {
    auto arr    = parrot::array<int>({});
    auto result = arr.minr();
    CHECK_EQ(result.value(), std::numeric_limits<int>::max());
}

// Test minr with negative values
TEST_CASE("ParrotTest - MinrNegativeTest") {
    auto arr    = parrot::array({-1, -5, 3, 2});
    auto result = arr.minr();
    CHECK_EQ(result.value(), -5);
}

// Test minmax function
TEST_CASE("ParrotTest - MinmaxTest") {
    auto arr    = parrot::array({3, 1, 7, 5, 2});
    auto result = arr.minmax().to_host();
    REQUIRE_EQ(result.size(), 1);
    CHECK_EQ(result[0].first, 1);   // minimum value
    CHECK_EQ(result[0].second, 7);  // maximum value
}

// Test minmax with negative values
TEST_CASE("ParrotTest - MinmaxNegativeTest") {
    auto arr    = parrot::array({-3, 1, -7, 5, 2});
    auto result = arr.minmax().to_host();
    REQUIRE_EQ(result.size(), 1);
    CHECK_EQ(result[0].first, -7);  // minimum value
    CHECK_EQ(result[0].second, 5);  // maximum value
}

// Test any() method with all zeros
TEST_CASE("ParrotTest - AnyAllZerosTest") {
    auto arr    = parrot::array({0, 0, 0, 0});
    auto result = arr.any();
    CHECK_FALSE(result.value());
}

// Test any() method with some non-zeros
TEST_CASE("ParrotTest - AnySomeNonZerosTest") {
    auto arr    = parrot::array({0, 0, 3, 0});
    auto result = arr.any();
    CHECK(result.value());
}

// Test all() method with all non-zeros
TEST_CASE("ParrotTest - AllNonZerosTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.all();
    CHECK(result.value());
}

// Test all() method with some zeros
TEST_CASE("ParrotTest - AllSomeZerosTest") {
    auto arr    = parrot::array({1, 0, 3, 4});
    auto result = arr.all();
    CHECK_FALSE(result.value());
}

// Test any() and all() with empty array
TEST_CASE("ParrotTest - AnyAllEmptyTest") {
    auto arr = parrot::array<int>({});
    CHECK_FALSE(
      arr.any().value());  // Empty array should return false for any()
    CHECK(
      arr.all()
        .value());  // Empty array should return true for all() (vacuously true)
}

// Test prod function
TEST_CASE("ParrotTest - ProdTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.prod();
    CHECK_EQ(result.value(), 24);  // product of 1,2,3,4 is 24
}

// Test prod with an empty array
TEST_CASE("ParrotTest - ProdEmptyTest") {
    auto arr    = parrot::array<int>({});
    auto result = arr.prod();
    CHECK_EQ(result.value(),
             1);  // product of an empty array is the identity (1)
}

// Test prod with floating point values
TEST_CASE("ParrotTest - ProdFloatTest") {
    auto arr    = parrot::array<float>({1.5F, 2.0F, 2.5F});
    auto result = arr.prod();
    CHECK(result.value() ==
          doctest::Approx(7.5F));  // product of 1.5*2.0*2.5 = 7.5
}

// Test reduce function with plus operation
TEST_CASE("ParrotTest - ReducePlusTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.reduce(0, parrot::add{});
    CHECK_EQ(result.value(), 10);
    CHECK_EQ(result.value(),
             arr.sum().value());  // Verify it matches the sum() function
}

// Test reduce function with multiplies operation
TEST_CASE("ParrotTest - ReduceMultipliesTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.reduce(1, parrot::mul{});
    CHECK_EQ(result.value(), 24);
    CHECK_EQ(result.value(),
             arr.prod().value());  // Verify it matches the prod() function
}

// Test reduce function with maximum operation
TEST_CASE("ParrotTest - ReduceMaximumTest") {
    auto arr    = parrot::array({1, 5, 3, 4});
    auto result = arr.reduce(std::numeric_limits<int>::lowest(), parrot::max{});
    CHECK_EQ(result.value(), 5);
    CHECK_EQ(result.value(),
             arr.maxr().value());  // Verify it matches the maxr() function
}

// Test reduce function with minimum operation
TEST_CASE("ParrotTest - ReduceMinimumTest") {
    auto arr    = parrot::array({5, 2, 3, 4});
    auto result = arr.reduce(std::numeric_limits<int>::max(), parrot::min{});
    CHECK_EQ(result.value(), 2);
    CHECK_EQ(result.value(),
             arr.minr().value());  // Verify it matches the minr() function
}

// Test stats::mode function
TEST_CASE("ParrotTest - StatsModeTest") {
    auto arr    = parrot::array({3, 1, 3, 1, 2, 3});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), 3);  // 3 appears most frequently (3 times)
}

// Test stats::mode with single mode
TEST_CASE("ParrotTest - StatsModeSingleTest") {
    auto arr    = parrot::array({1, 2, 2, 3});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), 2);  // 2 appears most frequently (2 times)
}

// Test stats::mode with all unique elements
TEST_CASE("ParrotTest - StatsModeUniqueTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), 1);  // All elements appear once, returns smallest
}

// Test stats::mode with single element
TEST_CASE("ParrotTest - StatsModeSingleElementTest") {
    auto arr    = parrot::array({42});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), 42);  // Single element is the mode
}

// Test stats::mode with negative numbers
TEST_CASE("ParrotTest - StatsModeNegativeTest") {
    auto arr    = parrot::array({-1, -2, -1, -3, -1});
    auto result = parrot::stats::mode(arr);
    CHECK_EQ(result.value(), -1);  // -1 appears most frequently (3 times)
}