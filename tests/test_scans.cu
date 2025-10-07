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

#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_common.hpp"

// Test sums function (inclusive scan with addition)
TEST_CASE("ParrotTest - SumsTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.sums();
    CHECK_EQ(result.size(), 4);
    auto expected_arr = parrot::array({1, 3, 6, 10});
    CHECK(check_match(result, expected_arr));
}

// Test prods function (inclusive scan with multiplication)
TEST_CASE("ParrotTest - ProdsTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.prods();
    CHECK_EQ(result.size(), 4);
    auto expected_arr = parrot::array({1, 2, 6, 24});
    CHECK(check_match(result, expected_arr));
}

// Test mins function (inclusive scan with minimum)
TEST_CASE("ParrotTest - MinsTest") {
    auto arr    = parrot::array({3, 1, 4, 2});
    auto result = arr.mins();
    CHECK_EQ(result.size(), 4);
    auto expected_arr = parrot::array({3, 1, 1, 1});
    CHECK(check_match(result, expected_arr));
}

// Test maxs function (inclusive scan with maximum)
TEST_CASE("ParrotTest - MaxsTest") {
    auto arr    = parrot::array({3, 1, 4, 2});
    auto result = arr.maxs();
    CHECK_EQ(result.size(), 4);
    auto expected_arr = parrot::array({3, 3, 4, 4});
    CHECK(check_match(result, expected_arr));
}

// Test anys function (inclusive scan with logical OR)
TEST_CASE("ParrotTest - AnysTest") {
    auto arr    = parrot::array({0, 1, 0, 0, 2});  // Non-zero treated as true
    auto result = arr.anys();
    CHECK_EQ(result.size(), 5);
    auto expected_arr = parrot::array({0, 1, 1, 1, 1});  // Result is 0 or 1
    CHECK(check_match(result, expected_arr));
}

// Test alls function (inclusive scan with logical AND)
TEST_CASE("ParrotTest - AllsTest") {
    auto arr = parrot::array(
      {1, 1, 0, 1, 2});  // Zero treated as false, non-zero as true
    auto result = arr.alls();
    CHECK_EQ(result.size(), 5);
    auto expected_arr = parrot::array({1, 1, 0, 0, 0});  // Result is 0 or 1
    CHECK(check_match(result, expected_arr));
}

// Test scan function with plus operation
TEST_CASE("ParrotTest - ScanPlusTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.scan(parrot::add{});
    CHECK_EQ(result.size(), 4);
    auto expected_arr = parrot::array({1, 3, 6, 10});
    CHECK(check_match(result, expected_arr));
    CHECK(check_match(result,
                      arr.sums()));  // Verify it matches the sums() function
}

// Test scan function with multiplies operation
TEST_CASE("ParrotTest - ScanMultipliesTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.scan(parrot::mul{});
    CHECK_EQ(result.size(), 4);
    auto expected_arr = parrot::array({1, 2, 6, 24});
    CHECK(check_match(result, expected_arr));
    CHECK(check_match(result,
                      arr.prods()));  // Verify it matches the prods() function
}

// Test scan function with minimum operation
TEST_CASE("ParrotTest - ScanMinimumTest") {
    auto arr    = parrot::array({3, 1, 4, 2});
    auto result = arr.scan(parrot::min{});
    CHECK_EQ(result.size(), 4);
    auto expected_arr = parrot::array({3, 1, 1, 1});
    CHECK(check_match(result, expected_arr));
    CHECK(check_match(result,
                      arr.mins()));  // Verify it matches the mins() function
}

// Test scan function with maximum operation
TEST_CASE("ParrotTest - ScanMaximumTest") {
    auto arr    = parrot::array({3, 1, 4, 2});
    auto result = arr.scan(parrot::max{});
    CHECK_EQ(result.size(), 4);
    auto expected_arr = parrot::array({3, 3, 4, 4});
    CHECK(check_match(result, expected_arr));
    CHECK(check_match(result,
                      arr.maxs()));  // Verify it matches the maxs() function
}

// Test scan with Axis=1 (column-wise)
TEST_CASE("ParrotTest - ScanColTest") {
    auto matrix = parrot::array({1, 2, 3, 4, 5, 6, 7, 8, 9}).reshape({3, 3});

    // Column-wise sums
    auto scan_col_sums     = matrix.scan<1>(parrot::add{});
    auto expected_col_sums = parrot::array({1,
                                         2,
                                         3,
                                         1 + 4,
                                         2 + 5,
                                         3 + 6,
                                         1 + 4 + 7,
                                         2 + 5 + 8,
                                         3 + 6 + 9})
                               .reshape({3, 3});
    CHECK(check_match(scan_col_sums, expected_col_sums));

    // Column-wise prods
    auto scan_col_prods     = matrix.scan<1>(parrot::mul{});
    auto expected_col_prods = parrot::array({1,
                                          2,
                                          3,
                                          1 * 4,
                                          2 * 5,
                                          3 * 6,
                                          1 * 4 * 7,
                                          2 * 5 * 8,
                                          3 * 6 * 9})
                                .reshape({3, 3});
    CHECK(check_match(scan_col_prods, expected_col_prods));

    // Column-wise mins
    auto scan_col_mins     = matrix.scan<1>(parrot::min{});
    auto expected_col_mins = parrot::array({1, 2, 3, 1, 2, 3, 1, 2, 3})
                               .reshape({3, 3});
    CHECK(check_match(scan_col_mins, expected_col_mins));

    // Column-wise maxs
    auto scan_col_maxs     = matrix.scan<1>(parrot::max{});
    auto expected_col_maxs = parrot::array({1, 2, 3, 4, 5, 6, 7, 8, 9})
                               .reshape({3, 3});
    CHECK(check_match(scan_col_maxs, expected_col_maxs));
}
