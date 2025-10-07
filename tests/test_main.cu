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

// Include all test files
#include "test_advanced_operations.cu"
#include "test_array_operations.cu"
#include "test_basic_operations.cu"
#include "test_integration.cu"
#include "test_math_operations.cu"
#include "test_multidimensional.cu"
#include "test_reductions.cu"
#include "test_scans.cu"
#include "test_sorting.cu"
#include "test_top10.cu"

// Additional integration tests can go here
TEST_CASE("ParrotTest - ChainedExampleTest") {
    auto arr    = parrot::array({3, 6, 9, 1});
    auto result = arr.append(1).sort().deltas().maxr();
    CHECK_EQ(result.value(),
             3);  // max delta in sorted [1,1,3,6,9] is 3 (between 6 and 9)
}

// Test the Python example min_differ_sum_double
TEST_CASE("ParrotTest - MinDifferSumDoubleTest") {
    auto result = parrot::range(20).min(5).differ().sum().dble();
    CHECK_EQ(result.value(), 8);
}

// Test replicate function with basic array
TEST_CASE("ParrotTest - ReplicateBasicTest") {
    auto arr      = parrot::array({1, 2, 3});
    auto result   = arr.replicate(2);
    auto expected = parrot::array({1, 1, 2, 2, 3, 3});
    CHECK(check_match(result, expected));
}

// Test replicate function with n=1 (should be identity)
TEST_CASE("ParrotTest - ReplicateIdentityTest") {
    auto arr      = parrot::array({5, 10, 15});
    auto result   = arr.replicate(1);
    auto expected = parrot::array({5, 10, 15});
    CHECK(check_match(result, expected));
}

// Test replicate function with larger n
TEST_CASE("ParrotTest - ReplicateLargeNTest") {
    auto arr      = parrot::array({7, 8});
    auto result   = arr.replicate(3);
    auto expected = parrot::array({7, 7, 7, 8, 8, 8});
    CHECK(check_match(result, expected));
}

// Test replicate function with single element
TEST_CASE("ParrotTest - ReplicateSingleElementTest") {
    auto arr      = parrot::array({42});
    auto result   = arr.replicate(4);
    auto expected = parrot::array({42, 42, 42, 42});
    CHECK(check_match(result, expected));
}

// Test replicate function with invalid n
TEST_CASE("ParrotTest - ReplicateInvalidNTest") {
    auto arr = parrot::array({1, 2, 3});
    CHECK_THROWS_AS((void)arr.replicate(0), std::invalid_argument);
    CHECK_THROWS_AS((void)arr.replicate(-1), std::invalid_argument);
}

// Test cross function with basic arrays
TEST_CASE("ParrotTest - CrossBasicTest") {
    auto arr1   = parrot::array({1, 2});
    auto arr2   = parrot::array({'a', 'b'});
    auto result = arr1.cross(arr2);

    CHECK_EQ(result.size(), 4);
    auto host_result = result.to_host();

    // Check the cartesian product: [(1, a), (1, b), (2, a), (2, b)]
    CHECK_EQ(host_result[0].first, 1);
    CHECK_EQ(host_result[0].second, 'a');
    CHECK_EQ(host_result[1].first, 1);
    CHECK_EQ(host_result[1].second, 'b');
    CHECK_EQ(host_result[2].first, 2);
    CHECK_EQ(host_result[2].second, 'a');
    CHECK_EQ(host_result[3].first, 2);
    CHECK_EQ(host_result[3].second, 'b');
}

// Test cross function with different sized arrays
TEST_CASE("ParrotTest - CrossDifferentSizesTest") {
    auto arr1   = parrot::array({10, 20, 30});
    auto arr2   = parrot::array({1, 2});
    auto result = arr1.cross(arr2);

    CHECK_EQ(result.size(), 6);
    auto host_result = result.to_host();

    // Check the cartesian product: [(10, 1), (10, 2), (20, 1), (20, 2), (30,
    // 1), (30, 2)]
    CHECK_EQ(host_result[0].first, 10);
    CHECK_EQ(host_result[0].second, 1);
    CHECK_EQ(host_result[1].first, 10);
    CHECK_EQ(host_result[1].second, 2);
    CHECK_EQ(host_result[2].first, 20);
    CHECK_EQ(host_result[2].second, 1);
    CHECK_EQ(host_result[3].first, 20);
    CHECK_EQ(host_result[3].second, 2);
    CHECK_EQ(host_result[4].first, 30);
    CHECK_EQ(host_result[4].second, 1);
    CHECK_EQ(host_result[5].first, 30);
    CHECK_EQ(host_result[5].second, 2);
}

// Test cross function with single element arrays
TEST_CASE("ParrotTest - CrossSingleElementTest") {
    auto arr1   = parrot::array({5});
    auto arr2   = parrot::array({100});
    auto result = arr1.cross(arr2);

    CHECK_EQ(result.size(), 1);
    auto host_result = result.to_host();

    CHECK_EQ(host_result[0].first, 5);
    CHECK_EQ(host_result[0].second, 100);
}

// Test cross function with empty arrays
TEST_CASE("ParrotTest - CrossEmptyArrayTest") {
    auto arr1 = parrot::array<int>({});
    auto arr2 = parrot::array({1, 2, 3});
    CHECK_THROWS_AS(arr1.cross(arr2), std::invalid_argument);

    auto arr3 = parrot::array({1, 2, 3});
    auto arr4 = parrot::array<int>({});
    CHECK_THROWS_AS(arr3.cross(arr4), std::invalid_argument);
}

// Test cross function with floating point arrays
TEST_CASE("ParrotTest - CrossFloatTest") {
    auto arr1   = parrot::array<float>({1.5F, 2.5F});
    auto arr2   = parrot::array<float>({10.0F, 20.0F});
    auto result = arr1.cross(arr2);

    CHECK_EQ(result.size(), 4);
    auto host_result = result.to_host();

    CHECK_EQ(host_result[0].first, 1.5F);
    CHECK_EQ(host_result[0].second, 10.0F);
    CHECK_EQ(host_result[1].first, 1.5F);
    CHECK_EQ(host_result[1].second, 20.0F);
    CHECK_EQ(host_result[2].first, 2.5F);
    CHECK_EQ(host_result[2].second, 10.0F);
    CHECK_EQ(host_result[3].first, 2.5F);
    CHECK_EQ(host_result[3].second, 20.0F);
}

// Test combined usage of replicate and cross
TEST_CASE("ParrotTest - ReplicateAndCrossCombinedTest") {
    auto arr1 = parrot::array({1, 2});
    auto arr2 = parrot::array({10, 20});

    // Test replicate followed by operations
    auto replicated          = arr1.replicate(2);
    auto expected_replicated = parrot::array({1, 1, 2, 2});
    CHECK(check_match(replicated, expected_replicated));

    // Test cross product
    auto crossed = arr1.cross(arr2);
    CHECK_EQ(crossed.size(), 4);

    // Test that replicate and cross work together as expected
    auto cross_result = arr1.cross(arr2);
    auto host_cross   = cross_result.to_host();

    // Verify the cross product is correct
    CHECK_EQ(host_cross[0].first, 1);
    CHECK_EQ(host_cross[0].second, 10);
    CHECK_EQ(host_cross[1].first, 1);
    CHECK_EQ(host_cross[1].second, 20);
    CHECK_EQ(host_cross[2].first, 2);
    CHECK_EQ(host_cross[2].second, 10);
    CHECK_EQ(host_cross[3].first, 2);
    CHECK_EQ(host_cross[3].second, 20);
}