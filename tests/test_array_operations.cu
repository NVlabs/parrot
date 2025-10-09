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

// Test add function with another array
TEST_CASE("ParrotTest - AddArrayTest") {
    auto arr1   = parrot::array({1, 2, 3, 4});
    auto arr2   = parrot::array({10, 20, 30, 40});
    auto result = arr1.add(arr2).sum();
    CHECK_EQ(result.value(), 110);  // sum of 11,22,33,44 = 110
}

// Test add function with arrays of different sizes
TEST_CASE("ParrotTest - AddArrayDifferentSizeTest") {
    auto arr1 = parrot::array({1, 2, 3, 4});
    auto arr2 = parrot::array({10, 20});
    CHECK_THROWS_AS(arr1.add(arr2), std::invalid_argument);
}

// Test add function with mixed operation (scalar and array)
TEST_CASE("ParrotTest - AddMixedTest") {
    auto arr1 = parrot::array({1, 2, 3, 4});
    auto arr2 = parrot::array({10, 20, 30, 40});

    // First add scalar, then add array
    auto result1 = arr1.add(5).add(arr2).sum();
    CHECK_EQ(result1.value(), 130);  // sum of 16,27,38,49 = 130

    // First add array, then add scalar
    auto result2 = arr1.add(arr2).add(5).sum();
    CHECK_EQ(result2.value(), 130);  // sum of 16,27,38,49 = 130
}

// Test times function with another array
TEST_CASE("ParrotTest - TimesArrayTest") {
    auto arr1   = parrot::array({1, 2, 3, 4});
    auto arr2   = parrot::array({10, 20, 30, 40});
    auto result = arr1.times(arr2).sum();
    CHECK_EQ(result.value(), 300);  // sum of 10,40,90,160 = 300
}

// Test min function with another array
TEST_CASE("ParrotTest - MinArrayTest") {
    auto arr1     = parrot::array({7, 2, 8, 4});
    auto arr2     = parrot::array({5, 3, 6, 9});
    auto result   = arr1.min(arr2);
    auto expected = parrot::array({5, 2, 6, 4});
    CHECK(check_match(result, expected));
}

// Test max function with another array
TEST_CASE("ParrotTest - MaxArrayTest") {
    auto arr1     = parrot::array({7, 2, 8, 4});
    auto arr2     = parrot::array({5, 3, 6, 9});
    auto result   = arr1.max(arr2);
    auto expected = parrot::array({7, 3, 8, 9});
    CHECK(check_match(result, expected));
}

// Test div function with another array
TEST_CASE("ParrotTest - DivArrayTest") {
    auto arr1     = parrot::array({10, 20, 15, 8});
    auto arr2     = parrot::array({2, 4, 3, 2});
    auto result   = arr1.div(arr2);
    auto expected = parrot::array({5, 5, 5, 4});
    CHECK(check_match(result, expected));
}

// Test div function with floating point arrays
TEST_CASE("ParrotTest - DivArrayFloatTest") {
    auto arr1     = parrot::array<float>({10.0F, 20.0F, 15.0F, 8.0F});
    auto arr2     = parrot::array<float>({2.0F, 4.0F, 3.0F, 2.5F});
    auto result   = arr1.div(arr2);
    auto expected = parrot::array<float>({5.0F, 5.0F, 5.0F, 3.2F});
    CHECK(check_match(result, expected));
}

// Test idiv function with another array
TEST_CASE("ParrotTest - IDivArrayTest") {
    auto arr1     = parrot::array({10, 21, 15, 8});
    auto arr2     = parrot::array({3, 5, 4, 3});
    auto result   = arr1.idiv(arr2);
    auto expected = parrot::array({3, 4, 3, 2});
    CHECK(check_match(result, expected));
}

// Test minus function with another array
TEST_CASE("ParrotTest - MinusArrayTest") {
    auto arr1     = parrot::array({10, 20, 15, 8});
    auto arr2     = parrot::array({2, 5, 3, 4});
    auto result   = arr1.minus(arr2);
    auto expected = parrot::array({8, 15, 12, 4});
    CHECK(check_match(result, expected));
}

// Test gte function with another array
TEST_CASE("ParrotTest - GteArrayTest") {
    auto arr1     = parrot::array({5, 3, 7, 4});
    auto arr2     = parrot::array({5, 2, 9, 6});
    auto result   = arr1.gte(arr2);
    auto expected = parrot::array({1, 1, 0, 0});  // Results are 0 or 1
    CHECK(check_match(result, expected));
}

// Test lte function with another array
TEST_CASE("ParrotTest - LteArrayTest") {
    auto arr1     = parrot::array({5, 3, 7, 4});
    auto arr2     = parrot::array({5, 2, 9, 6});
    auto result   = arr1.lte(arr2);
    auto expected = parrot::array({1, 0, 1, 1});  // Results are 0 or 1
    CHECK(check_match(result, expected));
}

// Test gt function with another array
TEST_CASE("ParrotTest - GtArrayTest") {
    auto arr1     = parrot::array({5, 3, 7, 4});
    auto arr2     = parrot::array({5, 2, 9, 6});
    auto result   = arr1.gt(arr2);
    auto expected = parrot::array({0, 1, 0, 0});  // Results are 0 or 1
    CHECK(check_match(result, expected));
}

// Test lt function with another array
TEST_CASE("ParrotTest - LtArrayTest") {
    auto arr1     = parrot::array({5, 3, 7, 4});
    auto arr2     = parrot::array({5, 2, 9, 6});
    auto result   = arr1.lt(arr2);
    auto expected = parrot::array({0, 0, 1, 1});  // Results are 0 or 1
    CHECK(check_match(result, expected));
}

// Test combined operations with array-array and scalar operations
TEST_CASE("ParrotTest - CombinedArrayAndScalarTest") {
    auto arr1 = parrot::array({2, 3, 4, 5});
    auto arr2 = parrot::array({10, 20, 30, 40});

    // Test times(array) followed by add(scalar)
    auto result1   = arr1.times(arr2).add(5);
    auto expected1 = parrot::array({25, 65, 125, 205});
    CHECK(check_match(result1, expected1));

    // Test max(scalar) followed by min(array)
    auto arr3      = parrot::array({15, 25, 35, 45});
    auto arr4      = parrot::array({25, 15, 40, 30});
    auto result2   = arr3.max(20).min(arr4);
    auto expected2 = parrot::array({20, 15, 40, 30});
    check_match_eq(result2, expected2);
}

// Test map2 function with a custom functor
TEST_CASE("ParrotTest - Map2Test") {
    // Using the binary functor defined in parrot.hpp for multiplication
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.map2(arr, parrot::mul{}).sum();
    CHECK_EQ(result.value(),
             30);  // sum of [1*1, 2*2, 3*3, 4*4] = 1+4+9+16 = 30

    // Test with two different arrays
    auto arr2    = parrot::array({2, 3, 4, 5});
    auto result2 = arr.map2(arr2, parrot::mul{}).sum();
    CHECK_EQ(result2.value(),
             40);  // sum of [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20] = 40
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

// Test replicate function with invalid n
TEST_CASE("ParrotTest - ReplicateInvalidNTest") {
    auto arr = parrot::array({1, 2, 3});
    CHECK_THROWS_AS((void)arr.replicate(0), std::invalid_argument);
    CHECK_THROWS_AS((void)arr.replicate(-1), std::invalid_argument);
}

// Test replicate function with mask array
TEST_CASE("ParrotTest - ReplicateMaskBasicTest") {
    auto arr      = parrot::array({1, 2, 3});
    auto mask     = parrot::array({2, 1, 3});
    auto result   = arr.replicate(mask);
    auto expected = parrot::array({1, 1, 2, 3, 3, 3});
    CHECK(check_match(result, expected));
}

// Test replicate function with mask array - identity case
TEST_CASE("ParrotTest - ReplicateMaskIdentityTest") {
    auto arr      = parrot::array({5, 10, 15});
    auto mask     = parrot::array({1, 1, 1});
    auto result   = arr.replicate(mask);
    auto expected = parrot::array({5, 10, 15});
    CHECK(check_match(result, expected));
}

// Test replicate function with mask array - some zeros
TEST_CASE("ParrotTest - ReplicateMaskZerosTest") {
    auto arr      = parrot::array({1, 2, 3, 4});
    auto mask     = parrot::array({2, 0, 1, 3});
    auto result   = arr.replicate(mask);
    auto expected = parrot::array({1, 1, 3, 4, 4, 4});
    CHECK(check_match(result, expected));
}

// Test replicate function with mask array - invalid mask size
TEST_CASE("ParrotTest - ReplicateMaskInvalidSizeTest") {
    auto arr  = parrot::array({1, 2, 3});
    auto mask = parrot::array({2, 1});  // Wrong size
    CHECK_THROWS_AS((void)arr.replicate(mask), std::invalid_argument);
}

// Test replicate function with mask array - negative values
TEST_CASE("ParrotTest - ReplicateMaskNegativeTest") {
    auto arr  = parrot::array({1, 2, 3});
    auto mask = parrot::array({2, -1, 1});  // Negative value
    CHECK_THROWS_AS((void)arr.replicate(mask), std::invalid_argument);
}

// Test replicate function with mask array - real world example (duplicate
// zeros)
TEST_CASE("ParrotTest - ReplicateMaskDuplicateZerosTest") {
    auto arr      = parrot::array({1, 0, 2, 3, 0, 4});
    auto mask     = arr.eq(0).add(1);  // Creates mask {1, 2, 1, 1, 2, 1}
    auto result   = arr.replicate(mask).take(arr.size());
    auto expected = parrot::array({1, 0, 0, 2, 3, 0});
    CHECK(check_match(result, expected));
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

// Test cross function with empty arrays
TEST_CASE("ParrotTest - CrossEmptyArrayTest") {
    auto arr1 = parrot::array<int>({});
    auto arr2 = parrot::array({1, 2, 3});
    CHECK_THROWS_AS(arr1.cross(arr2), std::invalid_argument);

    auto arr3 = parrot::array({1, 2, 3});
    auto arr4 = parrot::array<int>({});
    CHECK_THROWS_AS(arr3.cross(arr4), std::invalid_argument);
}

// Test outer function with multiplication
TEST_CASE("ParrotTest - OuterBasicTest") {
    auto arr1   = parrot::array({2, 3});
    auto arr2   = parrot::array({4, 5});
    auto result = arr1.outer(arr2, parrot::mul{});

    // Should be a 2x2 matrix with:
    // [2*4, 2*5]  = [8, 10]
    // [3*4, 3*5]  = [12, 15]
    CHECK_EQ(result.size(), 4);
    CHECK_EQ(result.shape()[0], 2);
    CHECK_EQ(result.shape()[1], 2);

    auto result_host = result.to_host();
    CHECK_EQ(result_host[0], 8);   // 2*4
    CHECK_EQ(result_host[1], 10);  // 2*5
    CHECK_EQ(result_host[2], 12);  // 3*4
    CHECK_EQ(result_host[3], 15);  // 3*5
}

// Test outer function with addition
TEST_CASE("ParrotTest - OuterAddTest") {
    auto arr1   = parrot::array({1, 2});
    auto arr2   = parrot::array({10, 20});
    auto result = arr1.outer(arr2, parrot::add{});

    // Should be a 2x2 matrix with:
    // [1+10, 1+20]  = [11, 21]
    // [2+10, 2+20]  = [12, 22]
    auto result_host = result.to_host();
    CHECK_EQ(result_host[0], 11);  // 1+10
    CHECK_EQ(result_host[1], 21);  // 1+20
    CHECK_EQ(result_host[2], 12);  // 2+10
    CHECK_EQ(result_host[3], 22);  // 2+20
}

// Test outer function with empty arrays
TEST_CASE("ParrotTest - OuterEmptyArrayTest") {
    auto arr1 = parrot::array<int>({});
    auto arr2 = parrot::array({1, 2, 3});
    CHECK_THROWS_AS(arr1.outer(arr2, parrot::mul{}), std::invalid_argument);

    auto arr3 = parrot::array({1, 2, 3});
    auto arr4 = parrot::array<int>({});
    CHECK_THROWS_AS(arr3.outer(arr4, parrot::mul{}), std::invalid_argument);
}

TEST_CASE("ParrotTest - OuterAddSumTest") {
    auto arr      = parrot::range(10);
    auto result   = arr.outer(arr, parrot::add{}).sum<2>();
    auto expected = arr.add(5).times(10).add(5);
    check_match_eq(result, expected);
}
