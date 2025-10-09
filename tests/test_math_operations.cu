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

#include <cmath>
#include <cstddef>
#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_common.hpp"

// Test double function
TEST_CASE("ParrotTest - DoubleTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.dble().sum();
    CHECK_EQ(result.value(), 20);  // sum of [2,4,6,8] = 20
}

// Test half function
TEST_CASE("ParrotTest - HalfTest") {
    auto arr    = parrot::array({2, 4, 6, 8});
    auto result = arr.half().sum();
    CHECK_EQ(result.value(), 10);  // sum of [1,2,3,4] = 10
}

// Test half with odd numbers
TEST_CASE("ParrotTest - HalfOddTest") {
    auto arr    = parrot::array({1, 3, 5, 7});
    auto result = arr.half().sum();
    CHECK_EQ(result.value(),
             6);  // sum of [0,1,2,3] = 6 (integer division truncates)
}

// Test half with floating point values
TEST_CASE("ParrotTest - HalfFloatTest") {
    auto arr    = parrot::array<float>({1.0F, 3.0F, 5.0F, 7.0F});
    auto result = arr.half().sum();
    CHECK(result.value() ==
          doctest::Approx(8.0F));  // sum of [0.5,1.5,2.5,3.5] = 8.0
}

// Test abs function with mixed sign values
TEST_CASE("ParrotTest - AbsTest") {
    auto arr    = parrot::array({-2, 3, -5, 7});
    auto result = arr.abs().sum();
    CHECK_EQ(result.value(), 17);  // sum of [2,3,5,7] = 17
}

// Test abs function with all negative values
TEST_CASE("ParrotTest - AbsNegativeTest") {
    auto arr    = parrot::array({-10, -20, -30});
    auto result = arr.abs().sum();
    CHECK_EQ(result.value(), 60);  // sum of [10,20,30] = 60
}

// Test abs function with floating point values
TEST_CASE("ParrotTest - AbsFloatTest") {
    auto arr    = parrot::array<float>({-1.5F, 2.5F, -3.5F});
    auto result = arr.abs().sum();
    CHECK(result.value() ==
          doctest::Approx(7.5F));  // sum of [1.5,2.5,3.5] = 7.5
}

// Test log function with positive values
TEST_CASE("ParrotTest - LogTest") {
    auto arr             = parrot::array<float>({1.0F, 2.0F, 3.0F});
    auto result          = arr.log().sum();
    float const expected = std::log(1.0F) + std::numbers::ln2_v<float> +
                           std::log(3.0F);
    CHECK(result.value() == doctest::Approx(expected));
}

// Test log function with e values
TEST_CASE("ParrotTest - LogETest") {
    auto arr = parrot::array<float>(
      {1.0F, std::numbers::e_v<float>, std::exp(2.0F)});
    auto result = arr.log().sum();
    CHECK(result.value() ==
          doctest::Approx(3.0F));  // sum of log(1)=0, log(e)=1, log(e²)=2 = 3
}

// Test exp function with values
TEST_CASE("ParrotTest - ExpTest") {
    auto arr             = parrot::array<float>({0.0F, 1.0F, 2.0F});
    auto result          = arr.exp().sum();
    float const expected = 1.0F + std::numbers::e_v<float> + std::exp(2.0F);
    CHECK(result.value() == doctest::Approx(expected));
}

// Test exp function with log values (they should cancel out)
TEST_CASE("ParrotTest - ExpLogTest") {
    auto arr    = parrot::array<float>({1.0F, 2.0F, 3.0F});
    auto result = arr.log().exp();
    CHECK(result.to_host()[0] == doctest::Approx(1.0F));
    CHECK(result.to_host()[1] == doctest::Approx(2.0F));
    CHECK(result.to_host()[2] == doctest::Approx(3.0F));
}

// Test sqrt function with perfect squares
TEST_CASE("ParrotTest - SqrtPerfectSquaresTest") {
    auto arr    = parrot::array<float>({4.0F, 9.0F, 16.0F, 25.0F});
    auto result = arr.sqrt().sum();
    CHECK(result.value() == doctest::Approx(14.0F));  // sum of [2,3,4,5] = 14
}

// Test sqrt function with non-perfect squares
TEST_CASE("ParrotTest - SqrtTest") {
    auto arr             = parrot::array<float>({2.0F, 3.0F, 5.0F});
    auto result          = arr.sqrt().sum();
    float const expected = std::numbers::sqrt2_v<float> +
                           std::numbers::sqrt3_v<float> + std::sqrt(5.0F);
    CHECK(result.value() == doctest::Approx(expected));
}

// Test sq function (square)
TEST_CASE("ParrotTest - SqTest") {
    auto arr    = parrot::array({1, 2, 3, 4});
    auto result = arr.sq().sum();
    CHECK_EQ(result.value(), 30);  // sum of [1,4,9,16] = 30
}

// Test sq function with negative values
TEST_CASE("ParrotTest - SqNegativeTest") {
    auto arr    = parrot::array({-1, -2, 3});
    auto result = arr.sq().sum();
    CHECK_EQ(result.value(), 14);  // sum of [1,4,9] = 14
}

// Test odd function with mixed values
TEST_CASE("ParrotTest - OddTest") {
    auto arr    = parrot::array({1, 2, 3, 4, 5});
    auto result = arr.odd().sum();
    CHECK_EQ(result.value(), 3);  // 1,3,5 are odd, so sum of [1,0,1,0,1] = 3
}

// Test odd function with all even values
TEST_CASE("ParrotTest - OddAllEvenTest") {
    auto arr    = parrot::array({2, 4, 6, 8});
    auto result = arr.odd().sum();
    CHECK_EQ(result.value(), 0);  // all even, so sum of [0,0,0,0] = 0
}

// Test even function with mixed values
TEST_CASE("ParrotTest - EvenTest") {
    auto arr    = parrot::array({1, 2, 3, 4, 5});
    auto result = arr.even().sum();
    CHECK_EQ(result.value(), 2);  // 2,4 are even, so sum of [0,1,0,1,0] = 2
}

// Test even function with all odd values
TEST_CASE("ParrotTest - EvenAllOddTest") {
    auto arr    = parrot::array({1, 3, 5, 7});
    auto result = arr.even().sum();
    CHECK_EQ(result.value(), 0);  // all odd, so sum of [0,0,0,0] = 0
}

// Test rand with integer array
TEST_CASE("ParrotTest - RandIntTest") {
    auto arr         = parrot::array({10, 20, 30, 40});
    auto result      = arr.rand();
    auto result_host = result.to_host();
    auto arr_host    = arr.to_host();

    REQUIRE_EQ(result_host.size(), arr_host.size());
    for (size_t i = 0; i < result_host.size(); i++) {
        CHECK_GE(result_host[i], 0);
        // rand() generates in [0, N), so should be strictly less than N if N >
        // 0
        if (arr_host[i] > 0) {
            CHECK_LT(result_host[i], arr_host[i]);
        } else {
            CHECK_EQ(result_host[i], 0);  // rand(0) should be 0
        }
    }
}

// Test rand with floating point array
TEST_CASE("ParrotTest - RandFloatTest") {
    auto arr         = parrot::array<float>({10.0F, 20.0F, 0.0F, 40.0F});
    auto result      = arr.rand();
    auto result_host = result.to_host();
    auto arr_host    = arr.to_host();

    REQUIRE_EQ(result_host.size(), arr_host.size());
    for (size_t i = 0; i < result_host.size(); i++) {
        CHECK_GE(result_host[i], 0.0F);
        // randf() generates in [0, N), should be strictly less than N if N > 0
        if (arr_host[i] > 0.0F) {
            CHECK_LT(result_host[i], arr_host[i]);
        } else {
            CHECK_EQ(result_host[i], 0.0F);  // randf(0.0) should be 0.0
        }
    }
}