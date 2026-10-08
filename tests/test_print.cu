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

#include <device_atomic_functions.h>
#include <thrust/device_vector.h>
#include <sstream>
#include <type_traits>
#include <vector>
#include "parrot.hpp"
#include "test_common.hpp"

namespace {

struct print_gpu_probe {
    int *evaluations;

    __host__ __device__ auto operator()(int value) const -> int {
#ifdef __CUDA_ARCH__
        atomicAdd(evaluations, 1);
        return (value * 10) + 1;
#else
        // Different host results make CPU evaluation fail the output check.
        return -value;
#endif
    }
};

}  // namespace

TEST_CASE("Print - EvaluateLazyExpressionsOnceOnGPU") {
    auto check_print = [](const auto &input) -> void {
        thrust::device_vector<int> evaluations(1, 0);
        auto expression = input.map(print_gpu_probe{evaluations.data().get()});
        std::stringstream output;
        auto result = expression.print(output);

        static_assert(std::is_same_v<decltype(result), decltype(expression)>);
        CHECK_EQ(output.str(), "11 21 31 41\n");
        CHECK_EQ(static_cast<int>(evaluations.front()), 4);
        CHECK_EQ(result.shape(), expression.shape());
        CHECK_EQ(result.storage(), expression.storage());

        // The return value still supports chaining and keeps its input alive.
        std::stringstream chained_output;
        result.add(1).print(chained_output);
        CHECK_EQ(chained_output.str(), "12 22 32 42\n");
        CHECK_EQ(static_cast<int>(evaluations.front()), 8);
    };

    SUBCASE("Counting iterator") { check_print(parrot::range(4)); }
    SUBCASE("Device-backed iterator") {
        check_print(parrot::array({1, 2, 3, 4}));
    }
}

TEST_CASE("Print - FormattingAndEmptyArrays") {
    SUBCASE("Scalar") {
        std::stringstream output;
        auto result = parrot::scalar(42).print(output);
        CHECK_EQ(output.str(), "42\n");
        CHECK_EQ(result.value(), 42);
        CHECK_EQ(result.rank(), 0);
    }
    SUBCASE("Empty") {
        std::stringstream output;
        auto result = parrot::array<int>({}).print(output);
        CHECK_EQ(output.str(), "\n");
        CHECK_EQ(result.size(), 0);
    }
    SUBCASE("Padding and delimiter") {
        std::stringstream output;
        parrot::array({1, -20, 300}).print(output, " | ");
        CHECK_EQ(output.str(), "  1 | -20 | 300\n");
    }
    SUBCASE("Pairs") {
        std::stringstream output;
        parrot::array({1, 20}).pairs(parrot::array({3, 4})).print(output);
        CHECK_EQ(output.str(), " (1, 3) (20, 4)\n");
    }
    SUBCASE("Matrix") {
        std::stringstream output;
        auto result = parrot::array({1, -20, 300, 4})
                        .reshape({2, 2})
                        .print(output);
        CHECK_EQ(output.str(), "  1 -20\n300   4\n");
        CHECK_EQ(result.shape(), std::vector<int>{2, 2});
    }
    SUBCASE("Higher rank") {
        std::stringstream output;
        auto result = parrot::range(8).reshape({2, 2, 2}).print(output, ",");
        CHECK_EQ(output.str(), "1,2,3,4\n5,6,7,8\n");
        CHECK_EQ(result.shape(), std::vector<int>{2, 2, 2});
    }
    SUBCASE("Const device iterator") {
        const thrust::device_vector<int> values(3, 7);
        std::stringstream output;
        parrot::fusion_array(values.begin(), values.end()).print(output);
        CHECK_EQ(output.str(), "7 7 7\n");
    }
}

TEST_CASE("Print - MaskedArrays") {
    auto input = parrot::range(6).times(10);
    auto mask  = parrot::array({0, 1, 0, 1, 0, 1});
    SUBCASE("Selected elements") {
        std::stringstream output;
        auto result = input.keep(mask).print(output);
        CHECK_EQ(output.str(), "20 40 60\n");
        CHECK_EQ(result.to_host(), std::vector<int>{20, 40, 60});
        CHECK_EQ(result.shape(), std::vector<int>{3});
    }
    SUBCASE("All filtered out") {
        std::stringstream output;
        auto result = input.keep(mask.times(0)).print(output);
        CHECK_EQ(output.str(), "\n");
        CHECK_EQ(result.size(), 0);
    }
}

TEST_CASE("Print - SoftmaxRetainsShapeAndChainableResult") {
    using parrot::literals::operator""_ic;
    auto softmax = [](const auto &matrix) -> auto {
        auto cols        = matrix.ncols();
        auto z           = matrix - matrix.maxr(2_ic).replicate(cols);
        auto numerator   = z.exp();
        auto denominator = numerator.sum(2_ic);
        return numerator / denominator.replicate(cols);
    };
    auto matrix = parrot::range(6).as<float>().reshape({2, 3});
    std::stringstream output;
    auto result = softmax(matrix).print(output);
    CHECK_EQ(result.shape(), std::vector<int>{2, 3});

    // Parse the formatted output so padding is independent of float rounding.
    for (int row = 0; row < 2; ++row) {
        for (auto expected : {0.0900306, 0.244728, 0.665241}) {
            double value = 0;
            REQUIRE(static_cast<bool>(output >> value));
            CHECK_EQ(value, doctest::Approx(expected).epsilon(1e-5));
        }
    }
    CHECK_EQ(result.sum().value(), doctest::Approx(2.0F));
}
