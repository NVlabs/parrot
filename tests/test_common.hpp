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

#pragma once

#include <doctest/doctest.h>  // Using the correct path
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <cmath>  // Include for std::abs and doctest::Approx
#include <cstddef>
#include <limits>  // Include for std::numeric_limits
#include <sstream>
#include <stdexcept>  // Include for std::invalid_argument
#include <string>
#include <utility>
#include <vector>
#include "../parrot.hpp"

// Helper to check if two fusion_arrays match
template <typename T1, typename T2>
auto check_match(const T1& result, const T2& expected) -> bool {
    if (result.size() != expected.size()) { return false; }
    auto result_host   = result.to_host();
    auto expected_host = expected.to_host();
    for (size_t i = 0; i < result_host.size(); ++i) {
        if (result_host[i] != expected_host[i]) {
            // Allow approximate comparison for floats
            if (std::is_floating_point_v<decltype(result_host[i])> ||
                std::is_floating_point_v<decltype(expected_host[i])>) {
                if (!(doctest::Approx(result_host[i]) == expected_host[i])) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }
    return true;
}

template <typename T1, typename T2>
auto check_match_eq(const T1& result, const T2& expected) {
    auto result_host   = result.to_host();
    auto expected_host = expected.to_host();
    CHECK_EQ(result_host.size(), expected_host.size());
    for (size_t i = 0; i < result_host.size(); ++i) {
        CHECK_EQ(result_host[i], expected_host[i]);
    }
}