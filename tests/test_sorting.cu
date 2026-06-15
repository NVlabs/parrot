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

#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <vector>
#include "test_common.hpp"

// Define a key function that extracts the parity (0 for even, 1 for odd)
struct EvenFirstFunctor {
    __host__ __device__ auto operator()(int x) const -> int { return x % 2; }
};

// Test sort function
TEST_CASE("ParrotTest - SortTest") {
    auto arr      = parrot::array({4, 2, 3, 1});
    auto sorted   = arr.sort();
    auto expected = parrot::array({1, 2, 3, 4});
    CHECK(check_match(sorted, expected));
    CHECK_EQ(sorted.sum().value(), arr.sum().value());  // Sanity check sum
}

// Test sort_by function with custom comparator
TEST_CASE("ParrotTest - SortByTest") {
    auto arr         = parrot::array({1, 4, 2, 3});
    auto sorted_desc = arr.sort_by(parrot::gt{});
    auto expected    = parrot::array({4, 3, 2, 1});
    CHECK(check_match(sorted_desc, expected));
    CHECK_EQ(sorted_desc.sum().value(),
             arr.sum().value());  // Check sum preservation
}

// Test sort_by_key function with a key extraction function
TEST_CASE("ParrotTest - SortByKeyTest") {
    auto arr = parrot::array({1, 4, 2, 3, 6, 5, 8, 7});

    // Sort by parity, which should group even numbers first, then odd numbers
    auto sorted = arr.sort_by_key(EvenFirstFunctor());

    // Check that even numbers come before odd numbers
    bool even_section = true;
    int count_even    = 0;
    int count_odd     = 0;
    auto sorted_host  = sorted
                          .to_host();  // Assuming to_host() returns std::vector

    REQUIRE_EQ(sorted_host.size(), arr.size());

    for (int const val : sorted_host) {
        if (val % 2 == 0) {
            count_even++;
            CHECK_MESSAGE(even_section, "Found even number after odd numbers");
        } else {
            count_odd++;
            even_section = false;  // Once we see an odd number, mark it
        }
    }

    // Verify we have the same count of odd and even numbers
    CHECK_EQ(count_even, 4);
    CHECK_EQ(count_odd, 4);

    // Verify total is the same
    CHECK_EQ(arr.sum().value(), sorted.sum().value());
}

// Check that a bool array sorts to all falses then all trues, with the number
// of trues preserved
static void check_bool_sorted(const std::vector<bool>& input) {
    auto arr    = parrot::array(input);
    auto sorted = arr.sort();
    auto host   = sorted.to_host();

    REQUIRE_EQ(host.size(), input.size());

    bool seen_true  = false;
    size_t num_true = 0;
    for (size_t i = 0; i < host.size(); ++i) {
        if (host[i]) {
            seen_true = true;
            ++num_true;
        } else {
            CHECK_MESSAGE(!seen_true,
                          "Found a false after a true (not sorted)");
        }
    }

    size_t expected_true = static_cast<size_t>(
      std::count(input.begin(), input.end(), true));
    CHECK_EQ(num_true, expected_true);
}

// Test sort function on a bool array (count + fill specialization)
TEST_CASE("ParrotTest - BoolSortMixed") {
    check_bool_sorted({true, false, true, false, false, true, false});
}

// Test bool sort with all false values
TEST_CASE("ParrotTest - BoolSortAllFalse") {
    check_bool_sorted({false, false, false, false});
}

// Test bool sort with all true values
TEST_CASE("ParrotTest - BoolSortAllTrue") {
    check_bool_sorted({true, true, true, true});
}

// Test bool sort with a single element
TEST_CASE("ParrotTest - BoolSortSingleElement") {
    check_bool_sorted({true});
    check_bool_sorted({false});
}

// Test bool sort with an empty array
TEST_CASE("ParrotTest - BoolSortEmpty") {
    auto arr    = parrot::array(std::vector<bool>{});
    auto sorted = arr.sort();
    CHECK_EQ(sorted.size(), 0);
}

// Test bool sort with an already-sorted array
TEST_CASE("ParrotTest - BoolSortAlreadySorted") {
    check_bool_sorted({false, false, true, true});
}

// Test sort on a non-bool array still uses the comparison sort
TEST_CASE("ParrotTest - SortNonBoolUnaffected") {
    auto arr      = parrot::array({3.5, 1.25, 2.75, 0.5});
    auto sorted   = arr.sort();
    auto expected = parrot::array({0.5, 1.25, 2.75, 3.5});
    CHECK(check_match(sorted, expected));
}

// Test uniq function for removing adjacent duplicates
TEST_CASE("ParrotTest - UniqTest") {
    auto arr    = parrot::array({1, 1, 2, 3, 3, 3, 4, 4, 1});
    auto result = arr.uniq();
    CHECK_EQ(result.size(), 5);
    auto expected_arr = parrot::array({1, 2, 3, 4, 1});
    CHECK(check_match(result, expected_arr));
}

// Test uniq function with no duplicates
TEST_CASE("ParrotTest - UniqNoDuplicatesTest") {
    auto arr    = parrot::array({1, 2, 3, 4, 5});
    auto result = arr.uniq();
    CHECK_EQ(result.size(), 5);
    CHECK(check_match(result, arr));
}

// Test uniq function with all duplicates
TEST_CASE("ParrotTest - UniqAllDuplicatesTest") {
    auto arr    = parrot::array({5, 5, 5, 5});
    auto result = arr.uniq();
    CHECK_EQ(result.size(), 1);
    CHECK_EQ(result.sum().value(), 5);
}