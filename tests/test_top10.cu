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

// Test the rain water test with rev
TEST_CASE("Top 10 - #1 Rain Water") {
    auto arr    = parrot::array({0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1});
    auto result = arr.maxs().min(arr.rev().maxs().rev()).minus(arr).sum();
    CHECK_EQ(result.value(), 6);
}

// Test the MCO (Maximum Consecutive Ones) example
TEST_CASE("Top 10 - #2 Maximum Consecutive Ones (MCO)") {
    SUBCASE("Test case 1") {
        auto nums   = parrot::array({1, 1, 0, 1, 1, 1});
        auto result = nums.chunk_by_reduce(parrot::eq{}, parrot::add{}).maxr();
        CHECK_EQ(result.value(), 3);
    }

    SUBCASE("Test case 2") {
        auto nums   = parrot::array({1, 0, 1, 1, 0, 1});
        auto result = nums.chunk_by_reduce(parrot::eq{}, parrot::add{}).maxr();
        CHECK_EQ(result.value(), 2);
    }
}

// Test the LCIS (Longest Consecutive Increasing Subsequence) example
TEST_CASE("Top 10 - #3 LCIS") {
    SUBCASE("Test case 1") {
        auto nums   = parrot::array({1, 3, 5, 4, 7});
        auto result = nums.map_adj(parrot::lt{})
                        .chunk_by_reduce(parrot::eq{}, parrot::add{})
                        .maxr()
                        .add(1);
        CHECK_EQ(result.value(), 3);
    }

    SUBCASE("Test case 2") {
        auto nums   = parrot::array({2, 2, 2, 2, 2});
        auto result = nums.map_adj(parrot::lt{})
                        .chunk_by_reduce(parrot::eq{}, parrot::add{})
                        .maxr()
                        .add(1);
        CHECK_EQ(result.value(), 1);
    }
}

// Test the maximum gap example
TEST_CASE("Top 10 - #6 Maximum Gap") {
    SUBCASE("Test case 1") {
        auto nums   = parrot::array({3, 6, 9, 1});
        auto result = nums.append(nums.back()).sort().deltas().maxr();
        CHECK_EQ(result.value(), 3);
    }

    SUBCASE("Test case 2") {
        auto nums   = parrot::array({10});
        auto result = nums.append(nums.back()).sort().deltas().maxr();
        CHECK_EQ(result.value(), 0);
    }
}

// Test the maximum gap count example
TEST_CASE("Top 10 - #7 Maximum Gap Count") {
    SUBCASE("Test case 1") {
        auto nums   = parrot::array({3, 6, 9, 1});
        auto d      = nums.sort().deltas();
        auto result = (d.maxr() == d).sum();
        CHECK_EQ(result.value(), 2);
    }

    SUBCASE("Test case 2") {
        auto nums   = parrot::array({2, 5, 8, 1});
        auto d      = nums.sort().deltas();
        auto result = (d.maxr() == d).sum();
        CHECK_EQ(result.value(), 2);
    }

    SUBCASE("Test case 3") {
        auto nums   = parrot::array({10});
        auto d      = nums.sort().deltas();
        auto result = (d.maxr() == d).sum();
        CHECK_EQ(result.value(), 0);
    }
}

// Test the full sushi freshness example
TEST_CASE("Top 10 - #5 Sushi For Two") {
    SUBCASE("Test case 1") {
        auto sushi  = parrot::array({2, 2, 2, 1, 1, 2, 2});
        auto result = sushi.differ()
                        .where()
                        .prepend(0)
                        .append(sushi.size())
                        .deltas()
                        .map_adj(parrot::min{})
                        .dble()
                        .maxr();
        CHECK_EQ(result.value(), 4);
    }

    SUBCASE("Test case 2") {
        auto sushi  = parrot::array({1, 2, 1, 2, 1, 2});
        auto result = sushi.differ()
                        .where()
                        .prepend(0)
                        .append(sushi.size())
                        .deltas()
                        .map_adj(parrot::min{})
                        .dble()
                        .maxr();
        CHECK_EQ(result.value(), 2);
    }

    SUBCASE("Test case 3") {
        auto sushi  = parrot::array({2, 2, 1, 1, 1, 2, 2, 2, 2});
        auto result = sushi.differ()
                        .where()
                        .prepend(0)
                        .append(sushi.size())
                        .deltas()
                        .map_adj(parrot::min{})
                        .dble()
                        .maxr();
        CHECK_EQ(result.value(), 6);
    }
}

// Test the TCO (Three Consecutive Odds) example
TEST_CASE("Top 10 - #8 Three Consecutive Odds (TCO)") {
    SUBCASE("Test case 1") {
        auto arr    = parrot::array({2, 6, 4, 1});
        auto result = arr  //
                        .odd()
                        .chunk_by_reduce(parrot::eq{}, parrot::add{})
                        .maxr()
                        .gte(3);
        CHECK_EQ(result.value(), false);
    }

    SUBCASE("Test case 2") {
        auto arr    = parrot::array({1, 2, 34, 3, 4, 5, 7, 23, 12});
        auto result = arr  //
                        .odd()
                        .chunk_by_reduce(parrot::eq{}, parrot::add{})
                        .maxr()
                        .gte(3);
        CHECK_EQ(result.value(), true);
    }
}

// Test the skyline example
TEST_CASE("Top 10 - #9 Skyline") {
    auto heights = parrot::array({1, 0, 3, 2, 5, 4});
    auto result  = heights.maxs().uniq().size();
    auto result2 = heights.maxs().distinct().size();
    CHECK_EQ(result, 3);
    CHECK_EQ(result2, 3);
}

// Test the ocean view example
TEST_CASE("Top 10 - #10 Ocean View 3") {
    SUBCASE("Test case 1") {
        auto nums     = parrot::array({4, 2, 3, 1});
        auto result   = nums.rev().maxs().differ().prepend(1).rev().where();
        auto result2  = nums.rev().maxs().differ().rev().append(1).where();
        auto expected = parrot::array({1, 3, 4});
        check_match_eq(result, expected);
        check_match_eq(result2, expected);
    }

    SUBCASE("Test case 2") {
        auto nums     = parrot::array({4, 3, 2, 1});
        auto result   = nums.rev().maxs().differ().prepend(1).rev().where();
        auto result2  = nums.rev().maxs().differ().rev().append(1).where();
        auto expected = parrot::array({1, 2, 3, 4});
        check_match_eq(result, expected);
        check_match_eq(result2, expected);
    }

    SUBCASE("Test case 3") {
        auto nums     = parrot::array({1, 3, 2, 4});
        auto result   = nums.rev().maxs().differ().prepend(1).rev().where();
        auto result2  = nums.rev().maxs().differ().rev().append(1).where();
        auto expected = parrot::array({4});
        check_match_eq(result, expected);
        check_match_eq(result2, expected);
    }

    SUBCASE("Test case 4") {
        auto nums     = parrot::array({2, 2, 2, 2});
        auto result   = nums.rev().maxs().differ().prepend(1).rev().where();
        auto result2  = nums.rev().maxs().differ().rev().append(1).where();
        auto expected = parrot::array({4});
        check_match_eq(result, expected);
        check_match_eq(result2, expected);
    }
}
