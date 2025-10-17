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
#include "test_common.hpp"

// #2
auto minimum_cost(auto arr) {
    return arr.drop(1).sort().take(2).append(arr.front()).sum();
}

TEST_CASE("Divide Array in Subarrays") {
    SUBCASE("Example 1") {
        auto arr = parrot::array({1, 2, 3, 12});
        CHECK_EQ(minimum_cost(arr).value(), 6);
    }
    SUBCASE("Example 2") {
        auto arr = parrot::array({5, 4, 3});
        CHECK_EQ(minimum_cost(arr).value(), 12);
    }
    SUBCASE("Example 3") {
        auto arr = parrot::array({10, 3, 1, 1});
        CHECK_EQ(minimum_cost(arr).value(), 12);
    }
}

// #3
auto return_to_boundary_count(auto arr) {  //
    return arr.sums().eq(0).sum();
}

TEST_CASE("Ant on the Boundary") {
    SUBCASE("Example 1") {
        auto arr = parrot::array({2, 3, -5});
        CHECK_EQ(return_to_boundary_count(arr).value(), 1);
    }
    SUBCASE("Example 2") {
        auto arr = parrot::array({3, 2, -3, -4});
        CHECK_EQ(return_to_boundary_count(arr).value(), 0);
    }
}

// #13
auto max_ice_cream(auto arr, auto coins) {
    return arr.sort().sums().lte(coins).sum();
}

TEST_CASE("Max Ice Cream") {
    SUBCASE("Example 1") {
        auto arr   = parrot::array({1, 3, 2, 4, 1});
        auto coins = 7;
        CHECK_EQ(max_ice_cream(arr, coins).value(), 4);
    }
    SUBCASE("Example 2") {
        auto arr   = parrot::array({10, 6, 8, 7, 7, 8});
        auto coins = 5;
        CHECK_EQ(max_ice_cream(arr, coins).value(), 0);
    }
    SUBCASE("Example 3") {
        auto arr   = parrot::array({1, 6, 3, 1, 2, 5});
        auto coins = 20;
        CHECK_EQ(max_ice_cream(arr, coins).value(), 6);
    }
}

// Sneaky Number
auto sneaky_number(auto arr) {
    auto sorted = arr.sort();
    auto mask   = sorted.map_adj(parrot::eq{}).prepend(0);
    return sorted.keep(mask);
}

TEST_CASE("Sneaky Number") {
    SUBCASE("Example 1") {
        auto arr = parrot::array({0, 1, 1, 0});
        check_match_eq(sneaky_number(arr), parrot::array({0, 1}));
    }
    SUBCASE("Example 2") {
        auto arr = parrot::array({0, 3, 2, 1, 3, 2});
        check_match_eq(sneaky_number(arr), parrot::array({2, 3}));
    }
    SUBCASE("Example 3") {
        auto arr = parrot::array({7, 1, 5, 4, 3, 4, 6, 0, 9, 5, 8, 2});
        check_match_eq(sneaky_number(arr), parrot::array({4, 5}));
    }
}

// Test sign function
TEST_CASE("Sign Function") {
    SUBCASE("Mixed positive, negative, and zero") {
        auto arr = parrot::array({-3, -1, 0, 1, 5});
        check_match_eq(arr.sign(), parrot::array({-1, -1, 0, 1, 1}));
    }
    SUBCASE("All positive") {
        auto arr = parrot::array({1, 2, 3, 4});
        check_match_eq(arr.sign(), parrot::array({1, 1, 1, 1}));
    }
    SUBCASE("All negative") {
        auto arr = parrot::array({-1, -2, -3});
        check_match_eq(arr.sign(), parrot::array({-1, -1, -1}));
    }
    SUBCASE("All zeros") {
        auto arr = parrot::array({0, 0, 0});
        check_match_eq(arr.sign(), parrot::array({0, 0, 0}));
    }
}

// Duplicate Zero
auto duplicate_zero(auto arr) {
    auto mask = (arr == 0) + 1;
    return arr.replicate(mask).take(arr.size());
}

auto duplicate_zero2(auto arr) {
    auto mask = parrot::scalar(2) - arr.sign();
    return arr.replicate(mask).take(arr.size());
}

TEST_CASE("Duplicate Zero") {
    SUBCASE("Example 1") {
        auto arr = parrot::array({1, 0, 2, 3, 0, 4, 5, 0});
        check_match_eq(duplicate_zero(arr),
                       parrot::array({1, 0, 0, 2, 3, 0, 0, 4}));
        check_match_eq(duplicate_zero2(arr),
                       parrot::array({1, 0, 0, 2, 3, 0, 0, 4}));
    }
    SUBCASE("Example 2") {
        auto arr = parrot::array({1, 2, 3});
        check_match_eq(duplicate_zero(arr), parrot::array({1, 2, 3}));
        check_match_eq(duplicate_zero2(arr), parrot::array({1, 2, 3}));
    }
    SUBCASE("Example 3") {
        auto arr = parrot::array({1, 2, 3, 0});
        check_match_eq(duplicate_zero(arr), parrot::array({1, 2, 3, 0}));
        check_match_eq(duplicate_zero2(arr), parrot::array({1, 2, 3, 0}));
    }
    SUBCASE("Example 4") {
        auto arr = parrot::array({0, 0, 1, 2});
        check_match_eq(duplicate_zero(arr), parrot::array({0, 0, 0, 0}));
        check_match_eq(duplicate_zero2(arr), parrot::array({0, 0, 0, 0}));
    }
    SUBCASE("Example 5") {
        auto arr = parrot::array({1, 2, 0, 3, 4});
        check_match_eq(duplicate_zero(arr), parrot::array({1, 2, 0, 0, 3}));
        check_match_eq(duplicate_zero2(arr), parrot::array({1, 2, 0, 0, 3}));
    }
}

// Max distance
auto max_distance(auto a, auto b) {
    return a.outer(b, parrot::minus{}).abs().maxr();
}

TEST_CASE("Max Distance") {
    SUBCASE("Example 1") {
        auto a = parrot::array({4, 5, 7});
        auto b = parrot::array({9, 1, 3, 4});
        CHECK_EQ(max_distance(a, b).value(), 6);
    }
    SUBCASE("Example 2") {
        auto a = parrot::array({2, 3, 5, 4});
        auto b = parrot::array({3, 2, 5, 5, 8, 7});
        CHECK_EQ(max_distance(a, b).value(), 6);
    }
    SUBCASE("Example 3") {
        auto a = parrot::array({2, 1, 11, 3});
        auto b = parrot::array({2, 5, 10, 2});
        CHECK_EQ(max_distance(a, b).value(), 9);
    }
    SUBCASE("Example 4") {
        auto a = parrot::array({1, 2, 3});
        auto b = parrot::array({3, 2, 1});
        CHECK_EQ(max_distance(a, b).value(), 2);
    }
    SUBCASE("Example 5") {
        auto a = parrot::array({1, 0, 2, 3});
        auto b = parrot::array({5, 0});
        CHECK_EQ(max_distance(a, b).value(), 5);
    }
}

// Zero Friend - PWC 343.1

auto zero_friend(auto arr) { return (arr - 0).abs().minr(); }

TEST_CASE("Zero Friend") {
    SUBCASE("Example 1") {
        auto arr = parrot::array({4, 2, -1, 3, -2});
        CHECK_EQ(zero_friend(arr).value(), 1);
    }
    SUBCASE("Example 2") {
        auto arr = parrot::array({-5, 5, -3, 3, -1, 1});
        CHECK_EQ(zero_friend(arr).value(), 1);
    }
    SUBCASE("Example 3") {
        auto arr = parrot::array({7, -3, 0, 2, -8});
        CHECK_EQ(zero_friend(arr).value(), 0);
    }

    SUBCASE("Example 4") {
        auto arr = parrot::array({-2, -5, -1, -8});
        CHECK_EQ(zero_friend(arr).value(), 1);
    }
    SUBCASE("Example 5") {
        auto arr = parrot::array({-2, 2, -4, 4, -1, 1});
        CHECK_EQ(zero_friend(arr).value(), 1);
    }
}

// Team Champion - PWC 343.2

using namespace parrot::literals;

auto team_champion(auto arr) {
    auto sums = arr.sum(2_ic);
    return (sums.maxr() == sums).where().front() - 1;
}

TEST_CASE("Team Champion") {
    SUBCASE("Example 1") {
        auto arr = parrot::matrix({{0, 1, 1},  //
                                   {0, 0, 1},
                                   {0, 0, 0}});
        CHECK_EQ(team_champion(arr), 0);
    }
    SUBCASE("Example 2") {
        auto arr = parrot::matrix({{0, 1, 0, 0},  //
                                   {0, 0, 0, 0},
                                   {1, 1, 0, 0},
                                   {1, 1, 1, 0}});
        ;
        CHECK_EQ(team_champion(arr), 3);
    }

    SUBCASE("Example 3") {
        auto arr = parrot::matrix({{0, 1, 0, 1},  //
                                   {0, 0, 1, 1},
                                   {1, 0, 0, 0},
                                   {0, 0, 1, 0}});
        CHECK_EQ(team_champion(arr), 0);
    }

    SUBCASE("Example 4") {
        auto arr = parrot::matrix({{0, 1, 1},  //
                                   {0, 0, 0},
                                   {0, 1, 0}});
        CHECK_EQ(team_champion(arr), 0);
    }

    SUBCASE("Example 5") {
        auto arr = parrot::matrix({{0, 0, 0, 0, 0},
                                   {1, 0, 0, 0, 0},
                                   {1, 1, 0, 1, 1},
                                   {1, 1, 0, 0, 0},
                                   {1, 1, 0, 1, 0}});
        CHECK_EQ(team_champion(arr), 2);
    }
}
