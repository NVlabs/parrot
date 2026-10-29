/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
#include <thrust/pair.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "test_common.hpp"

namespace {

template <typename T>
auto boolean_values(const std::vector<int>& input) {
    auto values = parrot::array(std::vector<T>(input.begin(), input.end()));
    if constexpr (std::is_same_v<T, bool>) {
        return values;
    } else {
        // Predicate results are numeric but have a known boolean domain.
        return values.eq(1);
    }
}

struct boolean_descending {
    template <typename T>
    __device__ auto operator()(T left, T right) const -> bool {
        return left > right;
    }
};

struct equivalent_booleans {
    template <typename T>
    __device__ auto operator()(T, T) const -> bool {
        return false;
    }
};

struct boolean_key {
    __device__ auto operator()(int value) const -> bool {
        return value % 2 != 0;
    }
};

template <typename T>
auto expected_runs(const std::vector<T>& input) {
    std::vector<thrust::pair<T, int>> runs;
    for (T value : input) {
        if (runs.empty() || runs.back().first != value) {
            runs.emplace_back(value, 1);
        } else {
            ++runs.back().second;
        }
    }
    return runs;
}

}  // namespace

TEST_CASE_TEMPLATE("Boolean algorithms handle empty, uniform and mixed inputs",
                   T,
                   bool,
                   int) {
    for (const auto& input :
         std::vector<std::vector<int>>{{},
                                       {0},
                                       {1},
                                       {0, 0, 0},
                                       {1, 1, 1},
                                       {1, 0},
                                       {1, 0, 1},
                                       {1, 0, 1, 0, 0, 1, 0}}) {
        CAPTURE(input.size());
        auto values    = boolean_values<T>(input);
        auto ascending = std::vector<T>(input.begin(), input.end());
        std::sort(ascending.begin(), ascending.end());
        auto descending = ascending;
        std::reverse(descending.begin(), descending.end());

        CHECK(values.sort().to_host() == ascending);
        CHECK(values.sort_by(parrot::lt{}).to_host() == ascending);
        CHECK(values.sort_by(parrot::gt{}).to_host() == descending);
        CHECK(values.sort_by(boolean_descending{}).to_host() == descending);
        auto equivalent = values.sort_by(equivalent_booleans{}).to_host();
        std::sort(equivalent.begin(), equivalent.end());
        CHECK(equivalent == ascending);

        auto distinct = ascending;
        distinct.erase(std::unique(distinct.begin(), distinct.end()),
                       distinct.end());
        CHECK(values.distinct().to_host() == distinct);
        CHECK(values.sort().distinct().to_host() == distinct);
        std::reverse(distinct.begin(), distinct.end());
        auto sorted_descending = values.sort_by(parrot::gt{});
        CHECK(sorted_descending.distinct().to_host() == distinct);

        CHECK(values.rle().to_host() ==
              expected_runs(std::vector<T>(input.begin(), input.end())));
        CHECK(values.sort().rle().to_host() == expected_runs(ascending));
        CHECK(sorted_descending.rle().to_host() == expected_runs(descending));
        // Reversal and boolean scans also establish natural ordering.
        CHECK(values.sort().rev().rle().to_host() == expected_runs(descending));
        CHECK(values.anys().rle().to_host() ==
              expected_runs(values.anys().to_host()));
        CHECK(values.alls().rle().to_host() ==
              expected_runs(values.alls().to_host()));

        int count = std::accumulate(input.begin(), input.end(), 0);
        auto sum  = values.sum();
        auto sums = values.sums();
        static_assert(std::is_same_v<typename decltype(sum)::value_type, int>);
        static_assert(std::is_same_v<typename decltype(sums)::value_type, int>);
        CHECK_EQ(sum.value(), count);
        std::vector<int> prefixes(input.size());
        std::partial_sum(input.begin(), input.end(), prefixes.begin());
        CHECK(sums.to_host() == prefixes);

        CHECK(values.maxs().to_host() == values.anys().to_host());
        CHECK(values.mins().to_host() == values.alls().to_host());
        CHECK(values.prods().to_host() == values.alls().to_host());
        CHECK_EQ(values.prod().value(), count == int(input.size()));
        if constexpr (std::is_same_v<T, bool>) {
            CHECK_EQ(values.maxr().value(), count != 0);
            CHECK_EQ(values.minr().value(), count == int(input.size()));
        }
        if (input.empty()) {
            CHECK_THROWS_AS(parrot::stats::mode(values), std::invalid_argument);
        } else {
            CHECK_EQ(parrot::stats::mode(values).value(),
                     count > int(input.size()) - count);
        }
    }
}

TEST_CASE_TEMPLATE("Boolean algorithms process only masked elements",
                   T,
                   bool,
                   int) {
    auto values = boolean_values<T>({1, 0, 1, 0, 1, 0, 0});
    // Nonzero mask values select an element once, regardless of magnitude.
    auto selected = values.keep(parrot::array({0, 2, 0, 0, -1, 0, 0}));
    CHECK_EQ(selected.size(), 2);
    CHECK(selected.sort().to_host() == std::vector<T>{0, 1});
    CHECK(selected.sort_by(parrot::gt{}).to_host() == std::vector<T>{1, 0});
    CHECK(selected.distinct().to_host() == std::vector<T>{0, 1});
    CHECK(selected.uniq().to_host() == std::vector<T>{0, 1});
    CHECK(selected.rle().to_host() == expected_runs(std::vector<T>{0, 1}));
    CHECK_EQ(selected.sum().value(), 1);
    CHECK(selected.sums().to_host() == std::vector<int>{0, 1});
    CHECK_EQ(parrot::stats::mode(selected).value(), 0);
    CHECK_EQ(selected.maxr().value(), 1);
    CHECK_EQ(selected.minr().value(), 0);
    CHECK_EQ(selected.prod().value(), 0);
    CHECK_EQ(selected.any().value(), 1);
    CHECK_EQ(selected.all().value(), 0);
    CHECK(selected.maxs().to_host() == std::vector<T>{0, 1});
    CHECK(selected.mins().to_host() == std::vector<T>{0, 0});
    CHECK(selected.prods().to_host() == std::vector<T>{0, 0});

    auto sorted_selected = values.sort().keep(
      parrot::array({0, 1, 0, 0, 1, 0, 0}));
    CHECK(sorted_selected.rle().to_host() ==
          expected_runs(std::vector<T>{0, 1}));
    CHECK(sorted_selected.distinct().to_host() == std::vector<T>{0, 1});

    auto all_removed = values.keep(parrot::array({0, 0, 0, 0, 0, 0, 0}));
    CHECK_EQ(all_removed.sort_by(parrot::lt{}).size(), 0);
    CHECK_EQ(all_removed.distinct().size(), 0);
    CHECK_EQ(all_removed.uniq().size(), 0);
    CHECK_EQ(all_removed.rle().size(), 0);
    CHECK_EQ(all_removed.sum().value(), 0);
    CHECK_EQ(all_removed.sums().size(), 0);
    CHECK_EQ(all_removed.prod().value(), 1);
    CHECK_EQ(all_removed.any().value(), 0);
    CHECK_EQ(all_removed.all().value(), 1);
    CHECK_THROWS_AS(parrot::stats::mode(all_removed), std::invalid_argument);

    auto merged_runs = boolean_values<T>({1, 0, 1}).keep(
      parrot::array({1, 0, 1}));
    CHECK(merged_runs.rle().to_host() == expected_runs(std::vector<T>{1, 1}));
    CHECK(merged_runs.uniq().to_host() == std::vector<T>{1});
    CHECK_EQ(parrot::stats::mode(merged_runs).value(), 1);
}

TEST_CASE("Boolean projections partition arbitrary payloads") {
    for (const auto& input : std::vector<std::vector<int>>{
           {}, {2}, {3}, {8, 2, 4}, {7, 3, 1}, {7, 2, 5, 4, 2, 1}}) {
        auto values = parrot::array(input);
        auto sorted = values.sort_by_key(boolean_key{});
        static_assert(decltype(sorted)::properties_type::order ==
                      parrot::order::unknown);
        auto host = sorted.to_host();
        CHECK(std::is_partitioned(
          host.begin(), host.end(), [](int value) { return value % 2 == 0; }));
        auto expected = input;
        std::sort(host.begin(), host.end());
        std::sort(expected.begin(), expected.end());
        CHECK(host == expected);
        expected.erase(std::unique(expected.begin(), expected.end()),
                       expected.end());
        CHECK(sorted.distinct().to_host() == expected);
    }
    auto masked = parrot::array({9, 4, 3, 2, 7})
                    .keep(parrot::array({0, 1, 0, 1, 1}));
    auto sorted = masked.sort_by_key(boolean_key{}).to_host();
    REQUIRE_EQ(sorted.size(), 3);
    CHECK_EQ(sorted[0] % 2, 0);
    CHECK_EQ(sorted[1] % 2, 0);
    CHECK_EQ(sorted[2], 7);
}

TEST_CASE("Boolean maximum keys select the first matching payload") {
    for (const auto& input : std::vector<std::vector<int>>{
           {}, {2}, {3}, {8, 2, 4}, {7, 3, 1}, {2, 4, 5, 7, 2, 1}}) {
        auto result = parrot::array(input).max_by_key(boolean_key{});
        if (input.empty()) {
            CHECK_EQ(result.size(), 0);
        } else {
            auto found = std::find_if(input.begin(),
                                      input.end(),
                                      [](int value) { return value % 2 != 0; });
            CHECK_EQ(result.value(),
                     found == input.end() ? input.front() : *found);
        }
    }
    auto masked = parrot::array({9, 4, 3, 5}).keep(parrot::array({0, 1, 0, 1}));
    CHECK_EQ(masked.max_by_key(boolean_key{}).value(), 5);
    auto lazy = parrot::range(4).add(1);
    CHECK_EQ(lazy.max_by_key(boolean_key{}).value(), 3);
    CHECK_EQ(lazy.take(0).max_by_key(boolean_key{}).size(), 0);
    auto all_removed = parrot::array({9, 4, 3, 5})
                         .keep(parrot::array({0, 0, 0, 0}));
    CHECK_EQ(all_removed.max_by_key(boolean_key{}).size(), 0);
}

TEST_CASE_TEMPLATE("Boolean sums count along matrix axes", T, bool, int) {
    auto values = boolean_values<T>({1, 1, 0, 1, 0, 1}).reshape({2, 3});
    CHECK(values.template sum<1>().to_host() == std::vector<int>{2, 1, 1});
    CHECK(values.template sum<2>().to_host() == std::vector<int>{2, 2});
    CHECK(values.template sums<1>().to_host() ==
          std::vector<int>{1, 1, 0, 2, 1, 1});
    CHECK(values.template sums<2>().to_host() ==
          std::vector<int>{1, 2, 2, 1, 1, 2});
    CHECK(values.template maxs<2>().to_host() ==
          std::vector<T>{1, 1, 1, 1, 1, 1});
    CHECK(values.template mins<2>().to_host() ==
          std::vector<T>{1, 1, 0, 1, 0, 0});
    CHECK(values.template prods<2>().to_host() ==
          std::vector<T>{1, 1, 0, 1, 0, 0});
}

TEST_CASE("Zero domains and constant booleans retain their guarantees") {
    auto zeros = parrot::range(5).map<parrot::properties<parrot::domain::zero>>(
      [] __host__ __device__(int) { return 0; });
    CHECK(zeros.distinct().to_host() == std::vector<int>{0});
    CHECK_EQ(zeros.sum().value(), 0);
    CHECK_EQ(parrot::stats::mode(zeros).value(), 0);
    CHECK(zeros.sort().rle().to_host() ==
          expected_runs(std::vector<int>(5, 0)));
    for (bool value : {false, true}) {
        auto constant = parrot::scalar(value).replicate(4);
        CHECK_EQ(constant.sum().value(), value ? 4 : 0);
        CHECK(constant.distinct().to_host() == std::vector<bool>{value});
        CHECK(constant.rle().to_host() ==
              expected_runs(std::vector<bool>(4, value)));
    }
    auto
      descending = boolean_values<int>({0, 1}).sort_by(parrot::gt{}).distinct();
    static_assert(
      parrot::is_descending_v<typename decltype(descending)::properties_type>);
    CHECK(descending.rle().to_host() == expected_runs(std::vector<int>{1, 0}));
    CHECK_THROWS_AS(parrot::stats::mode(parrot::array<int>({})),
                    std::invalid_argument);
}

TEST_CASE_TEMPLATE("Boolean counts span multiple GPU blocks", T, bool, int) {
    constexpr int n = 65537;
    std::vector<int> input(n);
    for (int i = 0; i < n; ++i) { input[i] = int(i % 3 == 0); }
    int const count = std::accumulate(input.begin(), input.end(), 0);
    auto values     = boolean_values<T>(input);
    CHECK_EQ(values.sum().value(), count);
    CHECK_EQ(values.sums().back(), count);
    CHECK_EQ(parrot::stats::mode(values).value(), 0);
    CHECK(values.distinct().to_host() == std::vector<T>{0, 1});
    auto sorted = values.sort();
    CHECK(sorted.rle().to_host() ==
          std::vector<thrust::pair<T, int>>{{T(0), n - count}, {T(1), count}});
}

TEST_CASE("Boolean mode counts independently of the storage width") {
    auto values = parrot::array(std::vector<unsigned char>(300, 1))
                    .map<parrot::properties<parrot::domain::boolean>>(
                      cuda::std::identity{});
    CHECK_EQ(parrot::stats::mode(values).value(), 1);
}
