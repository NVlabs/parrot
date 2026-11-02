/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>
#include "test_common.hpp"

namespace {

template <typename Array>
using properties_t = typename std::remove_cvref_t<Array>::properties_type;

struct parity {
    __host__ __device__ auto operator()(int value) const -> int {
        return value % 2;
    }
};

template <parrot::domain Domain = parrot::domain::unknown,
          parrot::order Order   = parrot::order::unknown,
          typename T>
auto property_values(const std::vector<T>& input) {
    auto values    = parrot::array(input);
    using iterator = decltype(values.begin());
    return parrot::fusion_array<iterator,
                                parrot::no_mask_t,
                                parrot::properties<Domain, Order>>(
      values.begin(), values.end(), values.storage());
}

template <typename Iterator, typename Properties>
auto without_properties(
  const parrot::fusion_array<Iterator, parrot::no_mask_t, Properties>& values) {
    return parrot::
      fusion_array<Iterator, parrot::no_mask_t, parrot::properties<>>(
        values.begin(), values.end(), values.storage(), values.shape());
}

struct absolute_less {
    __host__ __device__ auto operator()(int left, int right) const -> bool {
        return parrot::abs_functor<int>()(left) <
               parrot::abs_functor<int>()(right);
    }
};

}  // namespace

TEST_CASE("Compile-time property implications") {
    using positive = parrot::properties<parrot::domain::positive>;
    using negative = parrot::properties<parrot::domain::negative>;
    using zero     = parrot::properties<parrot::domain::zero>;
    using strict   = parrot::properties<parrot::domain::unknown,
                                        parrot::order::strictly_ascending>;

    static_assert(parrot::is_nonnegative_v<positive>);
    static_assert(parrot::is_nonzero_v<positive>);
    static_assert(parrot::is_nonpositive_v<negative>);
    static_assert(parrot::is_nonzero_v<negative>);
    static_assert(parrot::is_boolean_v<zero>);
    static_assert(parrot::is_nonnegative_v<zero>);
    static_assert(parrot::is_nonpositive_v<zero>);
    static_assert(parrot::is_ascending_v<strict>);
    static_assert(parrot::is_strict_v<strict>);
    static_assert(parrot::is_grouped_by_value_v<strict>);
}

TEST_CASE("Constructors establish properties") {
    auto booleans = parrot::array({true, false, true});
    auto integers = parrot::array({1, 2, 3});
    auto scalar   = parrot::scalar(42);
    auto range    = parrot::range(5);

    static_assert(parrot::is_boolean_v<properties_t<decltype(booleans)>>);
    static_assert(properties_t<decltype(integers)>::domain ==
                  parrot::domain::unknown);
    static_assert(properties_t<decltype(scalar)>::order ==
                  parrot::order::constant);
    static_assert(parrot::is_nonempty_v<properties_t<decltype(scalar)>>);
    static_assert(properties_t<decltype(range)>::domain ==
                  parrot::domain::positive);
    static_assert(properties_t<decltype(range)>::order ==
                  parrot::order::strictly_ascending);
    static_assert(parrot::is_nonempty_v<properties_t<decltype(range)>>);
}

TEST_CASE("Predicates retain numeric values and establish boolean domain") {
    auto values = parrot::array({1, 2, 3, 4});
    auto eq     = values.eq(2);
    auto neq    = values.neq(2);
    auto lt     = values.lt(2);
    auto lte    = values.lte(2);
    auto gt     = values.gt(2);
    auto gte    = values.gte(2);
    auto odd    = values.odd();
    auto even   = values.even();
    auto differ = values.differ();

    static_assert(std::is_same_v<typename decltype(eq)::value_type, int>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(eq)>>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(neq)>>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(lt)>>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(lte)>>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(gt)>>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(gte)>>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(odd)>>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(even)>>);
    static_assert(parrot::is_boolean_v<properties_t<decltype(differ)>>);

    CHECK_EQ(eq.sum().value(), 1);
    CHECK_EQ(odd.sum().value(), 2);
}

TEST_CASE("Ordering operations propagate natural order") {
    auto values      = parrot::array({3, 1, 2, 1});
    auto sorted      = values.sort();
    auto descending  = values.sort_by(parrot::gt{});
    auto projected   = values.sort_by_key(parity{});
    auto reversed    = sorted.rev();
    auto running_max = values.maxs();
    auto running_min = values.mins();
    auto running_any = values.eq(2).anys();
    auto running_all = values.eq(2).alls();

    static_assert(properties_t<decltype(sorted)>::order ==
                  parrot::order::ascending);
    static_assert(properties_t<decltype(descending)>::order ==
                  parrot::order::descending);
    static_assert(properties_t<decltype(projected)>::order ==
                  parrot::order::unknown);
    static_assert(properties_t<decltype(reversed)>::order ==
                  parrot::order::descending);
    static_assert(properties_t<decltype(running_max)>::order ==
                  parrot::order::ascending);
    static_assert(properties_t<decltype(running_min)>::order ==
                  parrot::order::descending);
    static_assert(parrot::is_boolean_v<properties_t<decltype(running_any)>>);
    static_assert(properties_t<decltype(running_any)>::order ==
                  parrot::order::ascending);
    static_assert(parrot::is_boolean_v<properties_t<decltype(running_all)>>);
    static_assert(properties_t<decltype(running_all)>::order ==
                  parrot::order::descending);
}

TEST_CASE("Stable selections preserve order conservatively") {
    auto range    = parrot::range(6);
    auto taken    = range.take(3);
    auto dropped  = range.drop(2);
    auto filtered = range.filter(
      [] __host__ __device__(int x) { return x % 2 != 0; });
    auto applied   = filtered.apply();
    auto reshaped  = range.reshape({2, 3});
    auto flattened = reshaped.flatten();

    static_assert(properties_t<decltype(taken)>::order ==
                  parrot::order::strictly_ascending);
    static_assert(!parrot::is_nonempty_v<properties_t<decltype(taken)>>);
    static_assert(properties_t<decltype(dropped)>::order ==
                  parrot::order::strictly_ascending);
    static_assert(properties_t<decltype(filtered)>::order ==
                  parrot::order::strictly_ascending);
    static_assert(!parrot::is_nonempty_v<properties_t<decltype(filtered)>>);
    static_assert(properties_t<decltype(applied)>::order ==
                  parrot::order::strictly_ascending);
    static_assert(properties_t<decltype(reshaped)>::order ==
                  parrot::order::strictly_ascending);
    static_assert(properties_t<decltype(flattened)>::order ==
                  parrot::order::strictly_ascending);

    CHECK(check_match(applied, parrot::array({1, 3, 5})));
}

TEST_CASE("Arbitrary transformations invalidate affected properties") {
    auto range      = parrot::range(5);
    auto mapped     = range.map([] __host__ __device__(int x) { return -x; });
    auto gathered   = range.gather(parrot::array({2, 0, 4}));
    auto transposed = range.reshape({1, 5}).transpose();

    static_assert(properties_t<decltype(mapped)>::domain ==
                  parrot::domain::unknown);
    static_assert(properties_t<decltype(mapped)>::order ==
                  parrot::order::unknown);
    static_assert(properties_t<decltype(gathered)>::order ==
                  parrot::order::unknown);
    static_assert(properties_t<decltype(transposed)>::domain ==
                  parrot::domain::positive);
    static_assert(properties_t<decltype(transposed)>::order ==
                  parrot::order::unknown);
}

TEST_CASE("Property-selected paths match generic behavior") {
    auto range = parrot::range(5);
    CHECK(check_match(range.sort(), parrot::array({1, 2, 3, 4, 5})));
    CHECK(check_match(range.maxs(), parrot::array({1, 2, 3, 4, 5})));
    CHECK(check_match(range.rev().mins(), parrot::array({5, 4, 3, 2, 1})));
    CHECK_EQ(range.minr().value(), 1);
    CHECK_EQ(range.maxr().value(), 5);

    auto predicate_values = parrot::array({3, 1, 4, 2}).lt(3);
    static_assert(
      std::is_same_v<typename decltype(predicate_values)::value_type, int>);
    CHECK(check_match(predicate_values.sort(), parrot::array({0, 0, 1, 1})));

    auto projected = parrot::array({2, 1, 2, 1}).sort_by_key(parity{});
    CHECK(check_match(projected.distinct(), parrot::array({1, 2})));

    // A known ordered lazy iterator can take the direct distinct/uniq path.
    auto lazy_distinct = range.distinct();
    static_assert(parrot::is_fusion_array_v<decltype(lazy_distinct)>);
    CHECK(check_match(lazy_distinct, range));
}

TEST_CASE("Integral constant algorithms match generic algorithms") {
    for (const auto& input : std::vector<std::vector<int>>{
           {}, {7}, {7, 7, 7, 7}, {0, 0, 0}, {-3, -3, -3}}) {
        auto values  = property_values<parrot::domain::unknown,
                                       parrot::order::constant>(input);
        auto generic = without_properties(values);
        CHECK(values.uniq().to_host() == generic.uniq().to_host());
        CHECK(values.distinct().to_host() == generic.distinct().to_host());
        CHECK(values.rle().to_host() == generic.rle().to_host());
        CHECK(values.uniq().storage() == values.storage());
        CHECK(values.rle().storage() == nullptr);
        CHECK_EQ(values.any().value(), generic.any().value());
        CHECK_EQ(values.all().value(), generic.all().value());
        CHECK(values.anys().to_host() == generic.anys().to_host());
        CHECK(values.alls().to_host() == generic.alls().to_host());
        if (input.empty()) {
            CHECK_THROWS_AS(parrot::stats::mode(values), std::invalid_argument);
            CHECK_THROWS_AS((void)values.minmax(), std::invalid_argument);
        } else {
            CHECK_EQ(parrot::stats::mode(values).value(), input.front());
            CHECK(values.minmax().value() == generic.minmax().value());
        }
    }
    auto matrix = parrot::matrix(7, {2, 3});
    CHECK(matrix.uniq().shape() == std::vector<int>{1});
    CHECK(matrix.distinct().shape() == std::vector<int>{1});
    CHECK(matrix.rle().shape() == std::vector<int>{1});
    CHECK(matrix.anys().shape() == std::vector<int>{2, 3});
    CHECK(matrix.anys().to_host() == std::vector<int>{7, 1, 1, 1, 1, 1});
    CHECK(matrix.alls<2>().to_host() == std::vector<int>{7, 1, 1, 7, 1, 1});
}

TEST_CASE("Strictly ordered values need no uniqueness mask or run reduction") {
    for (const auto& input :
         std::vector<std::vector<int>>{{}, {4}, {-5, -1, 0, 4, 9}}) {
        auto ascending = property_values<parrot::domain::unknown,
                                         parrot::order::strictly_ascending>(
          input);
        auto descending = ascending.rev();
        auto reversed   = input;
        std::reverse(reversed.begin(), reversed.end());
        CHECK(ascending.uniq().to_host() == input);
        CHECK(ascending.distinct().to_host() == input);
        CHECK(descending.uniq().to_host() == reversed);
        CHECK(descending.distinct().to_host() == reversed);
        CHECK(ascending.rle().to_host() ==
              without_properties(ascending).rle().to_host());
        CHECK(descending.rle().to_host() ==
              without_properties(descending).rle().to_host());
        CHECK(ascending.uniq().storage() == ascending.storage());
        CHECK(ascending.rle().storage() == ascending.storage());
        if (input.empty()) {
            CHECK_THROWS_AS(parrot::stats::mode(descending),
                            std::invalid_argument);
        } else {
            CHECK_EQ(parrot::stats::mode(ascending).value(), input.front());
            CHECK_EQ(parrot::stats::mode(descending).value(), input.front());
        }
    }
    auto matrix = parrot::range(6).reshape({2, 3});
    CHECK(matrix.uniq().shape() == std::vector<int>{6});
    CHECK(matrix.distinct().shape() == std::vector<int>{6});
    CHECK(matrix.rle().shape() == std::vector<int>{6});
}

TEST_CASE("Natural integral sorting reuses ordered storage") {
    for (const auto& input :
         std::vector<std::vector<int>>{{}, {4}, {1, 1, 3, 4, 4, 8}}) {
        auto ascending  = property_values<parrot::domain::unknown,
                                          parrot::order::ascending>(input);
        auto descending = ascending.rev();
        auto reversed   = input;
        std::reverse(reversed.begin(), reversed.end());
        CHECK(descending.sort().to_host() == input);
        CHECK(descending.sort().storage() == ascending.storage());
        CHECK(ascending.sort_by(parrot::lt{}).storage() == ascending.storage());
        CHECK(ascending.sort_by(parrot::gt{}).to_host() == reversed);
        CHECK(descending.sort_by(cuda::std::less<int>{}).to_host() == input);
        CHECK(descending.sort_by(cuda::std::greater<int>{}).to_host() ==
              reversed);
        CHECK(descending.to_host() == reversed);
        if (!input.empty()) {
            CHECK(descending.minmax().value() == ascending.minmax().value());
        }
    }
    auto matrix = parrot::range(6).rev().reshape({2, 3});
    CHECK(matrix.sort().shape() == std::vector<int>{6});
    CHECK(matrix.sort_by(parrot::gt{}).shape() == std::vector<int>{6});
    auto arbitrary = property_values<parrot::domain::unknown,
                                     parrot::order::ascending>(
                       std::vector<int>{-5, -1, 2, 4})
                       .sort_by(absolute_less{});
    static_assert(properties_t<decltype(arbitrary)>::order ==
                  parrot::order::unknown);
    CHECK(arbitrary.to_host() == std::vector<int>{-1, 2, 4, -5});
}

TEST_CASE("Ordered searches preserve first and last occurrence semantics") {
    for (const auto& input :
         std::vector<std::vector<int>>{{}, {4}, {-5, -5, -1, 0, 4, 4, 9}}) {
        auto ascending  = property_values<parrot::domain::unknown,
                                          parrot::order::ascending>(input);
        auto descending = ascending.rev();
        for (int value : {std::numeric_limits<int>::min(),
                          -6,
                          -5,
                          -1,
                          0,
                          1,
                          4,
                          9,
                          10,
                          std::numeric_limits<int>::max()}) {
            CHECK_EQ(ascending.index_of(value),
                     without_properties(ascending).index_of(value));
            CHECK_EQ(ascending.last_index_of(value),
                     without_properties(ascending).last_index_of(value));
            CHECK_EQ(descending.index_of(value),
                     without_properties(descending).index_of(value));
            CHECK_EQ(descending.last_index_of(value),
                     without_properties(descending).last_index_of(value));
        }
    }
    auto values = parrot::array({-5, -1, 0, 1, std::numeric_limits<int>::max()})
                    .sort();
    CHECK_EQ(values.index_of(std::numeric_limits<unsigned>::max()), 1);
    CHECK_EQ(values.last_index_of(std::numeric_limits<unsigned>::max()), 1);
    CHECK_EQ(values.index_of(std::int64_t(std::numeric_limits<int>::max()) + 1),
             -1);
    CHECK_EQ(values.index_of(0.5), -1);
    CHECK_EQ(values.index_of(-1LL), 1);
    CHECK_EQ(parrot::range(8).drop(2).take(3).index_of(4), 1);
    CHECK_EQ(parrot::scalar(7).repeat(5).last_index_of(7), 4);
}

TEST_CASE("Property shortcuts operate on logical masked values") {
    auto constant = parrot::matrix(7, {1, 5}).keep(
      parrot::array({0, 2, 0, -1, 0}));
    CHECK_EQ(constant.front(), 7);
    CHECK_EQ(constant.back(), 7);
    CHECK(constant.uniq().to_host() == std::vector<int>{7});
    CHECK(constant.distinct().to_host() == std::vector<int>{7});
    CHECK(constant.rle().to_host() ==
          std::vector<thrust::pair<int, int>>{{7, 2}});
    CHECK_EQ(parrot::stats::mode(constant).value(), 7);
    CHECK(constant.anys().to_host() == std::vector<int>{7, 1});
    CHECK(constant.anys().shape() == std::vector<int>{2});
    CHECK(constant.minmax().value() == thrust::make_pair(7, 7));

    auto ordered = parrot::array({-9, -5, 0, 1, 9})
                     .sort()
                     .keep(parrot::array({0, 1, 1, 0, 0}));
    CHECK_EQ(ordered.front(), -5);
    CHECK_EQ(ordered.back(), 0);
    CHECK(ordered.minmax().value() == thrust::make_pair(-5, 0));
    CHECK_EQ(ordered.index_of(0), 1);
    CHECK_EQ(ordered.last_index_of(0), 1);
    CHECK(ordered.sort_by(parrot::gt{}).to_host() == std::vector<int>{0, -5});
    auto unknown = parrot::array({100, 7, -3, -100})
                     .keep(parrot::array({0, 1, 1, 0}));
    CHECK(unknown.minmax().value() == thrust::make_pair(-3, 7));

    auto strict = parrot::range(5).keep(parrot::array({0, 1, 0, 1, 0}));
    CHECK(strict.uniq().to_host() == std::vector<int>{2, 4});
    CHECK(strict.rle().to_host() ==
          std::vector<thrust::pair<int, int>>{{2, 1}, {4, 1}});
    CHECK_EQ(parrot::stats::mode(strict.rev()).value(), 2);
    auto raw                  = parrot::range(6);
    auto mask                 = parrot::array({0, 1, 0, 1, 0, 0});
    using selected_properties = parrot::
      properties<parrot::domain::positive, parrot::order::strictly_ascending>;
    auto ranked = parrot::fusion_array<decltype(raw.begin()),
                                       decltype(mask.begin()),
                                       selected_properties>(
      raw.begin(),
      raw.end(),
      raw.storage(),
      std::vector<int>{2, 3},
      mask.begin(),
      mask.end(),
      mask.storage());
    CHECK_EQ(ranked.uniq().rank(), 1);
    CHECK(ranked.uniq().shape() == std::vector<int>{2});
    CHECK(ranked.uniq().to_host() == std::vector<int>{2, 4});
    auto removed = parrot::scalar(7).repeat(3).keep(parrot::array({0, 0, 0}));
    CHECK_EQ(removed.uniq().size(), 0);
    CHECK_EQ(removed.distinct().size(), 0);
    CHECK_EQ(removed.rle().size(), 0);
    CHECK_EQ(removed.any().value(), 0);
    CHECK_EQ(removed.all().value(), 1);
    CHECK_EQ(removed.anys().size(), 0);
    CHECK_THROWS_AS((void)removed.minmax(), std::invalid_argument);
    CHECK_THROWS_AS(parrot::stats::mode(removed), std::invalid_argument);
}

TEST_CASE_TEMPLATE("Zero domains reduce and scan without reading values",
                   T,
                   int,
                   bool) {
    for (int n : {0, 1, 6}) {
        auto values = property_values<parrot::domain::zero>(
          std::vector<T>(n, T(0)));
        using sum_value = typename decltype(values.sum())::value_type;
        static_assert(
          std::is_same_v<sum_value,
                         std::conditional_t<std::is_same_v<T, bool>, int, T>>);
        CHECK_EQ(values.sum().value(), 0);
        CHECK(values.sums().to_host() == std::vector<int>(n, 0));
        CHECK_EQ(values.any().value(), 0);
        CHECK_EQ(values.all().value(), n == 0);
        CHECK_EQ(values.where().size(), 0);
        CHECK(values.anys().to_host() == std::vector<T>(n, T(0)));
        CHECK(values.alls().to_host() == std::vector<T>(n, T(0)));
        CHECK(values.sign().to_host() == std::vector<int>(n, 0));
        if (n == 6) {
            auto matrix = values.reshape({2, 3});
            CHECK(matrix.sums().shape() == std::vector<int>{2, 3});
            CHECK(matrix.template sums<2>().to_host() ==
                  std::vector<int>(n, 0));
            CHECK_EQ(matrix.template sum<2>().size(), 2);
        }
    }
    auto selected = property_values<parrot::domain::zero>(
                      std::vector<T>(4, T(0)))
                      .keep(parrot::array({0, 2, 0, -1}));
    CHECK_EQ(selected.sum().value(), 0);
    CHECK(selected.sums().to_host() == std::vector<int>{0, 0});
    CHECK(selected.sums().shape() == std::vector<int>{2});
    CHECK_EQ(selected.where().size(), 0);
    CHECK_EQ(selected.all().value(), 0);
}

TEST_CASE("Zero reduction promotion matches narrow and wide numeric storage") {
    auto narrow = property_values<parrot::domain::zero>(
      std::vector<unsigned char>(4, 0));
    static_assert(std::is_same_v<typename decltype(narrow.sum())::value_type,
                                 unsigned char>);
    auto masked = narrow.keep(parrot::array({0, 1, 0, 1}));
    static_assert(
      std::is_same_v<typename decltype(masked.sum())::value_type, int>);
    static_assert(std::is_same_v<typename decltype(masked.sums())::value_type,
                                 unsigned char>);
    CHECK_EQ(masked.sum().value(), 0);
    auto wide = property_values<parrot::domain::zero>(
      std::vector<std::int64_t>(4, 0));
    static_assert(
      std::is_same_v<typename decltype(wide.sum())::value_type, std::int64_t>);
    CHECK_EQ(wide.sum().value(), 0);
}

TEST_CASE("Nonzero domains preserve truth scan first elements") {
    for (const auto& input :
         std::vector<std::vector<int>>{{}, {7}, {-4, 5, -1}, {7, 8, 9}}) {
        auto values  = property_values<parrot::domain::nonzero>(input);
        auto generic = without_properties(values);
        CHECK_EQ(values.any().value(), !input.empty());
        CHECK_EQ(values.all().value(), 1);
        CHECK(values.anys().to_host() == generic.anys().to_host());
        CHECK(values.alls().to_host() == generic.alls().to_host());
        std::vector<int> indices(input.size());
        std::iota(indices.begin(), indices.end(), 1);
        CHECK(values.where().to_host() == indices);
        static_assert(
          !parrot::is_boolean_v<properties_t<decltype(values.anys())>>);
        static_assert(properties_t<decltype(values.anys())>::order ==
                      parrot::order::unknown);
    }
    auto positive = parrot::range(6).reshape({2, 3});
    CHECK(positive.anys().to_host() == std::vector<int>{1, 1, 1, 1, 1, 1});
    CHECK(positive.anys<2>().to_host() == std::vector<int>{1, 1, 1, 4, 1, 1});
    CHECK(positive.where().to_host() == std::vector<int>{1, 2, 3, 4, 5, 6});
    auto masked = positive.flatten().keep(parrot::array({0, 1, 0, 0, 1, 0}));
    CHECK(masked.anys().to_host() == std::vector<int>{2, 1});
    CHECK(masked.where().to_host() == std::vector<int>{1, 2});
    auto removed = positive.flatten().keep(parrot::array({0, 0, 0, 0, 0, 0}));
    CHECK_EQ(removed.any().value(), 0);
    CHECK_EQ(removed.all().value(), 1);
    CHECK_EQ(removed.where().size(), 0);
    CHECK_EQ(parrot::array<int>({}).where().size(), 0);
    CHECK(parrot::array({0, 2, 0, -3}).where().to_host() ==
          std::vector<int>{2, 4});
    auto unknown_masked = parrot::array({0, 2, -3, 0})
                            .keep(parrot::array({0, 1, 1, 0}));
    CHECK(unknown_masked.where().to_host() == std::vector<int>{1, 2});
}

TEST_CASE("Truth conversion retains fractional and wide nonzero values") {
    auto fractions = parrot::array({0.25, -0.5});
    CHECK_EQ(fractions.any().value(), 1);
    CHECK_EQ(fractions.all().value(), 1);
    auto matrix = parrot::array({0.25, -0.5, 0.0, 0.125}).reshape({2, 2});
    CHECK(matrix.any<2>().to_host() == std::vector<double>{1, 1});
    CHECK(matrix.all<2>().to_host() == std::vector<double>{1, 0});
    auto wide = parrot::array(
      std::vector<std::uint64_t>{std::uint64_t(1) << 32});
    CHECK_EQ(wide.any().value(), 1);
    CHECK_EQ(wide.all().value(), 1);
    auto nan = parrot::array({std::numeric_limits<double>::quiet_NaN()});
    CHECK_EQ(nan.any().value(), 1);
    CHECK_EQ(nan.all().value(), 1);
    CHECK_EQ(parrot::array({-0.0, 0.0}).any().value(), 0);
}

TEST_CASE("Known nonzero indices support the largest representable size") {
    constexpr int n      = std::numeric_limits<int>::max();
    auto begin           = cuda::make_constant_iterator(7);
    using known_positive = parrot::
      properties<parrot::domain::positive, parrot::order::constant, true>;
    auto values  = parrot::fusion_array<decltype(begin),
                                        parrot::no_mask_t,
                                        known_positive>(begin, begin + n);
    auto indices = values.where();
    static_assert(std::is_same_v<typename decltype(indices)::value_type, int>);
    CHECK_EQ(indices.size(), n);
    CHECK_EQ(indices.front(), 1);
    CHECK_EQ(indices.back(), n);
}

TEST_CASE("Unproven truth scan values never claim boolean ordering") {
    auto values      = parrot::array({7, 0, 2});
    auto running_any = values.anys();
    auto running_all = values.alls();
    static_assert(!parrot::is_boolean_v<properties_t<decltype(running_any)>>);
    static_assert(properties_t<decltype(running_any)>::order ==
                  parrot::order::unknown);
    CHECK(running_any.to_host() == std::vector<int>{7, 1, 1});
    CHECK(running_all.to_host() == std::vector<int>{7, 0, 0});
    CHECK(running_any.rle().to_host() ==
          std::vector<thrust::pair<int, int>>{{7, 1}, {1, 2}});
    CHECK(running_any.distinct().to_host() == std::vector<int>{1, 7});
}

TEST_CASE("Sign transformations propagate only justified properties") {
    auto values = parrot::range(5);
    CHECK(values.abs().to_host() == std::vector<int>{1, 2, 3, 4, 5});
    auto negative = values.neg();
    static_assert(properties_t<decltype(negative)>::domain ==
                  parrot::domain::negative);
    static_assert(properties_t<decltype(negative)>::order ==
                  parrot::order::strictly_descending);
    CHECK(negative.to_host() == std::vector<int>{-1, -2, -3, -4, -5});
    auto signs = negative.sign();
    static_assert(properties_t<decltype(signs)>::order ==
                  parrot::order::constant);
    static_assert(std::is_same_v<typename decltype(signs)::value_type, int>);
    CHECK(signs.to_host() == std::vector<int>{-1, -1, -1, -1, -1});
    CHECK(signs.storage() == nullptr);
    auto nonnegative = property_values<parrot::domain::nonnegative,
                                       parrot::order::strictly_ascending>(
      std::vector<int>{0, 1, std::numeric_limits<int>::max()});
    CHECK(nonnegative.abs().storage() == nonnegative.storage());
    CHECK(nonnegative.neg().to_host() ==
          std::vector<int>{0, -1, -std::numeric_limits<int>::max()});
    auto boolean_signs = nonnegative.sign();
    static_assert(parrot::is_boolean_v<properties_t<decltype(boolean_signs)>>);
    static_assert(properties_t<decltype(boolean_signs)>::order ==
                  parrot::order::ascending);
    CHECK(boolean_signs.to_host() == std::vector<int>{0, 1, 1});
    auto unknown_sign = property_values<parrot::domain::unknown,
                                        parrot::order::strictly_ascending>(
      std::vector<int>{std::numeric_limits<int>::min(), -1, 0, 1});
    static_assert(properties_t<decltype(unknown_sign.neg())>::order ==
                  parrot::order::unknown);
    CHECK(unknown_sign.sign().to_host() == std::vector<int>{-1, -1, 0, 1});
    auto unsigned_values = property_values<parrot::domain::unknown,
                                           parrot::order::strictly_ascending>(
      std::vector<unsigned>{0, 1, std::numeric_limits<unsigned>::max()});
    CHECK(unsigned_values.abs().storage() == unsigned_values.storage());
    static_assert(properties_t<decltype(unsigned_values.neg())>::order ==
                  parrot::order::unknown);
    CHECK(unsigned_values.neg().to_host() ==
          std::vector<unsigned>{0, std::numeric_limits<unsigned>::max(), 1});
    auto selected = values.keep(parrot::array({0, 1, 0, 1, 0}));
    CHECK(selected.sign().to_host() == std::vector<int>{1, 1});
    CHECK(selected.neg().to_host() == std::vector<int>{-2, -4});
    CHECK(selected.abs().to_host() == std::vector<int>{2, 4});
}

TEST_CASE("Floating constant NaNs and signed zero retain generic behavior") {
    double const nan = std::numeric_limits<double>::quiet_NaN();
    auto values      = parrot::scalar(nan).repeat(3);
    CHECK_EQ(values.uniq().size(), 3);
    CHECK_EQ(values.distinct().size(), 3);
    auto runs = values.rle().to_host();
    REQUIRE_EQ(runs.size(), 3);
    for (auto run : runs) {
        CHECK(std::isnan(run.first));
        CHECK_EQ(run.second, 1);
    }
    CHECK(std::isnan(parrot::stats::mode(values).value()));
    auto zero = parrot::scalar(-0.0).repeat(3);
    CHECK(std::signbit(zero.abs().front()));
    auto result  = zero.minmax().value();
    auto generic = without_properties(zero).minmax().value();
    CHECK_EQ(std::signbit(result.first), std::signbit(generic.first));
    CHECK_EQ(std::signbit(result.second), std::signbit(generic.second));
    auto positive = property_values<parrot::domain::positive>(
      std::vector<double>{0.5, std::numeric_limits<double>::infinity()});
    CHECK(positive.sign().to_host() == std::vector<int>{1, 1});
}

TEST_CASE("Floating RLE preserves NaN boundaries and zero representatives") {
    double const nan      = std::numeric_limits<double>::quiet_NaN();
    double const infinity = std::numeric_limits<double>::infinity();
    for (const auto& input : std::vector<std::vector<double>>{{},
                                                              {nan},
                                                              {-0.0,
                                                               0.0,
                                                               infinity,
                                                               infinity,
                                                               nan,
                                                               nan,
                                                               -infinity,
                                                               -infinity,
                                                               0.0}}) {
        std::vector<thrust::pair<double, int>> expected;
        for (double value : input) {
            if (expected.empty() || expected.back().first != value) {
                expected.emplace_back(value, 1);
            } else {
                ++expected.back().second;
            }
        }
        auto actual = parrot::array(input).rle().to_host();
        REQUIRE_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            CHECK_EQ(actual[i].second, expected[i].second);
            if (std::isnan(expected[i].first)) {
                CHECK(std::isnan(actual[i].first));
            } else {
                CHECK_EQ(actual[i].first, expected[i].first);
                CHECK_EQ(std::signbit(actual[i].first),
                         std::signbit(expected[i].first));
            }
        }
    }
    auto selected = parrot::array({nan, 1.0, nan, 1.0, 1.0})
                      .keep(parrot::array({0, 1, 0, 1, 1}));
    CHECK(selected.rle().to_host() ==
          std::vector<thrust::pair<double, int>>{{1.0, 3}});
    std::vector<double> big(65537, 7.0);
    big[0] = big[32768] = big[32769] = big[65536] = nan;
    auto runs = parrot::array(big).rle().to_host();
    REQUIRE_EQ(runs.size(), 6);
    CHECK_EQ(runs[1].first, 7.0);
    CHECK_EQ(runs[1].second, 32767);
    CHECK_EQ(runs[4].first, 7.0);
    CHECK_EQ(runs[4].second, 32766);
}

TEST_CASE(
  "Lazy property results retain source storage and avoid output buffers") {
    auto runs = [] {
        return property_values<parrot::domain::unknown,
                               parrot::order::strictly_ascending>(
                 std::vector<int>{2, 4, 8})
          .rle();
    }();
    CHECK(runs.to_host() ==
          std::vector<thrust::pair<int, int>>{{2, 1}, {4, 1}, {8, 1}});
    auto sorted = [] {
        return property_values<parrot::domain::unknown,
                               parrot::order::descending>(
                 std::vector<int>{8, 4, 2})
          .sort();
    }();
    CHECK(sorted.to_host() == std::vector<int>{2, 4, 8});
    auto unique = [] { return parrot::scalar(7).repeat(100'000).uniq(); }();
    CHECK(unique.to_host() == std::vector<int>{7});
    auto big = parrot::range(100'000);
    CHECK_EQ(big.distinct().size(), 100'000);
    CHECK_EQ(big.rle().back().second, 1);
    CHECK(big.rle().storage() == nullptr);
    CHECK_EQ(parrot::stats::mode(big.rev()).value(), 1);
}
