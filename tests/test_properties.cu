/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "parrot.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <type_traits>
#include "test_common.hpp"

namespace {

template <typename Array>
using properties_t = typename std::remove_cvref_t<Array>::properties_type;

struct parity {
    __host__ __device__ auto operator()(int value) const -> int {
        return value % 2;
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
    auto running_any = values.anys();
    auto running_all = values.alls();

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
