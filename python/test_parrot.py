import numpy as np
import pytest

import parrot


class TestParrot:
    """Test suite for parrot functionality."""

    def test_basic_range_sum(self):
        """Test basic range and sum operation."""
        result = parrot.range(10).sum()
        expected = sum(range(10))  # 0+1+2+...+9 = 45
        assert result == expected

    def test_range_with_multiplication_and_addition(self):
        """Test range with multiplication and addition."""
        result = parrot.range(5).times(2).add(1).sum()
        # range(5) = [0,1,2,3,4] -> times(2) = [0,2,4,6,8] -> add(1) = [1,3,5,7,9] -> sum = 25
        expected = sum(x * 2 + 1 for x in range(5))
        assert result == expected

    def test_complex_operations(self):
        """Test more complex operations with square."""
        result = parrot.range(10).add(1).sq().sum()
        # range(10) = [0,1,2,...,9] -> add(1) = [1,2,3,...,10] -> sq = [1,4,9,...,100] -> sum = 385
        expected = sum(x * x for x in range(1, 11))
        assert result == expected

    def test_product_operation(self):
        """Test product operation."""
        result = parrot.range(5).add(1).prod()
        # range(5) = [0,1,2,3,4] -> add(1) = [1,2,3,4,5] -> prod = 120
        expected = 1 * 2 * 3 * 4 * 5
        assert result == expected

    def test_complex_chaining(self):
        """Test complex chaining operations."""
        result = parrot.range(20).times(3).add(5).minus(2).max()
        # range(20) = [0,1,2,...,19] -> times(3) = [0,3,6,...,57] -> add(5) = [5,8,11,...,62] -> minus(2) = [3,6,9,...,60] -> max = 60
        values = [(x * 3 + 5 - 2) for x in range(20)]
        expected = max(values)
        assert result == expected

    def test_constant_iterator(self):
        """Test constant iterator."""
        result = parrot.constant(5, 10).sum()  # Sum of 10 fives = 50
        expected = 5 * 10
        assert result == expected

    def test_scan(self):
        """Test inclusive scan (cumulative sum)."""
        result = parrot.range(10).scan()
        # range(10) = [0,1,2,3,4,5,6,7,8,9] -> scan = [0,1,3,6,10,15,21,28,36,45]
        expected = []
        cumsum = 0
        for x in range(10):
            cumsum += x
            expected.append(cumsum)
        # Convert CuPy array to NumPy array for comparison
        result_numpy = result.get() if hasattr(result, "get") else result
        np.testing.assert_array_equal(result_numpy, expected)

    def test_collect_to_array(self):
        """Test collect to array."""
        result = parrot.range(5).times(2).collect()
        expected = [x * 2 for x in range(5)]
        result_numpy = result.get() if hasattr(result, "get") else result
        np.testing.assert_array_equal(result_numpy, expected)

    def test_map(self):
        """Test map transform."""
        result = parrot.range(10).map(lambda x: x * x + 1).sum()
        # range(10) = [0,1,2,...,9] -> map = [1,2,5,10,17,26,37,50,65,82] -> sum = 295
        expected = sum(x * x + 1 for x in range(10))
        assert result == expected

    def test_absolute_value(self):
        """Test absolute value."""
        result = parrot.range(10).add(-5).abs().sum()
        # range(10) = [0,1,2,3,4,5,6,7,8,9] -> add(-5) = [-5,-4,-3,-2,-1,0,1,2,3,4] -> abs = [5,4,3,2,1,0,1,2,3,4] -> sum = 25
        expected = sum(abs(x) for x in range(-5, 5))
        assert result == expected

    def test_min_operation(self):
        """Test min operation."""
        result = parrot.range(10).add(5).min()
        expected = 5  # Minimum of [5,6,7,8,9,10,11,12,13,14]
        assert result == expected

    def test_max_operation(self):
        """Test max operation."""
        result = parrot.range(10).add(5).max()
        expected = 14  # Maximum of [5,6,7,8,9,10,11,12,13,14]
        assert result == expected

    def test_idiv(self):
        """Test integer division."""
        result = parrot.array([7, 10, 15, 3]).idiv(4).to_host()
        assert result == [1, 2, 3, 0]

    def test_scalar_min(self):
        """Test element-wise minimum with a scalar (clamping upper bound)."""
        result = parrot.range(10).min(5).to_host()
        # [0,1,2,3,4,5,6,7,8,9] clamped to at most 5 => [0,1,2,3,4,5,5,5,5,5]
        assert result == [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]

    def test_scalar_max(self):
        """Test element-wise maximum with a scalar (clamping lower bound)."""
        result = parrot.range(10).max(5).to_host()
        # [0,1,2,3,4,5,6,7,8,9] clamped to at least 5 => [5,5,5,5,5,5,6,7,8,9]
        assert result == [5, 5, 5, 5, 5, 5, 6, 7, 8, 9]

    def test_clamp_via_min_max(self):
        """Test clamping to a range using min then max (or vice versa)."""
        result = parrot.range(10).min(7).max(3).to_host()
        # [0..9] => min(7) => [0,1,2,3,4,5,6,7,7,7] => max(3) => [3,3,3,3,4,5,6,7,7,7]
        assert result == [3, 3, 3, 3, 4, 5, 6, 7, 7, 7]

    def test_new_unary_maps(self):
        """Test new unary transformation methods."""
        # Test neg
        result = parrot.range(5).neg().collect()
        expected = [-x for x in range(5)]
        result_numpy = result.get() if hasattr(result, "get") else result
        np.testing.assert_array_equal(result_numpy, expected)

        # Test double
        result = parrot.range(5).double().sum()
        expected = sum(x * 2 for x in range(5))
        assert result == expected

        # Test even/odd
        result = parrot.range(10).even().sum()
        expected = sum(1 if x % 2 == 0 else 0 for x in range(10))
        assert result == expected

    def test_new_binary_maps(self):
        """Test new binary transformation methods."""
        # Test div - use float dtype to handle division properly
        result = (
            parrot.range(5, dtype=np.float32).add(1).div(2).sum()
        )  # [1,2,3,4,5] / 2 = [0.5,1,1.5,2,2.5]
        expected = sum((x + 1) / 2 for x in range(5))
        assert (
            abs(result - expected) < 1e-6
        )  # Use slightly larger tolerance for float32

        # Test gt
        result = parrot.range(10).gt(5).sum()  # Count elements > 5
        expected = sum(1 if x > 5 else 0 for x in range(10))
        assert result == expected

    def test_new_reductions(self):
        """Test new reduction methods."""
        # Test all/any with even numbers
        result = parrot.range(5).times(2).even().all()  # [0,2,4,6,8] all even
        assert result

        result = parrot.range(10).gt(5).any()  # Any > 5?
        assert result

        # Test custom reduce
        result = (
            parrot.range(5).add(1).reduce(lambda a, b: a * b, 1)
        )  # Product of [1,2,3,4,5]
        expected = 1 * 2 * 3 * 4 * 5
        assert result == expected

    def test_new_scans(self):
        """Test new scan methods."""
        # Test sums (should be same as regular scan)
        result1 = parrot.range(5).sums()
        result2 = parrot.range(5).scan()
        np.testing.assert_array_equal(
            result1.get() if hasattr(result1, "get") else result1,
            result2.get() if hasattr(result2, "get") else result2,
        )

        # Test prods
        result = parrot.range(5).add(1).prods()  # Cumulative product of [1,2,3,4,5]
        expected = [1, 1 * 2, 1 * 2 * 3, 1 * 2 * 3 * 4, 1 * 2 * 3 * 4 * 5]
        result_numpy = result.get() if hasattr(result, "get") else result
        np.testing.assert_array_equal(result_numpy, expected)

    def test_array_creation(self):
        """Test new array creation functions."""
        # Test scalar
        result = parrot.scalar(42).sum()
        assert result == 42

        # Test matrix
        result = parrot.matrix(2, 3, 5).sum()  # 2x3 matrix of 5s = 6*5 = 30
        assert result == 30

    def test_sum_of_squares(self):
        """Test sum of squares."""
        result = parrot.range(3).add(10).sq().sum()
        assert result == 365


def run_all_tests():
    """Run all tests manually without pytest for CUDA environment."""
    test_class = TestParrot()

    # Get all test methods
    test_methods = [method for method in dir(test_class) if method.startswith("test_")]

    passed = 0
    failed = 0

    print("Running Parrot Tests")
    print("=" * 40)

    for test_method in test_methods:
        try:
            method = getattr(test_class, test_method)
            method()
            print(f"✓ {test_method}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    # Can run with pytest if available, or manually
    try:
        import sys

        if len(sys.argv) > 1 and sys.argv[1] == "--manual":
            run_all_tests()
        else:
            pytest.main([__file__])
    except ImportError:
        print("pytest not available, running tests manually...")
        run_all_tests()

    def test_operator_add_scalar(self):
        """Test array + scalar."""
        result = (parrot.range(10) + 1).sum()
        expected = sum(range(1, 11))
        assert result == expected

    def test_operator_add_array(self):
        """Test array + array."""
        a = parrot.range(10)
        b = parrot.range(10)
        result = (a + b).collect()
        expected = [x * 2 for x in range(10)]
        np.testing.assert_array_equal(result.get(), expected)

    def test_drop(self):
        """Test drop operation."""
        result = parrot.range(11).drop(1).collect()
        expected = list(range(1, 11))
        np.testing.assert_array_equal(result.get(), expected)

    def test_polymorphic_chain(self):
        """Test the requested example: range(10) + 1 == range(11).drop(1)"""
        # Test logic equality
        left = parrot.range(10) + 1
        right = parrot.range(11).drop(1)

        # Check strict equality (all elements equal)
        # (left == right) returns a ParrotArray of bools
        equality = (left == right).all()
        assert equality is True

        # Check sum equality
        assert left.sum() == right.sum()

    def test_mixed_ops(self):
        """Test mixed operators."""
        a = parrot.range(5)
        b = parrot.constant(2, 5)
        # (0..4) * 2 + 1
        result = (a * b + 1).collect()
        expected = [x * 2 + 1 for x in range(5)]
        np.testing.assert_array_equal(result.get(), expected)


class TestSushiForTwo:
    """Test suite for the Sushi For Two problem from Codeforces.

    This tests the full chain of operations:
    sushi.differ().where().prepend(0).append(sushi.size()).deltas().map_adj(min).double().maxr()
    """

    def test_sushi_case_1(self):
        """Test case 1: [2, 2, 2, 1, 1, 2, 2] -> 4"""
        sushi = parrot.array([2, 2, 2, 1, 1, 2, 2])
        # fmt: off
        result = (sushi.differ()
                       .where()
                       .prepend(0)
                       .append(sushi.size())
                       .deltas()
                       .map_adj(parrot.min_op)
                       .double()
                       .maxr())
        # fmt: on
        assert result == 4

    def test_sushi_case_2(self):
        """Test case 2: [1, 2, 1, 2, 1, 2] -> 2"""
        sushi = parrot.array([1, 2, 1, 2, 1, 2])
        # fmt: off
        result = (sushi.differ()
                       .where()
                       .prepend(0)
                       .append(sushi.size())
                       .deltas()
                       .map_adj(parrot.min_op)
                       .double()
                       .maxr())
        # fmt: on
        assert result == 2

    def test_sushi_case_3(self):
        """Test case 3: [2, 2, 1, 1, 1, 2, 2, 2, 2] -> 6"""
        sushi = parrot.array([2, 2, 1, 1, 1, 2, 2, 2, 2])
        # fmt: off
        result = (sushi.differ()
                       .where()
                       .prepend(0)
                       .append(sushi.size())
                       .deltas()
                       .map_adj(parrot.min_op)
                       .double()
                       .maxr())
        # fmt: on
        assert result == 6


class TestRainWater:
    """Test the rain water trapping problem."""

    def test_rain_water(self):
        """Test case: [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1] -> 6"""
        arr = parrot.array([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
        # fmt: off
        result = (arr.maxs()
                     .min_with(arr.rev().maxs().rev())
                     - arr).sum()
        # fmt: on
        assert result == 6


class TestMaxConsecutiveOnes:
    """Test Maximum Consecutive Ones (MCO) problem."""

    def test_mco_case_1(self):
        """Test case 1: [1, 1, 0, 1, 1, 1] -> 3"""
        nums = parrot.array([1, 1, 0, 1, 1, 1])
        result = nums.chunk_by_reduce(parrot.eq_op, parrot.add_op).maxr()
        assert result == 3

    def test_mco_case_2(self):
        """Test case 2: [1, 0, 1, 1, 0, 1] -> 2"""
        nums = parrot.array([1, 0, 1, 1, 0, 1])
        result = nums.chunk_by_reduce(parrot.eq_op, parrot.add_op).maxr()
        assert result == 2


class TestLCIS:
    """Test Longest Consecutive Increasing Subsequence problem."""

    def test_lcis_case_1(self):
        """Test case 1: [1, 3, 5, 4, 7] -> 3"""
        nums = parrot.array([1, 3, 5, 4, 7])
        # fmt: off
        result = (nums.map_adj(parrot.lt_op)
                      .chunk_by_reduce(parrot.eq_op, parrot.add_op)
                      .maxr() + 1)
        # fmt: on
        assert result == 3

    def test_lcis_case_2(self):
        """Test case 2: [2, 2, 2, 2, 2] -> 1"""
        nums = parrot.array([2, 2, 2, 2, 2])
        # fmt: off
        result = (nums.map_adj(parrot.lt_op)
                      .chunk_by_reduce(parrot.eq_op, parrot.add_op)
                      .maxr() + 1)
        # fmt: on
        assert result == 1


class TestMaximumGap:
    """Test Maximum Gap problem."""

    def test_max_gap_case_1(self):
        """Test case 1: [3, 6, 9, 1] -> 3"""
        nums = parrot.array([3, 6, 9, 1])
        result = nums.append(nums.back()).sort().deltas().maxr()
        assert result == 3

    def test_max_gap_case_2(self):
        """Test case 2: [10] -> 0"""
        nums = parrot.array([10])
        result = nums.append(nums.back()).sort().deltas().maxr()
        assert result == 0


class TestMaximumGapCount:
    """Test Maximum Gap Count problem."""

    def test_max_gap_count_case_1(self):
        """Test case 1: [3, 6, 9, 1] -> 2"""
        nums = parrot.array([3, 6, 9, 1])
        d = nums.sort().deltas()
        result = (d == d.maxr()).sum()
        assert result == 2

    def test_max_gap_count_case_2(self):
        """Test case 2: [2, 5, 8, 1] -> 2"""
        nums = parrot.array([2, 5, 8, 1])
        d = nums.sort().deltas()
        result = (d == d.maxr()).sum()
        assert result == 2

    def test_max_gap_count_case_3(self):
        """Test case 3: [10] -> 0"""
        nums = parrot.array([10])
        d = nums.sort().deltas()
        result = (d == d.maxr()).sum()
        assert result == 0


class TestThreeConsecutiveOdds:
    """Test Three Consecutive Odds (TCO) problem."""

    def test_tco_case_1(self):
        """Test case 1: [2, 6, 4, 1] -> False"""
        arr = parrot.array([2, 6, 4, 1])
        # fmt: off
        result = (arr.odd()
                     .chunk_by_reduce(parrot.eq_op, parrot.add_op)
                     .maxr() >= 3)
        # fmt: on
        assert result is False

    def test_tco_case_2(self):
        """Test case 2: [1, 2, 34, 3, 4, 5, 7, 23, 12] -> True"""
        arr = parrot.array([1, 2, 34, 3, 4, 5, 7, 23, 12])
        # fmt: off
        result = (arr.odd()
                     .chunk_by_reduce(parrot.eq_op, parrot.add_op)
                     .maxr() >= 3)
        # fmt: on
        assert result is True


class TestSkyline:
    """Test Skyline problem."""

    def test_skyline(self):
        """Test: [1, 0, 3, 2, 5, 4] -> 3 unique max values"""
        heights = parrot.array([1, 0, 3, 2, 5, 4])
        result1 = heights.maxs().uniq().size()
        result2 = heights.maxs().distinct().size()
        assert result1 == 3
        assert result2 == 3


class TestOceanView:
    """Test Ocean View problem."""

    def test_ocean_view_case_1(self):
        """Test case 1: [4, 2, 3, 1] -> [1, 3, 4]"""
        nums = parrot.array([4, 2, 3, 1])
        result1 = nums.rev().maxs().differ().prepend(1).rev().where()
        result2 = nums.rev().maxs().differ().rev().append(1).where()
        expected = parrot.array([1, 3, 4])
        assert result1.match(expected)
        assert result2.match(expected)

    def test_ocean_view_case_2(self):
        """Test case 2: [4, 3, 2, 1] -> [1, 2, 3, 4]"""
        nums = parrot.array([4, 3, 2, 1])
        result1 = nums.rev().maxs().differ().prepend(1).rev().where()
        result2 = nums.rev().maxs().differ().rev().append(1).where()
        expected = parrot.array([1, 2, 3, 4])
        assert result1.match(expected)
        assert result2.match(expected)

    def test_ocean_view_case_3(self):
        """Test case 3: [1, 3, 2, 4] -> [4]"""
        nums = parrot.array([1, 3, 2, 4])
        result1 = nums.rev().maxs().differ().prepend(1).rev().where()
        result2 = nums.rev().maxs().differ().rev().append(1).where()
        expected = parrot.array([4])
        assert result1.match(expected)
        assert result2.match(expected)

    def test_ocean_view_case_4(self):
        """Test case 4: [2, 2, 2, 2] -> [4]"""
        nums = parrot.array([2, 2, 2, 2])
        result1 = nums.rev().maxs().differ().prepend(1).rev().where()
        result2 = nums.rev().maxs().differ().rev().append(1).where()
        expected = parrot.array([4])
        assert result1.match(expected)
        assert result2.match(expected)


# ============================================================================
# Additional tests converted from C++ parrot test suite
# ============================================================================


class TestWhere:
    """Tests for the where() function."""

    def test_where_all(self):
        """Test where with all non-zeros."""
        arr = parrot.array([1, 1, 1, 1])
        indices = arr.where()
        assert indices.size() == 4
        expected = parrot.array([1, 2, 3, 4])
        assert indices.match(expected)

    def test_where_some(self):
        """Test where with some zeros."""
        arr = parrot.array([0, 1, 0, 1])
        indices = arr.where()
        assert indices.size() == 2
        expected = parrot.array([2, 4])
        assert indices.match(expected)

    def test_where_none(self):
        """Test where with all zeros."""
        arr = parrot.array([0, 0, 0, 0])
        indices = arr.where()
        assert indices.size() == 0


class TestKeep:
    """Tests for the keep() function."""

    def test_keep_simple(self):
        """Test keep with simple mask."""
        arr = parrot.array([1, 2, 3])
        mask = parrot.array([1, 0, 1])
        result = arr.keep(mask)
        expected = parrot.array([1, 3])
        assert result.match(expected)

    def test_keep_all_zeros(self):
        """Test keep with all zeros mask."""
        arr = parrot.array([1, 2, 3])
        mask = parrot.array([0, 0, 0])
        result = arr.keep(mask)
        assert result.size() == 0

    def test_keep_all_ones(self):
        """Test keep with all ones mask."""
        arr = parrot.array([1, 2, 3])
        mask = parrot.array([1, 1, 1])
        result = arr.keep(mask)
        assert result.match(arr)


class TestMatch:
    """Tests for the match() function."""

    def test_match_identical(self):
        """Test match with identical arrays."""
        arr1 = parrot.array([1, 2, 3, 4])
        arr2 = parrot.array([1, 2, 3, 4])
        assert arr1.match(arr2)

    def test_match_different(self):
        """Test match with different arrays."""
        arr1 = parrot.array([1, 2, 3, 4])
        arr2 = parrot.array([1, 2, 3, 5])
        assert not arr1.match(arr2)

    def test_match_different_length(self):
        """Test match with different length arrays."""
        arr1 = parrot.array([1, 2, 3])
        arr2 = parrot.array([1, 2, 3, 4])
        assert not arr1.match(arr2)


class TestTake:
    """Tests for the take() function."""

    def test_take_basic(self):
        """Test take with valid size."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.take(3)
        assert result.size() == 3
        expected = parrot.array([1, 2, 3])
        assert result.match(expected)

    def test_take_full_size(self):
        """Test take with full size."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.take(5)
        assert result.size() == 5
        assert result.match(arr)

    def test_take_zero(self):
        """Test take with zero size."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.take(0)
        assert result.size() == 0

    def test_take_invalid(self):
        """Test take with invalid size."""
        arr = parrot.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            arr.take(6)
        with pytest.raises(ValueError):
            arr.take(-1)


class TestDrop:
    """Tests for the drop() function."""

    def test_drop_basic(self):
        """Test drop 2 elements."""
        arr = parrot.array([1, 2, 3, 4, 5])
        dropped = arr.drop(2)
        assert dropped.size() == 3
        expected = parrot.array([3, 4, 5])
        assert dropped.match(expected)

    def test_drop_zero(self):
        """Test drop 0 elements."""
        arr = parrot.array([1, 2, 3, 4, 5])
        dropped = arr.drop(0)
        assert dropped.size() == 5

    def test_drop_all(self):
        """Test drop all elements."""
        arr = parrot.array([1, 2, 3, 4, 5])
        dropped = arr.drop(5)
        assert dropped.size() == 0


class TestFilter:
    """Tests for the filter() function."""

    def test_filter_even(self):
        """Test filter for even numbers."""
        arr = parrot.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        evens = arr.filter(lambda x: x % 2 == 0)
        expected = parrot.array([2, 4, 6, 8, 10])
        assert evens.match(expected)

    def test_filter_greater_than(self):
        """Test filter for numbers greater than 5."""
        arr = parrot.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        greater = arr.filter(lambda x: x > 5)
        expected = parrot.array([6, 7, 8, 9, 10])
        assert greater.match(expected)


class TestRev:
    """Tests for the rev() function."""

    def test_rev_basic(self):
        """Test basic reverse."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.rev()
        expected = parrot.array([5, 4, 3, 2, 1])
        assert result.match(expected)
        assert result.sum() == arr.sum()

    def test_rev_chained(self):
        """Test reverse with chained operations."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.rev().times(2)
        expected = parrot.array([10, 8, 6, 4, 2])
        assert result.match(expected)

    def test_rev_double(self):
        """Test double reverse returns original."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.rev().rev()
        assert result.match(arr)


class TestSort:
    """Tests for the sort() function."""

    def test_sort_basic(self):
        """Test basic sort."""
        arr = parrot.array([4, 2, 3, 1])
        result = arr.sort()
        expected = parrot.array([1, 2, 3, 4])
        assert result.match(expected)
        assert result.sum() == arr.sum()


class TestUniq:
    """Tests for the uniq() function."""

    def test_uniq_basic(self):
        """Test basic uniq."""
        arr = parrot.array([1, 1, 2, 3, 3, 3, 4, 4, 1])
        result = arr.uniq()
        assert result.size() == 5
        expected = parrot.array([1, 2, 3, 4, 1])
        assert result.match(expected)

    def test_uniq_no_duplicates(self):
        """Test uniq with no duplicates."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.uniq()
        assert result.size() == 5
        assert result.match(arr)

    def test_uniq_all_duplicates(self):
        """Test uniq with all duplicates."""
        arr = parrot.array([5, 5, 5, 5])
        result = arr.uniq()
        assert result.size() == 1
        assert result.sum() == 5


class TestSign:
    """Tests for the sign() function."""

    def test_sign_mixed(self):
        """Test sign with mixed positive, negative, and zero."""
        arr = parrot.array([-3, -1, 0, 1, 5])
        expected = parrot.array([-1, -1, 0, 1, 1])
        assert arr.sign().match(expected)

    def test_sign_all_positive(self):
        """Test sign with all positive."""
        arr = parrot.array([1, 2, 3, 4])
        expected = parrot.array([1, 1, 1, 1])
        assert arr.sign().match(expected)

    def test_sign_all_negative(self):
        """Test sign with all negative."""
        arr = parrot.array([-1, -2, -3])
        expected = parrot.array([-1, -1, -1])
        assert arr.sign().match(expected)

    def test_sign_all_zeros(self):
        """Test sign with all zeros."""
        arr = parrot.array([0, 0, 0])
        expected = parrot.array([0, 0, 0])
        assert arr.sign().match(expected)


class TestGather:
    """Tests for the gather() function."""

    def test_gather_basic(self):
        """Test basic gather."""
        source = parrot.array([10, 20, 30, 40, 50, 60])
        indices = parrot.array([0, 1, 3, 5])
        result = source.gather(indices)
        assert result.sum() == 130  # 10 + 20 + 40 + 60


class TestIntegration:
    """Integration tests - real algorithm problems."""

    def test_minimum_cost(self):
        """Divide Array in Subarrays - minimum cost."""

        def minimum_cost(arr):
            return arr.drop(1).sort().take(2).append(arr.front()).sum()

        arr1 = parrot.array([1, 2, 3, 12])
        assert minimum_cost(arr1) == 6

        arr2 = parrot.array([5, 4, 3])
        assert minimum_cost(arr2) == 12

        arr3 = parrot.array([10, 3, 1, 1])
        assert minimum_cost(arr3) == 12

    def test_return_to_boundary_count(self):
        """Ant on the Boundary - count returns to boundary."""

        def return_to_boundary_count(arr):
            return (arr.sums() == 0).sum()

        arr1 = parrot.array([2, 3, -5])
        assert return_to_boundary_count(arr1) == 1

        arr2 = parrot.array([3, 2, -3, -4])
        assert return_to_boundary_count(arr2) == 0

    def test_max_ice_cream(self):
        """Max Ice Cream - maximum bars you can buy."""

        def max_ice_cream(arr, coins):
            return (arr.sort().sums().lte(coins)).sum()

        arr1 = parrot.array([1, 3, 2, 4, 1])
        assert max_ice_cream(arr1, 7) == 4

        arr2 = parrot.array([10, 6, 8, 7, 7, 8])
        assert max_ice_cream(arr2, 5) == 0

        arr3 = parrot.array([1, 6, 3, 1, 2, 5])
        assert max_ice_cream(arr3, 20) == 6

    def test_zero_friend(self):
        """Zero Friend - closest to zero."""

        def zero_friend(arr):
            return (arr - 0).abs().minr()

        arr1 = parrot.array([4, 2, -1, 3, -2])
        assert zero_friend(arr1) == 1

        arr2 = parrot.array([-5, 5, -3, 3, -1, 1])
        assert zero_friend(arr2) == 1

        arr3 = parrot.array([7, -3, 0, 2, -8])
        assert zero_friend(arr3) == 0

        arr4 = parrot.array([-2, -5, -1, -8])
        assert zero_friend(arr4) == 1

    def test_check_order(self):
        """Check Order - indices where sorted differs from original."""

        def check_order(ints):
            return ints.sort().neq(ints).where().minus(1)

        ints1 = parrot.array([5, 2, 4, 3, 1])
        expected1 = parrot.array([0, 2, 3, 4])
        assert check_order(ints1).match(expected1)

        ints2 = parrot.array([1, 2, 1, 1, 3])
        expected2 = parrot.array([1, 3])
        assert check_order(ints2).match(expected2)

        ints3 = parrot.array([3, 1, 3, 2, 3])
        expected3 = parrot.array([0, 1, 3])
        assert check_order(ints3).match(expected3)

    def test_chained_example(self):
        """Test chained operations from integration tests."""
        arr = parrot.array([3, 6, 9, 1])
        result = arr.append(1).sort().deltas().maxr()
        assert result == 3


class TestScans:
    """Tests for scan operations."""

    def test_sums(self):
        """Test cumulative sum."""
        arr = parrot.array([1, 2, 3, 4])
        result = arr.sums()
        expected = parrot.array([1, 3, 6, 10])
        assert result.match(expected)

    def test_prods(self):
        """Test cumulative product."""
        arr = parrot.array([1, 2, 3, 4])
        result = arr.prods()
        expected = parrot.array([1, 2, 6, 24])
        assert result.match(expected)

    def test_mins(self):
        """Test cumulative minimum."""
        arr = parrot.array([3, 1, 4, 2])
        result = arr.mins()
        expected = parrot.array([3, 1, 1, 1])
        assert result.match(expected)

    def test_maxs(self):
        """Test cumulative maximum."""
        arr = parrot.array([3, 1, 4, 2])
        result = arr.maxs()
        expected = parrot.array([3, 3, 4, 4])
        assert result.match(expected)


class TestReductions:
    """Tests for reduction operations."""

    def test_deltas(self):
        """Test deltas (differences)."""
        arr = parrot.array([1, 3, 6, 10])
        result = arr.deltas().sum()
        assert result == 9  # 2 + 3 + 4

    def test_maxr(self):
        """Test maximum reduction."""
        arr = parrot.array([1, 5, 3, 2])
        assert arr.maxr() == 5

    def test_minr(self):
        """Test minimum reduction."""
        arr = parrot.array([1, 5, 3, 2])
        assert arr.minr() == 1

    def test_minr_negative(self):
        """Test minimum with negative values."""
        arr = parrot.array([-1, -5, 3, 2])
        assert arr.minr() == -5

    def test_any_all_zeros(self):
        """Test any with all zeros."""
        arr = parrot.array([0, 0, 0, 0])
        assert not arr.any()

    def test_any_some_nonzeros(self):
        """Test any with some non-zeros."""
        arr = parrot.array([0, 0, 3, 0])
        assert arr.any()

    def test_all_nonzeros(self):
        """Test all with all non-zeros."""
        arr = parrot.array([1, 2, 3, 4])
        assert arr.all()

    def test_all_some_zeros(self):
        """Test all with some zeros."""
        arr = parrot.array([1, 0, 3, 4])
        assert not arr.all()

    def test_prod(self):
        """Test product reduction."""
        arr = parrot.array([1, 2, 3, 4])
        assert arr.prod() == 24


class TestMathOperations:
    """Tests for math operations."""

    def test_double(self):
        """Test double (multiply by 2)."""
        arr = parrot.array([1, 2, 3, 4])
        assert arr.double().sum() == 20

    def test_half(self):
        """Test half (divide by 2)."""
        arr = parrot.array([2, 4, 6, 8])
        assert arr.half().sum() == 10

    def test_abs(self):
        """Test absolute value."""
        arr = parrot.array([-2, 3, -5, 7])
        assert arr.abs().sum() == 17

    def test_sq(self):
        """Test square."""
        arr = parrot.array([1, 2, 3, 4])
        assert arr.sq().sum() == 30

    def test_odd(self):
        """Test odd check."""
        arr = parrot.array([1, 2, 3, 4, 5])
        assert arr.odd().sum() == 3  # 1, 3, 5 are odd

    def test_even(self):
        """Test even check."""
        arr = parrot.array([1, 2, 3, 4, 5])
        assert arr.even().sum() == 2  # 2, 4 are even


class TestArbitraryTransformation:
    """Test arbitrary transformations like in thrust examples."""

    def test_arbitrary_transformation(self):
        """Test B * C + A pattern."""
        a = parrot.array([3, 4, 0, 8, 2])
        b = parrot.array([6, 7, 2, 1, 8])
        c = parrot.array([2, 5, 7, 4, 3])
        result = b * c + a
        expected = parrot.array([15, 39, 14, 12, 26])
        assert result.match(expected)

    def test_keep_sum(self):
        """Test permutation-like gather with keep."""
        source = parrot.array([10, 20, 30, 40, 50, 60])
        mask = parrot.array([1, 1, 0, 1, 0, 1])
        result = source.keep(mask).sum()
        assert result == 130  # 10 + 20 + 40 + 60


class TestToHost:
    """Tests for to_host() function."""

    def test_to_host_basic(self):
        """Test basic to_host."""
        arr = parrot.array([1, 2, 3, 4, 5])
        host = arr.to_host()
        assert host == [1, 2, 3, 4, 5]

    def test_to_host_after_ops(self):
        """Test to_host after operations."""
        arr = parrot.array([1, 2, 3, 4])
        result = arr.times(2).to_host()
        assert result == [2, 4, 6, 8]


class TestReduceAxis:
    """Tests for reduction along an axis (row-wise reduction)."""

    def test_reduce_axis_sum(self):
        """Test sum reduction along axis 1."""
        # Create a 3x4 matrix: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        arr = parrot.array(list(range(12))).reshape((3, 4))

        # Row-wise sum (axis=1 or 2)
        # Rows should be: 6, 22, 38

        # Test axis=1
        res1 = arr.sum(axis=1)
        assert res1.shape == (3,)
        assert res1.to_host() == [6, 22, 38]

        # Test axis=2 (compatibility)
        res2 = arr.sum(axis=2)
        assert res2.shape == (3,)
        assert res2.to_host() == [6, 22, 38]

    def test_reduce_axis_max(self):
        """Test max reduction along axis 1."""
        # [[0, 1],
        #  [2, 3]]
        arr = parrot.array([0, 1, 2, 3]).reshape((2, 2))

        # Row-wise max
        res = arr.max(axis=1)
        assert res.to_host() == [1, 3]

    def test_reduce_axis_min(self):
        """Test min reduction along axis 1."""
        # [[0, 1],
        #  [2, 3]]
        arr = parrot.array([0, 1, 2, 3]).reshape((2, 2))

        # Row-wise min
        res = arr.min(axis=1)
        assert res.to_host() == [0, 2]

    def test_reduce_axis_prod(self):
        """Test product reduction along axis 1."""
        # [[1, 2],
        #  [3, 4]]
        arr = parrot.array([1, 2, 3, 4]).reshape((2, 2))

        # Row-wise prod
        res = arr.prod(axis=1)
        # [1*2, 3*4] = [2, 12]
        assert res.to_host() == [2, 12]

    def test_reduce_axis_all(self):
        """Test logical all reduction along axis 1."""
        # [[1, 0],
        #  [1, 1]]
        arr = parrot.array([1, 0, 1, 1]).reshape((2, 2))

        # Row-wise all
        res = arr.all(axis=1)
        # [False, True] -> [0, 1] usually (returns ParrotArray of dtype)
        # all() reduces with `and`.
        # row 0: 1 and 0 -> 0 (False)
        # row 1: 1 and 1 -> 1 (True)
        assert res.to_host() == [0, 1]

    def test_reduce_axis_any(self):
        """Test logical any reduction along axis 1."""
        # [[0, 0],
        #  [1, 0]]
        arr = parrot.array([0, 0, 1, 0]).reshape((2, 2))

        # Row-wise any
        res = arr.any(axis=1)
        # row 0: 0 or 0 -> 0
        # row 1: 1 or 0 -> 1
        assert res.to_host() == [0, 1]

    def test_reduce_global_behavior_unchanged(self):
        """Test that default reduction behavior remains unchanged."""
        arr = parrot.array([1, 2, 3, 4])
        assert arr.sum(axis=0) == 10
        assert arr.max(axis=0) == 4

    def test_reduce_axis_empty_cols(self):
        """Test reduction on empty rows."""
        # 3x0 matrix
        arr = parrot.matrix(3, 0, 0)  # 3 rows, 0 cols

        # Sum of empty row is 0 (init value for sum)
        res = arr.sum(axis=1)
        assert res.length == 3
        assert res.to_host() == [0, 0, 0]

        # Prod of empty row is 1
        res = arr.prod(axis=1)
        assert res.to_host() == [1, 1, 1]


class TestAstype:
    """Tests for the astype() method."""

    def test_astype_int_to_float(self):
        """Test converting int32 to float32."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.astype(np.float32)
        assert result.dtype == np.float32
        assert result.to_host() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_astype_with_operations(self):
        """Test astype followed by operations."""
        arr = parrot.array([1, 2, 3, 4])
        result = arr.astype(np.float32).div(2)
        assert result.dtype == np.float32
        assert result.to_host() == [0.5, 1.0, 1.5, 2.0]

    def test_astype_preserves_shape(self):
        """Test that astype preserves shape."""
        arr = parrot.array([1, 2, 3, 4]).reshape((2, 2))
        result = arr.astype(np.float32)
        assert result.shape == (2, 2)


class TestReplicate:
    """Tests for the replicate() method."""

    def test_replicate_basic(self):
        """Test basic replicate functionality."""
        arr = parrot.array([1, 2, 3])
        result = arr.replicate(2)
        expected = [1, 1, 2, 2, 3, 3]
        assert result.to_host() == expected

    def test_replicate_three(self):
        """Test replicate with n=3."""
        arr = parrot.array([10, 20])
        result = arr.replicate(3)
        expected = [10, 10, 10, 20, 20, 20]
        assert result.to_host() == expected

    def test_replicate_one(self):
        """Test replicate with n=1 (identity)."""
        arr = parrot.array([1, 2, 3])
        result = arr.replicate(1)
        assert result.to_host() == [1, 2, 3]

    def test_replicate_length(self):
        """Test that replicate produces correct length."""
        arr = parrot.array([1, 2, 3, 4, 5])
        result = arr.replicate(4)
        assert result.length == 20

    def test_replicate_with_operations(self):
        """Test replicate with preceding operations."""
        arr = parrot.array([1, 2, 3]).times(2)  # [2, 4, 6]
        result = arr.replicate(2)
        expected = [2, 2, 4, 4, 6, 6]
        assert result.to_host() == expected


class TestRepeat:
    """Tests for the repeat() method - scalar repetition."""

    def test_repeat_basic(self):
        """Test basic scalar repeat."""
        result = parrot.scalar(7).repeat(5).to_host()
        assert result == [7, 7, 7, 7, 7]

    def test_repeat_one(self):
        """Test repeat with n=1."""
        result = parrot.scalar(42).repeat(1).to_host()
        assert result == [42]

    def test_repeat_non_scalar_raises(self):
        """Test that repeat on a non-scalar array raises."""
        with pytest.raises(ValueError, match="scalar"):
            parrot.array([1, 2, 3]).repeat(3)

    def test_repeat_invalid_n_raises(self):
        """Test that repeat with n<=0 raises."""
        with pytest.raises(ValueError):
            parrot.scalar(1).repeat(0)

    def test_repeat_with_ops(self):
        """Test repeat result can be used in further operations."""
        result = parrot.scalar(3).repeat(4).times(2).to_host()
        assert result == [6, 6, 6, 6]


class TestCross:
    """Tests for the cross() method - cartesian product."""

    def test_cross_basic(self):
        """Test basic cartesian product with a sum op."""
        a = parrot.array([1, 2])
        b = parrot.array([10, 20, 30])
        result = a.cross(b).map(lambda t: t[0] + t[1]).to_host()
        # [1+10, 1+20, 1+30, 2+10, 2+20, 2+30]
        assert result == [11, 21, 31, 12, 22, 32]

    def test_cross_length(self):
        """Test cartesian product length is len(a)*len(b)."""
        a = parrot.array([1, 2, 3])
        b = parrot.array([4, 5])
        result = a.cross(b)
        assert result.length == 6

    def test_cross_mul(self):
        """Test cartesian product with multiplication."""
        a = parrot.array([1, 2])
        b = parrot.array([3, 4])
        result = a.cross(b).map(lambda t: t[0] * t[1]).to_host()
        # [1*3, 1*4, 2*3, 2*4]
        assert result == [3, 4, 6, 8]

    def test_cross_empty_raises(self):
        """Test that cross with empty array raises."""
        with pytest.raises(ValueError, match="empty"):
            parrot.array([1]).cross(parrot.array([]))


class TestEnumerate:
    """Tests for the enumerate() method."""

    def test_enumerate_basic(self):
        """Test basic enumerate - sum of value and index."""
        result = parrot.array([10, 20, 30]).enumerate().map(lambda t: t[0] + t[1]).to_host()
        # [(10,0), (20,1), (30,2)] -> [10, 21, 32]
        assert result == [10, 21, 32]

    def test_enumerate_index_only(self):
        """Test extracting just the index from enumerate."""
        result = parrot.array([5, 5, 5]).enumerate().map(lambda t: t[1]).to_host()
        assert result == [0, 1, 2]

    def test_enumerate_length(self):
        """Test enumerate preserves length."""
        arr = parrot.range(7)
        assert arr.enumerate().length == 7


class TestTranspose:
    """Tests for the transpose() method."""

    def test_transpose_basic(self):
        """Test basic 2x3 transpose."""
        # Matrix: [[0,1,2],[3,4,5]] (2 rows, 3 cols)
        m = parrot.range(6).reshape((2, 3))
        t = m.transpose()
        assert t.shape == (3, 2)
        # Transposed: [[0,3],[1,4],[2,5]]
        assert t.to_host() == [[0, 3], [1, 4], [2, 5]]

    def test_transpose_square(self):
        """Test transpose of a square matrix."""
        # [[1,2],[3,4]]
        m = parrot.array([1, 2, 3, 4]).reshape((2, 2))
        t = m.transpose()
        assert t.shape == (2, 2)
        # [[1,3],[2,4]]
        assert t.to_host() == [[1, 3], [2, 4]]

    def test_transpose_1d_raises(self):
        """Test that transpose on a 1D array raises."""
        with pytest.raises(ValueError, match="rank 2"):
            parrot.range(5).transpose()

    def test_double_transpose(self):
        """Test that transposing twice returns the original."""
        m = parrot.range(12).reshape((3, 4))
        original = m.to_host()
        result = m.transpose().transpose().to_host()
        assert result == original


class TestNrowsNcols:
    """Tests for nrows() and ncols() methods."""

    def test_nrows_ncols_basic(self):
        """Test basic nrows and ncols."""
        arr = parrot.array(list(range(12))).reshape((3, 4))
        assert arr.nrows() == 3
        assert arr.ncols() == 4

    def test_nrows_ncols_square(self):
        """Test nrows and ncols on square matrix."""
        arr = parrot.array(list(range(9))).reshape((3, 3))
        assert arr.nrows() == 3
        assert arr.ncols() == 3

    def test_nrows_ncols_1d_raises(self):
        """Test that nrows/ncols raise on 1D array."""
        arr = parrot.array([1, 2, 3])
        with pytest.raises(ValueError):
            arr.nrows()
        with pytest.raises(ValueError):
            arr.ncols()


class TestOuter:
    """Tests for the outer() method - lazy outer product computation."""

    def test_outer_basic_mul(self):
        """Test basic outer product with multiplication."""
        a = parrot.array([1, 2, 3])
        b = parrot.array([10, 20])
        result = a.outer(b, parrot.mul_op)

        # Shape should be (3, 2)
        assert result.shape == (3, 2)
        assert result.length == 6

        # Expected: [[1*10, 1*20], [2*10, 2*20], [3*10, 3*20]]
        #         = [[10, 20], [20, 40], [30, 60]]
        expected = parrot.array([10, 20, 20, 40, 30, 60]).reshape((3, 2))
        assert result.match(expected)

    def test_outer_with_addition(self):
        """Test outer product with addition operation."""
        a = parrot.array([1, 2, 3])
        b = parrot.array([100, 200])
        result = a.outer(b, parrot.add_op)

        # Expected: [[1+100, 1+200], [2+100, 2+200], [3+100, 3+200]]
        #         = [[101, 201], [102, 202], [103, 203]]
        expected = parrot.array([101, 201, 102, 202, 103, 203]).reshape((3, 2))
        assert result.match(expected)

    def test_outer_shape(self):
        """Test outer product produces correct shape."""
        a = parrot.array([1, 2])
        b = parrot.array([3, 4, 5])
        result = a.outer(b, parrot.mul_op)

        # Shape should be (2, 3)
        assert result.shape == (2, 3)
        assert result.length == 6

        # Expected: [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]]
        #         = [[3, 4, 5], [6, 8, 10]]
        expected = np.array([[3, 4, 5], [6, 8, 10]])
        np.testing.assert_array_equal(result.collect().get(), expected)

    def test_outer_larger_arrays(self):
        """Test outer product with larger arrays."""
        a = parrot.range(10)  # [0, 1, 2, ..., 9]
        b = parrot.range(5)  # [0, 1, 2, 3, 4]
        result = a.outer(b, parrot.mul_op)

        assert result.shape == (10, 5)
        assert result.length == 50

        # Spot check some values
        collected = result.collect().get()
        assert collected[0, 0] == 0 * 0  # 0
        assert collected[5, 3] == 5 * 3  # 15
        assert collected[9, 4] == 9 * 4  # 36

    def test_outer_with_transforms(self):
        """Test outer product with preceding transforms."""
        a = parrot.array([1, 2, 3]).times(2)  # [2, 4, 6]
        b = parrot.array([10, 20]).add(5)  # [15, 25]
        result = a.outer(b, parrot.mul_op)

        # Expected: [[2*15, 2*25], [4*15, 4*25], [6*15, 6*25]]
        #         = [[30, 50], [60, 100], [90, 150]]
        expected = np.array([[30, 50], [60, 100], [90, 150]])
        np.testing.assert_array_equal(result.collect().get(), expected)

    def test_outer_sum(self):
        """Test that we can sum the outer product."""
        a = parrot.array([1, 2])
        b = parrot.array([3, 4, 5])
        result = a.outer(b, parrot.mul_op).sum()

        # All elements: 1*3 + 1*4 + 1*5 + 2*3 + 2*4 + 2*5 = 3+4+5+6+8+10 = 36
        assert result == 36

    def test_outer_with_subtraction(self):
        """Test outer product with subtraction operation."""
        a = parrot.array([10, 20])
        b = parrot.array([1, 2, 3])
        result = a.outer(b, parrot.sub_op)

        # Expected: [[10-1, 10-2, 10-3], [20-1, 20-2, 20-3]]
        #         = [[9, 8, 7], [19, 18, 17]]
        expected = np.array([[9, 8, 7], [19, 18, 17]])
        np.testing.assert_array_equal(result.collect().get(), expected)

    def test_outer_empty_raises(self):
        """Test that outer with empty arrays raises ValueError."""
        a = parrot.array([1, 2, 3])
        b = parrot.array([]).take(0)  # Empty array

        with pytest.raises(ValueError, match="arrays must not be empty"):
            a.outer(b, parrot.mul_op)

    def test_outer_single_elements(self):
        """Test outer product of single element arrays."""
        a = parrot.array([5])
        b = parrot.array([7])
        result = a.outer(b, parrot.mul_op)

        assert result.shape == (1, 1)
        assert result.collect().get()[0, 0] == 35

    def test_outer_custom_binary_lambda(self):
        """Test outer product with custom binary lambda."""
        a = parrot.array([1, 2, 3])
        b = parrot.array([10, 20])

        # Custom lambda: x + y * 2
        result = a.outer(b, lambda x, y: x + y * 2)

        # Expected: [[1+20, 1+40], [2+20, 2+40], [3+20, 3+40]]
        #         = [[21, 41], [22, 42], [23, 43]]
        expected = np.array([[21, 41], [22, 42], [23, 43]])
        np.testing.assert_array_equal(result.collect().get(), expected)

    def test_outer_custom_lambda_complex(self):
        """Test outer product with more complex custom lambda."""
        a = parrot.array([1, 2])
        b = parrot.array([3, 4])

        # Custom lambda: x * x + y
        result = a.outer(b, lambda x, y: x * x + y)

        # Expected: [[1+3, 1+4], [4+3, 4+4]] = [[4, 5], [7, 8]]
        expected = np.array([[4, 5], [7, 8]])
        np.testing.assert_array_equal(result.collect().get(), expected)


class TestSoftmax:
    """Tests for softmax computation using replicate and astype."""

    def test_softmax_simple(self):
        """Test softmax on a simple 2x3 matrix."""
        # Create a 2x3 matrix
        matrix = parrot.range(6, dtype=np.float32).reshape((2, 3))
        # [[0, 1, 2],
        #  [3, 4, 5]]

        cols = matrix.ncols()

        # Numerically stable softmax: subtract max per row
        z = matrix - matrix.max(axis=2).replicate(cols)
        num = z.exp()
        den = num.sum(axis=2)
        result = num / den.replicate(cols)

        # Verify: each row should sum to 1.0
        row_sums = result.sum(axis=2)
        for s in row_sums.to_host():
            assert abs(s - 1.0) < 1e-5

    def test_softmax_100x100(self):
        """Test softmax on a 100x100 matrix (like the C++ example)."""
        # Create a 100x100 matrix
        matrix = parrot.range(10000, dtype=np.float32).reshape((100, 100))

        cols = matrix.ncols()

        # Calculate the row-wise softmax
        z = matrix - matrix.max(axis=2).replicate(cols)
        num = z.exp()
        den = num.sum(axis=2)
        result = num / den.replicate(cols)

        # Verify: each row should sum to 1.0
        row_sums = result.sum(axis=2)
        for s in row_sums.to_host():
            assert abs(s - 1.0) < 1e-4

    def test_replicate_for_broadcasting(self):
        """Test that replicate correctly broadcasts row-wise results."""
        # 3 rows, 4 cols: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        arr = parrot.array(list(range(12))).reshape((3, 4))

        # Row sums: [6, 22, 38]
        row_sums = arr.sum(axis=2)
        assert row_sums.to_host() == [6, 22, 38]

        # Replicate to match original size
        replicated = row_sums.replicate(4)
        # Should be: [6, 6, 6, 6, 22, 22, 22, 22, 38, 38, 38, 38]
        expected = [6, 6, 6, 6, 22, 22, 22, 22, 38, 38, 38, 38]
        assert replicated.to_host() == expected
