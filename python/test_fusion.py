# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""GPU contracts for F1–F7 and the API behavior-by-representation matrix."""

from unittest.mock import patch

import cupy as cp
import numpy as np
import pytest
from cuda.compute import iterators, types
from cuda.compute._cpp_compile import compile_cpp_op_code
from cuda.compute.op import RawOp

import parrot


@pytest.fixture(params=["materialized", "lazy", "masked"])
def represented(request):
    if request.param == "materialized":
        return parrot.array([2, 4, 6])
    if request.param == "lazy":
        return parrot.range(3).add(1).times(2)
    return parrot.array([99, 2, 0, 4, 6]).keep([0, 1, 0, 2, -1])


@pytest.mark.parametrize(
    "operation",
    [
        "map",
        "scalar_map",
        "array_map",
        "rev",
        "gather",
        "filter",
        "keep",
        "reduce",
        "scan",
        "apply",
    ],
)
def test_representation_matrix(represented, operation):
    source = represented
    if operation == "map":
        assert source.map(lambda x: x + 1).to_host() == [3, 5, 7]
    elif operation == "scalar_map":
        assert source.map2(1, parrot.add_op).to_host() == [3, 5, 7]
    elif operation == "array_map":
        assert source.map2(parrot.array([1, 2, 3]), parrot.add_op).to_host() == [
            3,
            6,
            9,
        ]
    elif operation == "rev":
        assert source.rev().to_host() == [6, 4, 2]
    elif operation == "gather":
        assert source.gather(parrot.array([2, 0])).to_host() == [6, 2]
    elif operation == "filter":
        result = source.filter(lambda x: x > 2)
        assert result.has_mask
        assert result.to_host() == [4, 6]
    elif operation == "keep":
        assert source.keep([0, -1, 2]).to_host() == [4, 6]
    elif operation == "reduce":
        assert source.reduce(parrot.add_op, 0) == 12
    elif operation == "scan":
        np.testing.assert_array_equal(source.scan(parrot.add_op, 0), [2, 6, 12])
    elif source.has_mask:
        result = source.apply()
        assert not result.has_mask
        assert isinstance(result._data, cp.ndarray)
        assert result.to_host() == [2, 4, 6]
    else:
        with pytest.raises(ValueError, match="masked"):
            source.apply()
    assert source.to_host() == [2, 4, 6]


def test_f1_independent_maps_and_filter_snapshots():
    source = parrot.array([-2, 0, 2, 4])
    squared = source.sq()
    negated = source.neg()
    selected = source.filter(lambda x: x > 0)
    shifted = selected.map(lambda x: x + 1).map(lambda x: x * 2)
    assert squared.to_host() == [4, 0, 4, 16]
    assert negated.to_host() == [2, 0, -2, -4]
    assert source.to_host() == [-2, 0, 2, 4]
    assert selected.to_host() == [2, 4]
    assert shifted.to_host() == [6, 10]


@pytest.mark.parametrize("length", [0, 1, 4])
def test_f2_rank_zero_extension_in_both_directions(length):
    values = parrot.range(length)
    scalar = parrot.scalar(3).map(lambda x: x + 2)
    assert scalar.shape == ()
    assert scalar.rank() == 0
    assert scalar.add(values).to_host() == [5 + i for i in range(length)]
    assert values.minus(scalar).to_host() == [i - 5 for i in range(length)]
    assert scalar.add(values).shape == (length,)
    assert scalar.value() == 5
    assert scalar.add(parrot.scalar(2)).shape == ()
    with pytest.raises(ValueError, match="Shape mismatch"):
        parrot.array([5]).add(parrot.range(3))


def test_f2_shape_types_and_custom_binary_functions():
    matrix = parrot.range(4).reshape((2, 2))
    result = parrot.scalar(0.5, dtype=np.float64).add(matrix)
    assert result.shape == (2, 2)
    assert result.dtype == np.float64
    assert result.to_host() == [[0.5, 1.5], [2.5, 3.5]]
    assert matrix.map2(2, lambda x, y: x * x + y).to_host() == [[2, 3], [6, 11]]
    assert parrot.range(3).map2(parrot.range(3), lambda x, y: x + 2 * y).to_host() == [
        0,
        3,
        6,
    ]


def test_f3_zip_maps_preserve_intermediate_precision_and_logical_alignment():
    op = lambda x: x + 1  # noqa: E731
    op.__annotations__ = {"x": np.int64, "return": np.float32}
    left = parrot.array([16_777_218], dtype=np.int64).map(op)
    assert left.map2(parrot.array([1], dtype=np.int64), parrot.add_op).to_host() == [
        16_777_221
    ]
    a = parrot.array([1, 99, 3, 99]).keep([1, 0, 2, 0])
    b = parrot.array([99, 10, 20]).keep([0, -1, 1])
    assert a.add(b).to_host() == [11, 23]
    with pytest.raises(ValueError, match="Shape mismatch"):
        a.add(parrot.array([1]))


def test_f4_views_and_products_do_not_collect_lazy_inputs():
    source = parrot.range(4).add(1).sq()
    indices = parrot.array([3, 1])
    scalar = parrot.scalar(7)
    other = parrot.range(2).add(1)
    with patch.object(
        parrot.ParrotArray, "collect", side_effect=AssertionError("eager collection")
    ):
        with patch.object(cp, "empty", side_effect=AssertionError("eager allocation")):
            outputs = [
                source.rev().rev(),
                source.drop(1).take(2),
                source.gather(indices),
                source.replicate(2),
                source.cycle(6),
                scalar.repeat(3),
                source.map_adj(parrot.add_op),
                other.outer(other, parrot.mul_op),
                other.cross(other).map(lambda t: t[0] + t[1]),
            ]
    expected = [
        [1, 4, 9, 16],
        [4, 9],
        [16, 4],
        [1, 1, 4, 4, 9, 9, 16, 16],
        [1, 4, 9, 16, 1, 4],
        [7, 7, 7],
        [5, 13, 25],
        [[1, 2], [2, 4]],
        [2, 3, 3, 4],
    ]
    for result, values in zip(outputs, expected, strict=True):
        assert result.to_host() == values
    assert source.gather(indices).gather(parrot.array([1, 0])).to_host() == [4, 16]


def test_f4_masked_indices_and_metadata_boundaries():
    indices = parrot.array([99, 2, 99, 0]).keep([0, 1, 0, -1])
    source = parrot.range(4).add(10)
    assert source.gather(indices).to_host() == [12, 10]
    masked = source.keep([0, 1, 0, 1])
    assert masked.gather(parrot.array([1, 0])).to_host() == [13, 11]
    assert masked.reshape((1, 2)).to_host() == [[11, 13]]
    assert masked.flatten().to_host() == [11, 13]
    assert masked.replicate(2).to_host() == [11, 11, 13, 13]
    assert masked.outer(parrot.scalar(2), parrot.mul_op).to_host() == [[22], [26]]


@pytest.mark.parametrize("representation", ["materialized", "lazy", "masked"])
def test_f5_empty_consumers(representation):
    if representation == "materialized":
        source = parrot.array([])
    elif representation == "lazy":
        source = parrot.range(0).map(lambda x: x + 1)
    else:
        source = parrot.array([0, 0]).keep([0, 0]).map(lambda x: 100 // x)
    assert source.reduce(parrot.add_op, 7) == 7
    assert source.sum() == 0
    assert source.scan().size == 0
    assert source.sums().to_host() == []
    assert source.filter(lambda x: x > 0).apply().to_host() == []


@pytest.mark.parametrize("mask", [[0, 2, 0, -1], [0, 0, 0, 0], [1, 1, 1, 1]])
def test_f6_f7_construction_is_lazy(mask):
    source = parrot.array([0, 2, 0, 4]).keep(mask)
    with patch.object(cp, "empty", side_effect=AssertionError("device allocation")):
        with patch.object(
            parrot.ParrotArray, "collect", side_effect=AssertionError("collection")
        ):
            with patch.object(
                parrot.algorithms, "select", side_effect=AssertionError("selection")
            ):
                result = (
                    source.map(lambda x: x + 1)
                    .map2(2, parrot.mul_op)
                    .rev()
                    .filter(lambda x: x > 3)
                )
    assert result.has_mask
    expected = [(x + 1) * 2 for x, keep in zip([0, 2, 0, 4], mask, strict=True) if keep]
    assert result.to_host() == [x for x in reversed(expected) if x > 3]


def observed_values():
    """A GPU observation, independent of the library's conditional iterator."""
    counters = cp.zeros(2, dtype=np.int32)
    code = compile_cpp_op_code("""
        extern "C" __device__ void parrot_observe_value(void* state, void* input, void* output) {
            int* counters = *static_cast<int**>(state);
            int value = *static_cast<int*>(input);
            atomicAdd(counters, 1);
            if (value == 0) atomicAdd(counters + 1, 1);
            *static_cast<int*>(output) = value == 0 ? -1 : 100 / value;
        }
    """)
    op = RawOp(
        name="parrot_observe_value",
        ltoir=code,
        state=np.uint64(counters.data.ptr).tobytes(),
        state_alignment=8,
    )
    source = parrot.array([0, 2, 0, 4])
    iterator = iterators.TransformIterator(
        source._get_composed_iterator(), op, value_type=types.from_numpy_dtype(np.int32)
    )
    return parrot.ParrotArray(iterator=iterator, dtype=np.int32, length=4), counters


@pytest.mark.parametrize("consumer", ["collect", "sum", "reduce", "scan", "filter"])
def test_f7_rejected_values_are_never_evaluated_on_gpu(consumer):
    source, counters = observed_values()
    selected = source.keep([0, 2, 0, -1])
    assert selected.size() == 2
    assert selected.shape == (2,)
    assert counters.get().tolist() == [0, 0]
    result = selected.map(lambda x: x + 1).map(lambda x: x * 2).rev()
    if consumer == "collect":
        assert result.to_host() == [52, 102]
    elif consumer == "sum":
        with patch.object(
            parrot.ParrotArray,
            "_apply_mask_if_needed",
            side_effect=AssertionError("sum compacted"),
        ):
            assert result.sum() == 154
    elif consumer == "reduce":
        assert result.reduce(parrot.add_op, 0) == 154
    elif consumer == "scan":
        np.testing.assert_array_equal(result.scan(), [52, 154])
    else:
        assert result.filter(lambda x: x > 60).filter(lambda x: x < 200).to_host() == [
            102
        ]
    observations = counters.get().tolist()
    assert observations[0] >= 2
    assert observations[1] == 0
    counters.fill(0)
    assert source.keep([0, 0, 0, 0]).filter(lambda x: x > 0).sum() == 0
    assert counters.get().tolist() == [0, 0]


def test_mask_counts_boolean_sum_and_nonfinite_selection():
    values = parrot.array([float("nan"), 2, float("inf"), 4], dtype=np.float64)
    selected = values.keep([0, 2, 0, -1])
    assert selected.size() == 2
    assert selected.sum() == 6
    assert selected.apply().sum() == 6
    assert (
        parrot.array([1] * 300, dtype=np.uint8)
        .keep(parrot.array([1] * 300, dtype=np.uint8))
        .size()
        == 300
    )
    assert parrot.array([True, False, True], dtype=bool).sum() == 2
    assert parrot.array([True, False, True], dtype=bool).keep([1, 1, 1]).sum() == 2
    assert selected.keep(parrot.array([0, 1, 0]).keep([0, 1, 1])).to_host() == [2]
    with pytest.raises(ValueError, match="Shape mismatch"):
        selected.keep([1, 0, 1, 0])


def test_masked_logical_boundaries_after_scalar_mapping():
    values = parrot.array([99, 2, 99, 4]).keep([0, 1, 0, 1]).add(1)
    assert values.prepend(1).to_host() == [1, 3, 5]
    assert values.append(7).to_host() == [3, 5, 7]
    assert values.prepend(1).size() == 3
    assert values.append(7).size() == 3
    assert values.match(parrot.array([3, 5]))
    assert values.pairs(parrot.array([10, 20])).map(
        lambda t: t[0] + t[1]
    ).to_host() == [13, 25]
    assert values.enumerate().map(lambda t: t[0] + t[1]).to_host() == [3, 6]
    assert values.min_with(parrot.array([4, 4])).to_host() == [3, 4]
    assert values.max_with(parrot.array([4, 4])).to_host() == [4, 5]
    assert (
        values.keep([0, 0]).chunk_by_reduce(parrot.eq_op, parrot.add_op).to_host() == []
    )


def test_named_binary_callbacks_closures_and_distinct_bindings():
    scale = 3

    def combine(x, y):
        value = x * scale
        return value + y

    values = parrot.range(3)
    assert values.map2(1, combine).to_host() == [1, 4, 7]
    assert values.map2(2, combine).to_host() == [2, 5, 8]
    assert values.map2(1, combine).map2(2, combine).to_host() == [5, 14, 23]
    assert values.map2(values, combine).to_host() == [0, 4, 8]
