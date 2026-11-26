# Parrot Python - Fluent CUDA API

A fluent, array-based API for `cuda.compute` that allows you to write expressive GPU computations. Inspired by the C++ Parrot library.

```python
import parrot

# Basic operations
result = parrot.range(10).times(2).add(1).sum()  # (0-9) * 2 + 1, then sum
print(result)  # 100

# Create arrays from data
arr = parrot.array([2, 2, 2, 1, 1, 2, 2])
print(arr.differ().where().collect().get())  # [3, 5]

# Complex algorithmic chains (Sushi For Two problem)
# fmt: off
sushi = parrot.array([2, 2, 2, 1, 1, 2, 2])
result = (sushi.differ()
               .where()
               .prepend(0)
               .append(sushi.size())
               .deltas()
               .map_adj(parrot.min_op)
               .double()
               .maxr())
# fmt: on
print(result)  # 4
```

## Features

- **Fluent API**: Chain operations naturally with method chaining
- **High Performance**: Uses `cuda.compute` under the hood for optimized GPU execution
- **Transform Composition**: Multiple transforms are efficiently composed into single operations
- **Algorithmic Primitives**: Solve complex problems with elegant chains of operations

## Fusion and array representations

An array is a **materialized array**, **lazy array expression**, or **masked
array expression**. A pending mask retains the physical input extent while
its logical values are the positions where the mask is nonzero. Negative and
other nonbinary mask values select once. `size()` and `shape` report logical
selection length; obtaining them can evaluate predicates without storing values.

| Operation | Materialized array | Lazy expression | Masked expression |
| --- | --- | --- | --- |
| `map(f)` | Lazy map | Compose maps | Masked map |
| `map2(s,h)` — raw scalar | Lazy map | Compose maps | Masked map |
| `map2(b,h)` — array | Lazy zip-map | Compose zip-maps | Compact → zip-map |
| `rev()` | Indexed view | Compose indices | Reverse data and mask |
| `gather(j)` | Indexed view | Compose indices | Compact → view |
| `filter(p)` | Masked expression | Masked expression | Combine predicate masks |
| `keep(m)` | Masked expression | Masked expression | Compact → mask |
| `reduce(op,init)` | Read stored values | Evaluate input | Compact → reduce |
| `scan(op,init)` | Read stored values | Evaluate input | Compact → scan |
| `apply()` | Raises ValueError | Raises ValueError | Materialized array |

F1 composes unary maps, F2 broadcasts a genuine scalar, F3 composes zip-maps,
F4 composes valid index mappings, and F5 passes mapped iterators to consumers.
F6 introduces a pending predicate mask. F7 preserves a mask through unary maps,
raw scalar maps, reversal, and repeated filtering. These constructions do not
count masks, allocate selected values, or launch selection. Compaction and
`sum()` check selection before reading a mapped value. Repeated predicates
short-circuit positions rejected by the preceding mask. A predicate can still
read its own upstream producer when deciding whether to keep a position.

Array `map2` compacts either masked operand before aligning their logical
positions. Gather likewise interprets a masked index operand's selected values.
A second `keep` aligns its mask with the receiver's selected sequence. Generic
reduce and scan retain compaction boundaries; global `sum()` instead uses a
guarded iterator without a compacted input buffer. Scans store their output.

Maps and unary helpers return independent expressions, so branches can reuse a
source safely. Assign the return value when using a standalone operation
(for example, `a = a.map(f)`). Iterator children retain the source allocations they reference.
No implicit cache is created: consuming an expression again can reevaluate it.
Callbacks must be pure, and gather indices must be valid zero-based indices.
Custom binary callbacks need inspectable Python source; lambdas, named functions,
and captured scalar constants are supported.

`scalar(s)` has shape `()` and broadcasts in either array-operand position.
A length-one vector has shape `(1,)` and does not broadcast to another length.
Scalar extension supports empty vectors and preserves a matrix operand's shape.
Python retains its zero-based `range` and `enumerate`, one-based `where`,
`reduce(op, init)` argument order, and
NumPy output from generic `scan`; `sums` and similar helpers remain chainable.
The array dtype controls stored output, including after an annotated callback;
intermediate iterator types are preserved during composition. Array-array maps
use the common NumPy operand dtype, while raw scalar maps retain the receiver's
output dtype. Boolean `sum` counts into an integer. These are Python contracts,
not a promise of identical C++ promotions or property-specialization dispatch.

Run both the existing and fusion GPU tests with `python -m pytest python -q`.
The Python GPU job in Unit Tests runs for PRs targeting `experimental-parrot-python`.
The guarded iterator adapter requires cuda-cccl 1.2.1 or later.

## Installation

```bash
pip install "numpy>=1.21.0,<2.5" cupy "cuda-cccl[cu13]>=1.2.1"  # or [cu12] for CUDA 12
```

NumPy is temporarily limited to versions below 2.5 because `numba-cuda-mlir` 0.5.4 initializes a unitless `NaT`, which NumPy 2.5 deprecates. The bound can be lifted once the CUDA compiler dependency fixes that initialization.

## API Reference

### Array Creation

| Function                                  | Description                      |
| ----------------------------------------- | -------------------------------- |
| `parrot.array(data, dtype=np.int32)`      | Create from list/array           |
| `parrot.range(n, dtype=np.int32)`         | Create sequence [0, 1, ..., n-1] |
| `parrot.constant(value, length, dtype)`   | Create constant sequence         |
| `parrot.scalar(value, dtype=np.int32)`    | Create rank-zero scalar      |
| `parrot.matrix(rows, cols, value, dtype)` | Create matrix (flattened)        |

### Unary Transforms (Lazy)

| Method      | Description           |
| ----------- | --------------------- |
| `.map(op)`  | Apply custom function |
| `.abs()`    | Absolute value        |
| `.double()` | Multiply by 2         |
| `.half()`   | Divide by 2           |
| `.neg()`    | Negate                |
| `.sq()`     | Square                |
| `.sqrt()`   | Square root           |
| `.exp()`    | Exponential (e^x)     |
| `.log()`    | Natural logarithm     |
| `.even()`   | Check if even (1/0)   |
| `.odd()`    | Check if odd (1/0)    |

### Deterministic random values

`.drand()` defaults to seed `42`; `.drand(seed=1234)` selects a reproducible
sequence. Each nonnegative input element bounds its result in `[0, value)`;
zero stays zero. Integer results truncate toward zero. Dtype and shape are
preserved, and generation stays lazy until a consuming operation runs.

```python
bounds = parrot.array([100, 100, 100, 100])
print(bounds.drand().to_host())           # [25, 8, 57, 22]
print(bounds.drand(seed=1234).to_host())  # [71, 20, 74, 34]
```

The 32-bit LCG advances as `state = 1664525 * state + 1013904223` modulo `2**32`;
element `i` uses state `i + 1`. Integer seeds wrap modulo `2**32`. Logarithmic
skip-ahead computes each index independently, so repeat evaluation and GPU
scheduling do not change results. Equal inputs and seeds match C++ `.drand()`.
Filtered inputs are compacted before assigning indices.

### Binary Transforms (Lazy)

| Method           | Description                 |
| ---------------- | --------------------------- |
| `.add(value)`    | Add value                   |
| `.minus(value)`  | Subtract value              |
| `.times(factor)` | Multiply by factor          |
| `.div(value)`    | Divide by value             |
| `.gt(value)`     | Greater than (1/0)          |
| `.gte(value)`    | Greater than or equal (1/0) |
| `.lt(value)`     | Less than (1/0)             |
| `.lte(value)`    | Less than or equal (1/0)    |

### Operator Overloading

```python
a + b      # Element-wise addition (scalar or array)
a - b      # Element-wise subtraction
a * b      # Element-wise multiplication
a / b      # Element-wise division
a == b     # Element-wise equality (returns 1/0)
```

### Adjacent Operations

| Method         | Description                                   |
| -------------- | --------------------------------------------- |
| `.map_adj(op)` | Apply binary op to adjacent pairs             |
| `.differ()`    | Check if adjacent elements differ (1/0)       |
| `.deltas()`    | Compute differences between adjacent elements |

### Reductions (Eager)

| Method               | Description             |
| -------------------- | ----------------------- |
| `.reduce(op, init)`  | Custom reduction        |
| `.sum()`             | Sum all elements        |
| `.prod()`            | Product of all elements |
| `.max()` / `.maxr()` | Find maximum            |
| `.min()` / `.minr()` | Find minimum            |
| `.all()`             | Logical AND             |
| `.any()`             | Logical OR              |

### Scans (Eager, Chainable)

| Method            | Description           |
| ----------------- | --------------------- |
| `.scan(op, init)` | Custom inclusive scan |
| `.sums()`         | Cumulative sum        |
| `.prods()`        | Cumulative product    |
| `.maxs()`         | Cumulative maximum    |
| `.mins()`         | Cumulative minimum    |
| `.alls()`         | Cumulative AND        |
| `.anys()`         | Cumulative OR         |

### Array Manipulation

| Method            | Description                         |
| ----------------- | ----------------------------------- |
| `.prepend(value)` | Add value at beginning              |
| `.append(value)`  | Add value at end                    |
| `.drop(n)`        | Skip first n elements               |
| `.rev()`          | Reverse array                       |
| `.sort()`         | Sort ascending                      |
| `.uniq()`         | Remove adjacent duplicates          |
| `.distinct()`     | Remove all duplicates (sort + uniq) |

### Filtering & Indexing

| Method                       | Description                                  |
| ---------------------------- | -------------------------------------------- |
| `.filter(predicate)` | Retain a lazy predicate mask |
| `.keep(mask)` | Retain a pending selection mask |
| `.where()`                   | Get 1-indexed positions of non-zero elements |
| `.chunk_by_reduce(pred, op)` | Group consecutive elements and reduce        |

### Element Access

| Method     | Description                              |
| ---------- | ---------------------------------------- |
| `.front()` | Get first element                        |
| `.back()`  | Get last element                         |
| `.value()` | Extract scalar from single-element array |
| `.size()`  | Get array length                         |

### Array Comparison

| Method             | Description                             |
| ------------------ | --------------------------------------- |
| `.min_with(other)` | Element-wise minimum with another array |
| `.max_with(other)` | Element-wise maximum with another array |
| `.match(other)`    | Check if arrays are equal               |

### Materialization

| Method       | Description               |
| ------------ | ------------------------- |
| `.collect()` | Materialize to CuPy array |

### Binary Operators (for map_adj, chunk_by_reduce)

| Operator        | Description                |
| --------------- | -------------------------- |
| `parrot.min_op` | Minimum of two values      |
| `parrot.max_op` | Maximum of two values      |
| `parrot.lt_op`  | Less than (returns 1/0)    |
| `parrot.gt_op`  | Greater than (returns 1/0) |
| `parrot.eq_op`  | Equal (returns 1/0)        |
| `parrot.neq_op` | Not equal (returns 1/0)    |
| `parrot.add_op` | Addition                   |
| `parrot.sub_op` | Subtraction                |
| `parrot.mul_op` | Multiplication             |

## Example Problems

### Rain Water Trapping

```python
arr = parrot.array([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
result = (arr.maxs().min_with(arr.rev().maxs().rev()) - arr).sum()
# result = 6
```

### Maximum Consecutive Ones

```python
nums = parrot.array([1, 1, 0, 1, 1, 1])
result = nums.chunk_by_reduce(parrot.eq_op, parrot.add_op).maxr()
# result = 3
```

### Maximum Gap

```python
nums = parrot.array([3, 6, 9, 1])
result = nums.append(nums.back()).sort().deltas().maxr()
# result = 3
```

### Ocean View

```python
nums = parrot.array([4, 2, 3, 1])
result = nums.rev().maxs().differ().prepend(1).rev().where()
# result matches [1, 3, 4]
```

## Requirements

- Python 3.10+
- NumPy
- CuPy
- CUDA toolkit with `cuda.compute` support (cuda-cccl)
- NVIDIA GPU with CUDA support
