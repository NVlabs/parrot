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

## Installation

```bash
pip install numpy cupy "cuda-cccl[cu13]"  # or [cu12] for CUDA 12
```

## API Reference

### Array Creation

| Function                                  | Description                      |
| ----------------------------------------- | -------------------------------- |
| `parrot.array(data, dtype=np.int32)`      | Create from list/array           |
| `parrot.range(n, dtype=np.int32)`         | Create sequence [0, 1, ..., n-1] |
| `parrot.constant(value, length, dtype)`   | Create constant sequence         |
| `parrot.scalar(value, dtype=np.int32)`    | Create single-element array      |
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

- Python 3.8+
- NumPy
- CuPy
- CUDA toolkit with `cuda.compute` support (cuda-cccl)
- NVIDIA GPU with CUDA support
