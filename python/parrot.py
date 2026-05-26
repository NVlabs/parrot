"""Vendored from https://github.com/NVlabs/parrot/blob/experimental-parrot-python/python/parrot.py."""

import ast
import builtins
import functools
import inspect
import operator
import random
import textwrap
import time
import types
from typing import Any, Callable, Optional, Tuple, Union

import cupy as cp
import numpy as np

try:
    import cuda.compute.algorithms as algorithms
    import cuda.compute.iterators as iterators
    import cuda.compute.types as cccl_types
except ImportError as e:
    raise ImportError(
        "cuda.compute is required but not available.\n"
        "Try one of these installation methods:\n"
        "1. Install from PyPI: pip install cuda-cccl[cu13] (or [cu12])\n"
        "2. Install CUDA toolkit >= 12.0 from NVIDIA\n"
        f"Original error: {e}"
    ) from e


# Mapping from binary ops to their tuple versions will be populated after ops are defined
_OP_TO_ZIP = {}


def from_cupy(arr):
    """Wrap a CuPy array as a ParrotArray for fusion."""
    # Convert numpy dtype object to numpy type for compatibility
    dtype = np.dtype(arr.dtype).type
    pa = ParrotArray(data=arr, iterator=arr, dtype=dtype, length=len(arr))
    return pa


# Global counter for entropy mixing (similar to C++ rnd::global_counter)
_global_rand_counter = 0


def _binary_to_tuple_op(func):
    """Convert a binary function (x, y) -> result to tuple form (t) -> result.

    Uses AST transformation to create a new function without closures,
    which is necessary for CUDA compilation with Numba.

    Args:
        func: A binary function or lambda like `lambda x, y: x + y`

    Returns:
        A tuple-based function like `lambda t: t[0] + t[1]`

    Raises:
        ValueError: If the function cannot be converted
    """
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source).strip()

        # Parse the source and find the lambda node
        # The source might be just a lambda or a larger expression containing one
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError:
            tree = ast.parse(source, mode="exec")

        # Walk the tree to find the lambda node
        lambda_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                lambda_node = node
                break

        if lambda_node is None:
            raise ValueError("Could not find lambda in source")

        args = lambda_node.args.args
        if len(args) != 2:
            raise ValueError(f"Expected binary function with 2 args, got {len(args)}")

        # Get parameter names
        param1 = args[0].arg
        param2 = args[1].arg

        # Create transformer to replace param references with tuple indexing
        class ParamReplacer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == param1:
                    return ast.Subscript(
                        value=ast.Name(id="t", ctx=ast.Load()),
                        slice=ast.Constant(value=0),
                        ctx=node.ctx,
                    )
                elif node.id == param2:
                    return ast.Subscript(
                        value=ast.Name(id="t", ctx=ast.Load()),
                        slice=ast.Constant(value=1),
                        ctx=node.ctx,
                    )
                return node

        # Transform the body
        new_body = ParamReplacer().visit(lambda_node.body)

        # Create new lambda: lambda t: <transformed_body>
        new_lambda = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="t")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=new_body,
        )

        # Wrap in Expression for eval
        new_tree = ast.Expression(body=new_lambda)
        ast.fix_missing_locations(new_tree)

        # Compile and execute to get the new function
        code = compile(new_tree, "<generated>", "eval")
        return eval(code)  # noqa: S307

    except Exception as e:
        raise ValueError(
            f"Could not convert binary op to tuple form: {e}. "
            "Try using a tuple-based lambda instead: lambda t: t[0] * t[1]"
        ) from e


def _make_rand_op(extra_entropy: int):
    """Create a random functor for integer types.

    This mirrors the C++ rand_functor - using a hash-based approach for deterministic
    per-element random values that can be computed lazily.

    Args:
        extra_entropy: A seed value mixed with the index for randomness

    Returns:
        A function that takes (idx, val) tuple and returns random int in [0, val)
    """
    # Capture entropy as a constant for CUDA compilation
    _entropy = extra_entropy

    def rand_op(t):
        idx = t[0]
        val = t[1]

        # Improved hash mixing for better distribution (same as C++)
        h1 = (idx ^ _entropy) & 0xFFFFFFFF
        h1 = (((h1 >> 16) ^ h1) * 0x45D9F3B) & 0xFFFFFFFF
        h1 = (((h1 >> 16) ^ h1) * 0x45D9F3B) & 0xFFFFFFFF
        h1 = ((h1 >> 16) ^ h1) & 0xFFFFFFFF

        # Generate random float in [0, 1) using the hash as seed
        rand_val = (h1 & 0x7FFFFF) / float(0x800000)  # 23 bits -> [0, 1)

        # Scale by value and return integer in [0, val)
        return int(rand_val * val)

    return rand_op


class ParrotArray:
    """A fluent API for cuda.compute operations."""

    def __init__(self, data=None, iterator=None, dtype=np.int32, mask=None, length=0):
        self._data = data
        self._iterator = iterator
        self.dtype = dtype
        self._transforms = []
        self.length = length
        # Track base CuPy array for lazy fusion (e.g., deltas().map_adj())
        self._base_cupy = None
        # Track offset info for adjacent-pair operations
        self._adj_offset = 0  # How many adjacent-pair ops have been applied
        self._shape = None
        # Lazy mask support: stores a ParrotArray mask that will be applied lazily
        # When mask is set, operations can either work directly with the mask
        # or call _apply_mask_if_needed() to materialize the filtered result
        self._mask = mask
        # For ZipIterator-backed (tuple-valued) arrays from pairs()/enumerate(),
        # remember the two source arrays so __repr__ can format tuples by
        # collecting each side independently (a ZipIterator cannot be passed
        # to unary_transform, so the normal materialization path fails).
        # b is None means "indices" (enumerate).
        self._pair_parts: Optional[Tuple["ParrotArray", Optional["ParrotArray"]]] = None
        if data is not None:
            # If we implement from_array, we'd set length here
            pass

    @property
    def has_mask(self):
        """Check if this array has a lazy mask that hasn't been applied yet."""
        return self._mask is not None

    @property
    def shape(self):
        """Get the shape of the array."""
        if self._shape is None:
            return (self.length,)
        return self._shape

    def __repr__(self):
        """Show a preview of the array contents."""
        if self._iterator is None:
            return "ParrotArray(empty)"

        # Tuple-valued arrays from pairs()/enumerate(). The underlying
        # ZipIterator cannot be fed to unary_transform (scalar output only),
        # so we materialize each side independently and zip on the host. Only
        # valid when no post-zip transforms have been applied (after .map()
        # the output is scalar again and the normal path works).
        if (
            self._pair_parts is not None
            and not self._transforms
            and self.length <= 1000
        ):
            try:
                return self._repr_pairs()
            except Exception:
                # Fall through to generic strategies on failure
                pass

        # Strategy 1: Full representation for small arrays (<= 1000 elements)
        # This matches numpy's behavior for shape visualization (e.g. 2D matrices)
        if self.length <= 1000:
            try:
                # Materialize and move to host for numpy formatting
                h_arr = self.collect().get()

                prefix = "ParrotArray("
                # Use comma separator to match repr() style
                arr_str = np.array2string(h_arr, separator=", ", prefix=prefix)

                return f"{prefix}{arr_str}, dtype={self.dtype.__name__})"
            except Exception:
                # Fallback to preview strategy on failure
                pass

        # Strategy 2: Preview for large arrays or failures
        # Collect a preview (up to 10 elements)
        preview_len = min(self.length, 10)
        if preview_len == 0:
            return f"ParrotArray([], dtype={self.dtype.__name__}, length=0)"

        # Create a temporary copy to collect preview
        temp = ParrotArray(iterator=self._iterator, dtype=self.dtype)
        temp.length = preview_len
        temp._transforms = list(self._transforms)

        try:
            preview = temp.collect().get()
            # Convert to Python types for cleaner display
            preview_list = [x.item() if hasattr(x, "item") else x for x in preview]
            if self.length <= 10:
                arr_str = str(preview_list)
            else:
                arr_str = str(preview_list)[:-1] + ", ...]"
            return f"ParrotArray({arr_str}, dtype={self.dtype.__name__}, length={self.length})"
        except Exception:
            return f"ParrotArray(length={self.length}, dtype={self.dtype.__name__}, transforms={len(self._transforms)})"

    def _repr_pairs(self):
        """Format a pair-valued (ZipIterator-backed) array as a list of tuples.

        Materializes each side of the zip independently and combines on the
        host.  ``self._pair_parts[1] is None`` indicates ``enumerate()`` —
        the second component is synthesized as ``range(length)``.
        """
        assert self._pair_parts is not None

        def _to_host_list(src):
            host = src.collect()
            host = host.get() if hasattr(host, "get") else np.asarray(host)
            return host.reshape(-1).tolist()

        a_src, b_src = self._pair_parts
        a_list = _to_host_list(a_src)
        if b_src is None:
            b_list = list(builtins.range(self.length))
        else:
            b_list = _to_host_list(b_src)

        pairs_list = list(zip(a_list[: self.length], b_list[: self.length]))
        return f"ParrotArray({pairs_list}, dtype={self.dtype.__name__})"

    def _sanitize_op(self, op):
        """Ensure op has a valid name and annotations for CUDA."""
        # Fast path: not a lambda — skip renaming and expensive inspect calls.
        # The cuda.compute library handles type inference from the iterator
        # when annotations are absent; we only need inspect for lambdas.
        name = getattr(op, "__name__", "")
        if name and name != "<lambda>":
            return op

        # Handle lambda names which cause compilation errors
        if name == "<lambda>":
            # Create a new function with same code but valid name
            # Hash bytecode AND closure values for correct caching
            # (same lambda with different captured values = different hash)
            closure_hash = 0
            if op.__closure__:
                closure_vals = tuple(c.cell_contents for c in op.__closure__)
                closure_hash = hash(closure_vals)
            code_hash = hash((op.__code__.co_code, closure_hash))
            new_name = f"op_{code_hash:x}".replace("-", "m")
            op = types.FunctionType(
                op.__code__,
                op.__globals__,
                name=new_name,
                argdefs=op.__defaults__,
                closure=op.__closure__,
            )

        # Add annotations if missing (helps Numba inference)
        if not getattr(op, "__annotations__", None):
            try:
                sig = inspect.signature(op)
                annotations = {}
                for param in sig.parameters:
                    annotations[param] = self.dtype
                if "return" not in annotations:
                    annotations["return"] = self.dtype
                op.__annotations__ = annotations
            except Exception:
                pass

        return op

    def _sanitize_predicate(self, pred):
        """Prepare predicate for CUDA selection; default return type is uint8."""
        # Fast path: not a lambda — skip renaming and expensive inspect calls.
        name = getattr(pred, "__name__", "")
        if name and name != "<lambda>":
            return pred

        if name == "<lambda>":
            # Hash bytecode AND closure values for correct caching
            closure_hash = 0
            if pred.__closure__:
                closure_vals = tuple(c.cell_contents for c in pred.__closure__)
                closure_hash = hash(closure_vals)
            code_hash = hash((pred.__code__.co_code, closure_hash))
            new_name = f"pred_{code_hash:x}".replace("-", "m")
            pred = types.FunctionType(
                pred.__code__,
                pred.__globals__,
                name=new_name,
                argdefs=pred.__defaults__,
                closure=pred.__closure__,
            )

        if not getattr(pred, "__annotations__", None):
            try:
                sig = inspect.signature(pred)
                annotations = {param: self.dtype for param in sig.parameters}
                annotations["return"] = np.uint8
                pred.__annotations__ = annotations
            except Exception:
                pass
        elif "return" not in pred.__annotations__:
            pred.__annotations__["return"] = np.uint8

        return pred

    def _get_composed_iterator(self):
        """Get the final iterator with all transforms applied.

        Each TransformIterator eagerly infers its output value_type from the
        underlying iterator's value_type, so chaining works naturally.
        """
        if self._iterator is None:
            raise ValueError("No data source. Call range() or constant() first.")

        if not self._transforms:
            return self._iterator

        cache_key = (id(self._iterator), len(self._transforms))
        if hasattr(self, "_composed_cache") and self._composed_cache[0] == cache_key:
            return self._composed_cache[1]

        final_iterator = self._iterator
        for transform in self._transforms:
            final_iterator = iterators.TransformIterator(final_iterator, transform)

        self._composed_cache = (cache_key, final_iterator)
        return final_iterator

    def _get_nestable_iterator(self):
        """Get composed iterator safe for use as a child of ZipIterator/PermutationIterator.

        For composed iterators (those with transforms), _get_composed_iterator
        already sets value_type to the correct OUTPUT type, so no swap is needed.

        For raw iterators (no transforms) whose value_type doesn't match the
        expected output dtype (e.g. TransformIterators produced by prepend/append
        whose value_type is set to their INPUT ZipValue type), we temporarily
        swap to the output dtype so parent iterators build correct type metadata,
        then restore afterward.

        Returns (iterator, restore_func) — caller MUST call restore_func() after
        constructing the parent iterator.
        """
        it = self._get_composed_iterator()
        if not hasattr(it, "value_type"):
            return it, lambda: None
        output_td = cccl_types.from_numpy_dtype(self.dtype)
        if it.value_type != output_td:
            saved = it._value_type
            it._value_type = output_td
            return it, lambda: setattr(it, "_value_type", saved)
        return it, lambda: None

    def range(self, n: int, dtype=np.int32):
        """Create a counting iterator with 'n' elements starting from 0."""
        self._iterator = iterators.CountingIterator(dtype(0))
        self.dtype = dtype
        self.length = n
        return self

    def constant(self, value: Union[int, float], length: int, dtype=np.int32):
        """Create a constant iterator with the given value and length."""
        self._iterator = iterators.ConstantIterator(dtype(value))
        self.dtype = dtype
        self.length = length
        return self

    def drop(self, n: int):
        """Skip the first n elements.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            n: Number of elements to drop from the beginning

        Returns:
            A new ParrotArray without the first n elements

        Raises:
            ValueError: If n < 0 or n > size()
        """
        # Apply mask if present before drop operation
        if self.has_mask:
            return self._apply_mask_if_needed().drop(n)

        if self._iterator is None:
            raise ValueError("No data source.")
        if n < 0:
            raise ValueError("Cannot drop negative elements")
        if n > self.length:
            raise ValueError(
                f"Cannot drop {n} elements from array of size {self.length}"
            )

        if n == 0:
            return self._clone()

        if n == self.length:
            return self._empty_result()

        # Materialize and slice
        d_data = self.collect()
        d_dropped = d_data[n:].copy()

        result = ParrotArray(data=d_dropped, dtype=self.dtype)
        result._iterator = d_dropped
        result.length = self.length - n
        return result

    def _clone(self):
        """Create a shallow copy of the array."""
        new_obj = ParrotArray(
            data=self._data, iterator=self._iterator, dtype=self.dtype, mask=self._mask
        )
        new_obj.length = self.length
        new_obj._transforms = list(self._transforms)
        new_obj._base_cupy = self._base_cupy
        new_obj._adj_offset = self._adj_offset
        new_obj._shape = self._shape
        return new_obj

    def _empty_result(self):
        """Create an empty ParrotArray with this array's dtype."""
        result = ParrotArray(dtype=self.dtype)
        result._iterator = iterators.ConstantIterator(self.dtype(0))
        result.length = 0
        return result

    def _apply_mask_if_needed(self):
        """Apply the lazy mask if present, returning a new ParrotArray with filtered data.

        This is the equivalent of copy_if/select - it materializes the mask by
        actually filtering the data. Operations that need contiguous filtered
        data should call this first.

        Returns:
            If no mask: returns self unchanged
            If has mask: returns a new ParrotArray with the mask applied (filtered data)
        """
        if not self.has_mask:
            return self

        # Perform the actual selection (copy_if equivalent)
        mask = self._mask

        # Materialize mask to a raw array so the select kernel sees correct types
        d_mask = mask.collect()

        # Use module-level predicate for JIT caching
        pred = _keep_pred
        d_out = cp.empty(self.length, dtype=self.dtype)
        d_num_selected = cp.empty(1, dtype=np.int32)
        self_it, restore_self = self._get_nestable_iterator()
        in_iter = iterators.ZipIterator(self_it, d_mask)
        restore_self()
        out_iter = iterators.ZipIterator(d_out, iterators.DiscardIterator())

        algorithms.select(d_in=in_iter, d_out=out_iter, d_num_selected_out=d_num_selected, cond=pred, num_items=self.length)

        out_len = int(d_num_selected.get()[0])
        if out_len == 0:
            return self._empty_result()

        d_kept = d_out[:out_len]
        result = ParrotArray(data=d_kept, dtype=self.dtype)
        result._iterator = d_kept
        result.length = out_len
        # Note: mask is NOT copied - the result is unmasked
        return result

    def apply(self):
        """Force application of any lazy mask, returning a materialized array.

        This is the public API to explicitly trigger mask materialization.
        Useful when you want to ensure the mask is applied before further operations.

        Returns:
            A new ParrotArray with the mask applied (no mask on result)

        Raises:
            ValueError: If called on an array without a mask
        """
        if not self.has_mask:
            raise ValueError("apply() can only be called on masked arrays")
        return self._apply_mask_if_needed()

    def __add__(self, other):
        if not isinstance(other, (int, float, np.number, ParrotArray)):
            return NotImplemented
        return self.map2(other, add_op)

    def __sub__(self, other):
        if not isinstance(other, (int, float, np.number, ParrotArray)):
            return NotImplemented
        return self.map2(other, sub_op)

    def __mul__(self, other):
        if not isinstance(other, (int, float, np.number, ParrotArray)):
            return NotImplemented
        return self.map2(other, mul_op)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, np.number, ParrotArray)):
            return NotImplemented
        return self.map2(other, div_op)

    def __eq__(self, other):
        if not isinstance(other, (int, float, np.number, ParrotArray)):
            return NotImplemented
        return self.map2(other, eq_op)

    def __ne__(self, other):
        if not isinstance(other, (int, float, np.number, ParrotArray)):
            return NotImplemented
        return self.map2(other, neq_op)

    def __mod__(self, other):
        if not isinstance(other, (int, float, np.number, ParrotArray)):
            return NotImplemented
        return self.map2(other, mod_op)

    def __floordiv__(self, other):
        if not isinstance(other, (int, float, np.number, ParrotArray)):
            return NotImplemented
        return self.map2(other, idiv_op)

    def times(self, arg):
        """Multiply each element by a scalar or another array."""
        return self.map2(arg, mul_op)

    def add(self, arg):
        """Add a scalar or another array to each element."""
        return self.map2(arg, add_op)

    def minus(self, arg):
        """Subtract a scalar or another array from each element."""
        return self.map2(arg, sub_op)

    def sq(self):
        """Square each element."""

        def square_op(x):
            return x * x

        self._transforms.append(square_op)
        return self

    def abs(self):
        """Take absolute value of each element."""

        def abs_op(x):
            return x if x >= 0 else -x

        self._transforms.append(abs_op)
        return self

    def map(self, op: Callable):
        """Apply a custom transformation function."""
        self._transforms.append(self._sanitize_op(op))
        return self

    def map2(self, other, op: Callable) -> "ParrotArray":
        """Apply a binary operation element-wise with a scalar or another array.

        Mirrors C++ fusion_array::map2: accepts a scalar or a ParrotArray and a
        binary functor ``(a, b) -> value``.  For scalars the functor is
        partially applied and fused as a TransformIterator; for arrays a
        ZipIterator + TransformIterator is used.  Both paths are fully lazy.

        Args:
            other: A scalar (int/float) or a ParrotArray
            op: A binary function ``(a, b) -> value``

        Returns:
            A new ParrotArray with the operation applied lazily
        """
        if self.has_mask:
            return self._apply_mask_if_needed().map2(other, op)

        if isinstance(other, (int, float, np.number)):
            binder = _OP_TO_SCALAR_BIND.get(op)
            if binder is not None:
                return self._clone().map(binder(other))
            _val = other
            def bound_op(x):
                return op(x, _val)
            return self._clone().map(bound_op)

        if isinstance(other, ParrotArray):
            if other.has_mask:
                return self.map2(other._apply_mask_if_needed(), op)

            if self.length != other.length:
                raise ValueError(f"Shape mismatch: {self.length} vs {other.length}")

            zip_op = _OP_TO_ZIP.get(op)
            if zip_op is None:
                def zip_op(t):
                    return op(t[0], t[1])

            a, restore_a = self._get_nestable_iterator()
            b, restore_b = other._get_nestable_iterator()
            zip_iter = iterators.ZipIterator(a, b)
            restore_a()
            restore_b()

            result = ParrotArray(iterator=zip_iter, dtype=self.dtype, length=self.length)
            result._shape = self._shape
            return result.map(zip_op)

        raise TypeError(f"map2: unsupported operand type {type(other)}")

    def neg(self):
        """Negate each element."""

        def neg_op(x):
            return -x

        self._transforms.append(neg_op)
        return self

    def double(self):
        """Double each element (multiply by 2)."""

        def double_op(x):
            return x * 2

        self._transforms.append(double_op)
        return self

    def half(self):
        """Halve each element (divide by 2)."""

        def half_op(x):
            return x / 2

        self._transforms.append(half_op)
        return self

    def even(self):
        """Check if each element is even (returns 1 for even, 0 for odd)."""

        def even_op(x):
            return 1 if x % 2 == 0 else 0

        self._transforms.append(even_op)
        return self

    def odd(self):
        """Check if each element is odd (returns 1 for odd, 0 for even)."""

        def odd_op(x):
            return 1 if x % 2 == 1 else 0

        self._transforms.append(odd_op)
        return self

    def sqrt(self):
        """Take square root of each element."""

        def sqrt_op(x):
            return x**0.5

        self._transforms.append(sqrt_op)
        return self

    def exp(self):
        """Take exponential (e^x) of each element."""
        import math

        _exp = math.exp

        def exp_op(x):
            return _exp(x)

        exp_op.__annotations__ = {"x": self.dtype, "return": self.dtype}
        self._transforms.append(exp_op)
        return self

    def log(self):
        """Take natural logarithm of each element."""
        import math

        _log = math.log

        def log_op(x):
            return _log(x)

        log_op.__annotations__ = {"x": self.dtype, "return": self.dtype}
        self._transforms.append(log_op)
        return self

    def div(self, arg):
        """Divide each element by a scalar or another array."""
        return self.map2(arg, div_op)

    def idiv(self, arg):
        """Integer-divide each element by a scalar or another array."""
        return self.map2(arg, idiv_op)

    def mod(self, arg):
        """Compute modulo by a scalar or another array."""
        return self.map2(arg, mod_op)

    def gt(self, arg):
        """Element-wise greater-than (scalar or array). Returns 1 or 0."""
        return self.map2(arg, gt_op)

    def gte(self, arg):
        """Element-wise greater-than-or-equal (scalar or array). Returns 1 or 0."""
        return self.map2(arg, gte_op)

    def lt(self, arg):
        """Element-wise less-than (scalar or array). Returns 1 or 0."""
        return self.map2(arg, lt_op)

    def lte(self, arg):
        """Element-wise less-than-or-equal (scalar or array). Returns 1 or 0."""
        return self.map2(arg, lte_op)

    def sum(self, axis: int = 0):
        """Sum all elements using reduction."""
        return self.reduce(_reduce_add, 0, axis=axis)

    def prod(self, axis: int = 0):
        """Multiply all elements using reduction."""
        return self.reduce(_reduce_mul, 1, axis=axis)

    def max(self, value=None, axis: int = 0):
        """Element-wise maximum with a scalar or array, or reduction when no value given."""
        if value is not None:
            return self.map2(value, max_op)

        if self.dtype == np.int32:
            init_val = np.iinfo(np.int32).min
        elif self.dtype == np.int64:
            init_val = np.iinfo(np.int64).min
        else:
            init_val = float("-inf")

        return self.reduce(_reduce_max, init_val, axis=axis)

    def min(self, value=None, axis: int = 0):
        """Element-wise minimum with a scalar or array, or reduction when no value given."""
        if value is not None:
            return self.map2(value, min_op)

        if self.dtype == np.int32:
            init_val = np.iinfo(np.int32).max
        elif self.dtype == np.int64:
            init_val = np.iinfo(np.int64).max
        else:
            init_val = float("inf")

        return self.reduce(_reduce_min, init_val, axis=axis)

    def all(self, axis: int = 0):
        """Check if all elements are non-zero (logical AND)."""
        return self.reduce(_reduce_and, True, axis=axis)

    def any(self, axis: int = 0):
        """Check if any element is non-zero (logical OR)."""
        return self.reduce(_reduce_or, False, axis=axis)

    def maxr(self, axis: int = 0):
        """Alias for max() - find maximum element."""
        return self.max(axis=axis)

    def minr(self, axis: int = 0):
        """Alias for min() - find minimum element."""
        return self.min(axis=axis)

    def reduce(self, op: Callable, init_value: Any, axis: int = 0):
        """Perform reduction operation using cuda.compute.reduce_into or reduce_by_key.

        For masked arrays, the mask is applied first before reduction.
        """
        # Apply mask if present before reducing
        if self.has_mask:
            return self._apply_mask_if_needed().reduce(op, init_value, axis)

        op = self._sanitize_op(op)

        if axis == 0:
            final_iterator = self._get_composed_iterator()

            # Set up reduction
            h_init = np.array([init_value], dtype=self.dtype)
            d_output = cp.empty(1, dtype=self.dtype)

            # Instantiate reduction
            algorithms.reduce_into(d_in=final_iterator, d_out=d_output, num_items=self.length, op=op, h_init=h_init)

            return d_output[0].get().item()

        if axis == 1 or axis == 2:
            if self.shape is None or len(self.shape) != 2:
                raise ValueError("Row-wise reduction requires 2D array")

            rows, cols = self.shape

            # Use segmented_reduce
            # Offsets: start[i] = i * cols, end[i] = (i + 1) * cols

            def start_offset_op(i):
                return i * cols

            def end_offset_op(i):
                return (i + 1) * cols

            # Manually annotate to ensure integer types for offsets
            start_offset_op.__annotations__ = {"i": np.int32, "return": np.int32}
            end_offset_op.__annotations__ = {"i": np.int32, "return": np.int32}

            start_offsets = iterators.TransformIterator(
                iterators.CountingIterator(np.int32(0)), start_offset_op
            )

            end_offsets = iterators.TransformIterator(
                iterators.CountingIterator(np.int32(0)), end_offset_op
            )

            d_out = cp.empty(rows, dtype=self.dtype)
            h_init = np.array([init_value], dtype=self.dtype)

            algorithms.segmented_reduce(
                d_in=self._get_composed_iterator(),
                d_out=d_out,
                num_segments=rows,
                start_offsets_in=start_offsets,
                end_offsets_in=end_offsets,
                op=op,
                h_init=h_init,
            )

            result_arr = ParrotArray(data=d_out, dtype=self.dtype)
            result_arr._iterator = d_out
            result_arr.length = rows

            return result_arr

        raise ValueError(f"Unsupported axis: {axis}. Use 0 (global) or 1/2 (row-wise).")

    def scan(self, op: Callable = None, init_value: Any = 0):
        """Perform inclusive scan operation.

        For masked arrays, the mask is applied first (materializing the filter).
        """
        # Apply mask if present before scan operation
        if self.has_mask:
            return self._apply_mask_if_needed().scan(op, init_value)

        if op is None:
            op = _reduce_add

        op = self._sanitize_op(op)
        final_iterator = self._get_composed_iterator()

        # Set up scan
        h_init = np.array([init_value], dtype=self.dtype)
        d_output = cp.empty(self.length, dtype=self.dtype)

        # Instantiate scan
        algorithms.inclusive_scan(d_in=final_iterator, d_out=d_output, op=op, init_value=h_init, num_items=self.length)

        return d_output.get()

    def _scan_to_array(self, op: Callable, init_value: Any):
        """Perform scan and return result as ParrotArray for chaining.

        For masked arrays, the mask is applied first (materializing the filter).
        """
        # Apply mask if present before scan operation
        if self.has_mask:
            return self._apply_mask_if_needed()._scan_to_array(op, init_value)

        op = self._sanitize_op(op)
        final_iterator = self._get_composed_iterator()

        h_init = np.array([init_value], dtype=self.dtype)
        d_output = cp.empty(self.length, dtype=self.dtype)

        algorithms.inclusive_scan(d_in=final_iterator, d_out=d_output, op=op, init_value=h_init, num_items=self.length)

        result = ParrotArray(data=d_output, dtype=self.dtype)
        result._iterator = d_output
        result.length = self.length
        return result

    def sums(self):
        """Cumulative sum scan."""
        return self._scan_to_array(_reduce_add, 0)

    def prods(self):
        """Cumulative product scan."""
        return self._scan_to_array(_reduce_mul, 1)

    def maxs(self):
        """Cumulative maximum scan."""
        # Use appropriate minimum value for the data type
        if self.dtype == np.int32:
            init_val = np.iinfo(np.int32).min
        elif self.dtype == np.int64:
            init_val = np.iinfo(np.int64).min
        else:
            init_val = float("-inf")
        return self._scan_to_array(_reduce_max, init_val)

    def mins(self):
        """Cumulative minimum scan."""
        # Use appropriate maximum value for the data type
        if self.dtype == np.int32:
            init_val = np.iinfo(np.int32).max
        elif self.dtype == np.int64:
            init_val = np.iinfo(np.int64).max
        else:
            init_val = float("inf")
        return self._scan_to_array(_reduce_min, init_val)

    def alls(self):
        """Cumulative logical AND scan."""
        return self.scan(_reduce_and, True)

    def anys(self):
        """Cumulative logical OR scan."""
        return self.scan(_reduce_or, False)

    def collect(self):
        """Collect results into a CuPy array.

        For masked arrays, this applies the mask first (materializing the filter).
        """
        # Apply mask if present before collecting
        if self.has_mask:
            return self._apply_mask_if_needed().collect()

        # Fast path: if data is already a raw CuPy array with no transforms, return it directly
        if self._data is not None and not self._transforms:
            arr = self._data
            if len(arr) != self.length:
                arr = arr[: self.length]
            if self._shape is not None and self._shape != (self.length,):
                arr = arr.reshape(self._shape)
            return arr
        if isinstance(self._iterator, cp.ndarray) and not self._transforms:
            arr = self._iterator
            if len(arr) != self.length:
                arr = arr[: self.length]
            if self._shape is not None and self._shape != (self.length,):
                arr = arr.reshape(self._shape)
            return arr

        final_iterator = self._get_composed_iterator()

        # Use unary_transform to materialize the iterator
        d_output = cp.empty(self.length, dtype=self.dtype)

        # Use module-level identity functions for JIT caching
        if self.dtype in (np.float32, np.float64):
            identity_op = _identity_float
        else:
            identity_op = _identity_int

        # Instantiate and run transform (no temporary storage needed)
        algorithms.unary_transform(d_in=final_iterator, d_out=d_output, op=identity_op, num_items=self.length)

        if self._shape is not None and self._shape != (self.length,):
            return d_output.reshape(self._shape)

        return d_output

    def size(self):
        """Return the number of elements in the array.

        For masked arrays, this returns the count of elements where mask is truthy.
        This is computed lazily without materializing the filtered array.
        """
        if self.has_mask:
            # Count the number of truthy mask values
            return self._mask.gt(0).sum()
        return self.length

    def value(self):
        """Extract the scalar value from a single-element array.

        For masked arrays, the mask is applied first (materializing the filter).

        Returns:
            The first element as a Python scalar.
        """
        result = self.collect()  # collect() handles mask application
        if len(result) == 0:
            raise ValueError("Cannot get value from empty array")
        return result[0].get().item()

    def _make_adj_iter(self, d_data, offset, transforms):
        """Create iterator for adjacent pairs: applies transforms to zip(d_data[i], d_data[i+offset])."""
        it = iterators.ZipIterator(d_data[:-offset], d_data[offset:])
        for t in transforms:
            it = iterators.TransformIterator(it, t)
        return it

    def map_adj(self, op: Callable):
        """Apply a binary operation to adjacent pairs of elements.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            op: A binary function that takes (a, b) and returns a result

        Returns:
            A new ParrotArray with length-1 elements containing op(arr[i], arr[i+1])

        LAZY operation — materializes input first to ensure correct random access
        via PermutationIterator.
        """
        if self.has_mask:
            return self._apply_mask_if_needed().map_adj(op)

        if self.length < 2:
            return self._empty_result()

        zip_op = _OP_TO_ZIP.get(op)
        if zip_op is None:
            raise ValueError(
                f"map_adj only supports known binary ops: {', '.join(op.__name__ for op in _OP_TO_ZIP.keys())}"
            )

        # Materialize so PermutationIterator has a raw array for random access
        d_data = self.collect()
        tuple_op = self._sanitize_op(zip_op)

        idx_iter = iterators.CountingIterator(np.int32(0))
        idx_next_iter = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), _plus_one
        )

        iter_a = iterators.PermutationIterator(d_data, idx_iter)
        iter_b = iterators.PermutationIterator(d_data, idx_next_iter)
        zip_iter = iterators.ZipIterator(iter_a, iter_b)

        result = ParrotArray(iterator=zip_iter, dtype=self.dtype)
        result.length = self.length - 1
        result._transforms.append(tuple_op)
        result._base_cupy = d_data
        return result

    def differ(self):
        """Check if adjacent elements are different.

        Returns:
            A ParrotArray containing 1 where arr[i] != arr[i+1], 0 otherwise
        """
        return self.map_adj(neq_op)

    def deltas(self):
        """Compute differences between adjacent elements.

        Returns:
            A ParrotArray containing arr[i+1] - arr[i] for each adjacent pair
        """
        return self.map_adj(delta_op)

    def prepend(self, value):
        """Add a value at the beginning of the array.

        Args:
            value: The value to prepend

        Returns:
            A new ParrotArray with the value at the beginning
        """
        if self.length == 0:
            result = ParrotArray(dtype=self.dtype)
            result._iterator = iterators.ConstantIterator(self.dtype(value))
            result.length = 1
            return result

        size = self.length
        d_data = self.collect()
        prepend_value = self.dtype(value)

        def prepend_index_op(idx):
            return idx - 1 if idx > 0 else 0

        def prepend_value_op(t):
            return prepend_value if t[0] == 0 else t[1]

        index_iter = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), prepend_index_op
        )
        perm_iter = iterators.PermutationIterator(d_data, index_iter)
        zip_iter = iterators.ZipIterator(
            iterators.CountingIterator(np.int32(0)), perm_iter
        )
        prepend_iter = iterators.TransformIterator(zip_iter, prepend_value_op)

        result = ParrotArray(iterator=prepend_iter, dtype=self.dtype)
        result.length = size + 1
        result._base_cupy = d_data
        return result

    def append(self, value):
        """Add a value at the end of the array.

        Args:
            value: The value to append

        Returns:
            A new ParrotArray with the value at the end
        """
        if self.length == 0:
            result = ParrotArray(dtype=self.dtype)
            result._iterator = iterators.ConstantIterator(self.dtype(value))
            result.length = 1
            return result

        size = self.length
        d_data = self.collect()
        append_value = self.dtype(value)

        def append_index_op(idx):
            return idx if idx < size else 0

        def append_value_op(t):
            return t[1] if t[0] < size else append_value

        index_iter = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), append_index_op
        )
        perm_iter = iterators.PermutationIterator(d_data, index_iter)
        zip_iter = iterators.ZipIterator(
            iterators.CountingIterator(np.int32(0)), perm_iter
        )
        append_iter = iterators.TransformIterator(zip_iter, append_value_op)

        result = ParrotArray(iterator=append_iter, dtype=self.dtype)
        result.length = size + 1
        result._base_cupy = d_data
        return result

    def where(self):
        """Get 1-indexed positions where elements are non-zero.

        For masked arrays, the mask is applied first (materializing the filter).

        Returns:
            A new ParrotArray containing 1-indexed positions where elements are truthy
        """
        # Apply mask if present before where operation
        if self.has_mask:
            return self._apply_mask_if_needed().where()

        return range(self.length, dtype=self.dtype).keep(self) + 1

    def rev(self):
        """Reverse the array.

        SEMI-LAZY operation - uses ReverseIterator if data is already materialized,
        otherwise collects first (reverse requires random access to the data).

        For masked arrays, the mask is applied first (materializing the filter).

        Returns:
            A new ParrotArray with elements in reverse order
        """
        # Must have materialized data for reverse (need random access)
        # Collect if we only have a transform chain (or mask)
        if self._data is not None and not self._transforms and not self.has_mask:
            d_data = self._data
        else:
            d_data = self.collect()  # collect() handles mask application

        # Use ReverseIterator for lazy reverse access
        rev_iter = iterators.ReverseIterator(d_data)

        result = ParrotArray(dtype=self.dtype)
        result._iterator = rev_iter
        result.length = len(d_data)
        result._base_cupy = d_data  # Keep reference to prevent GC
        return result

    def sort(self):
        """Sort the array in ascending order.

        For masked arrays, the mask is applied first (materializing the filter).

        Returns:
            A new ParrotArray with sorted elements
        """
        # Apply mask if present (collect handles this)
        d_data = self.collect()
        d_sorted = cp.sort(d_data)

        result = ParrotArray(data=d_sorted, dtype=self.dtype)
        result._iterator = d_sorted
        result.length = len(d_sorted)
        return result

    def back(self):
        """Get the last element of the array.

        For masked arrays, the mask is applied first (materializing the filter).

        Returns:
            The last element as a Python scalar
        """
        d_data = self.collect()  # collect() handles mask application
        if len(d_data) == 0:
            raise ValueError("Cannot get back from empty array")
        return d_data[-1].get().item()

    def front(self):
        """Get the first element of the array.

        For masked arrays, the mask is applied first (materializing the filter).

        Returns:
            The first element as a Python scalar
        """
        d_data = self.collect()  # collect() handles mask application
        if len(d_data) == 0:
            raise ValueError("Cannot get front from empty array")
        return d_data[0].get().item()

    def uniq(self):
        """Remove adjacent duplicate elements.

        For masked arrays, the mask is applied first (materializing the filter).

        Returns:
            A new ParrotArray with adjacent duplicates removed
        """
        # Apply mask if present before uniq operation
        if self.has_mask:
            return self._apply_mask_if_needed().uniq()

        if self.length == 0:
            return self._empty_result()
        # Use GPU unique_by_key to remove adjacent duplicates.
        in_keys = self._get_composed_iterator()
        in_items = iterators.CountingIterator(np.int32(0))

        d_out_keys = cp.empty(self.length, dtype=self.dtype)
        d_out_items = cp.empty(self.length, dtype=np.int32)
        d_out_num_selected = cp.empty(1, dtype=np.int32)

        algorithms.unique_by_key(
            d_in_keys=in_keys,
            d_in_items=in_items,
            d_out_keys=d_out_keys,
            d_out_items=d_out_items,
            d_out_num_selected=d_out_num_selected,
            op=eq_op,
            num_items=self.length,
        )

        out_len = int(d_out_num_selected.get()[0])
        d_unique = d_out_keys[:out_len]

        result = ParrotArray(data=d_unique, dtype=self.dtype)
        result._iterator = d_unique
        result.length = out_len
        return result

    def distinct(self):
        """Remove all duplicate elements (sort then uniq).

        Returns:
            A new ParrotArray with all duplicates removed
        """
        return self.sort().uniq()

    def chunk_by_reduce(self, pred, op):
        """Group consecutive elements by predicate and reduce each group.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            pred: Binary predicate that returns true if two elements belong to same group
            op: Binary reduction operation to apply within each group

        Returns:
            A new ParrotArray with one reduced value per group
        """
        # Apply mask if present (collect handles this)
        if self.length == 0:
            return self._empty_result()

        d_data = self.collect()  # collect() handles mask application
        h_data = d_data.get()

        # Group consecutive elements and reduce
        groups = []
        current_val = h_data[0]
        for i in builtins.range(1, len(h_data)):
            if pred(h_data[i - 1], h_data[i]):
                # Same group - reduce
                current_val = op(current_val, h_data[i])
            else:
                # New group - save current and start new
                groups.append(current_val)
                current_val = h_data[i]
        groups.append(current_val)  # Don't forget the last group

        d_groups = cp.array(groups, dtype=self.dtype)
        result = ParrotArray(data=d_groups, dtype=self.dtype)
        result._iterator = d_groups
        result.length = len(groups)
        return result

    def min_with(self, other):
        """Elementwise minimum with another array.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            other: Another ParrotArray of the same length

        Returns:
            A new ParrotArray with elementwise minimum values
        """
        # Apply masks if present (collect handles this)
        d_self = self.collect()
        d_other = other.collect()

        if len(d_self) != len(d_other):
            raise ValueError(f"Shape mismatch: {len(d_self)} vs {len(d_other)}")

        d_result = cp.minimum(d_self, d_other)

        result = ParrotArray(data=d_result, dtype=self.dtype)
        result._iterator = d_result
        result.length = len(d_result)
        return result

    def max_with(self, other):
        """Elementwise maximum with another array.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            other: Another ParrotArray of the same length

        Returns:
            A new ParrotArray with elementwise maximum values
        """
        # Apply masks if present (collect handles this)
        d_self = self.collect()
        d_other = other.collect()

        if len(d_self) != len(d_other):
            raise ValueError(f"Shape mismatch: {len(d_self)} vs {len(d_other)}")

        d_result = cp.maximum(d_self, d_other)

        result = ParrotArray(data=d_result, dtype=self.dtype)
        result._iterator = d_result
        result.length = len(d_result)
        return result

    def match(self, other):
        """Check if this array matches another element-by-element.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            other: Another ParrotArray to compare with

        Returns:
            True if all elements are equal, False otherwise
        """
        # Apply masks if present (collect handles this)
        d_self = self.collect()
        d_other = other.collect()

        if len(d_self) != len(d_other):
            return False

        return bool(cp.all(d_self == d_other))

    def keep(self, mask):
        """Keep elements where the mask is truthy (LAZY operation).

        This is a lazy operation - the mask is stored but not immediately applied.
        The actual filtering (copy_if/select) only happens when:
        - A reduction operation (sum, max, etc.) is called
        - collect() is called to materialize the array
        - Operations that require contiguous data (sort, uniq, etc.) are called
        - apply() is explicitly called

        This allows operations like .keep(mask).sum() to potentially use
        optimized masked reduction without materializing the filtered array.

        Args:
            mask: A ParrotArray mask (same length) with truthy values to keep

        Returns:
            A new ParrotArray with lazy mask stored (not yet applied)
        """
        if self.length == 0:
            return self._empty_result()

        # Convert mask to ParrotArray if needed
        if isinstance(mask, ParrotArray):
            if mask.length != self.length:
                raise ValueError(f"Shape mismatch: {self.length} vs {mask.length}")
            mask_arr = mask
        else:
            d_mask = cp.asarray(mask, dtype=np.int32)
            if len(d_mask) != self.length:
                raise ValueError(f"Shape mismatch: {self.length} vs {len(d_mask)}")
            mask_arr = ParrotArray(data=d_mask, dtype=np.int32)
            mask_arr._iterator = d_mask
            mask_arr.length = len(d_mask)

        # If we already have a mask, we need to combine them (AND operation)
        # For now, apply the existing mask first, then store the new one
        if self.has_mask:
            # Apply existing mask, then add new mask
            unmasked = self._apply_mask_if_needed()
            return unmasked.keep(mask_arr)

        # Create a new array with the mask stored lazily
        result = self._clone()
        result._mask = mask_arr
        return result

    def take(self, n: int):
        """Take the first n elements of the array.

        LAZY operation - just adjusts the length without materializing or copying.
        Since iterators are positional streams, reducing the length is sufficient.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            n: Number of elements to take

        Returns:
            A new ParrotArray with the first n elements

        Raises:
            ValueError: If n < 0 or n > size()
        """
        # Apply mask if present before take operation
        if self.has_mask:
            return self._apply_mask_if_needed().take(n)

        if n < 0 or n > self.length:
            raise ValueError(f"take: n must be between 0 and {self.length}, got {n}")

        if n == 0:
            return self._empty_result()

        result = self._clone()
        result.length = n
        return result

    def filter(self, predicate):
        """Filter elements based on a predicate function.

        For masked arrays, the mask is applied first (materializing the filter),
        then the predicate filter is applied.

        Args:
            predicate: A function that takes an element and returns True to keep it

        Returns:
            A new ParrotArray with only the elements for which predicate returns True
        """
        # Apply mask if present before filter operation
        if self.has_mask:
            return self._apply_mask_if_needed().filter(predicate)

        if self.length == 0:
            return self._empty_result()

        pred = self._sanitize_predicate(predicate)
        d_out = cp.empty(self.length, dtype=self.dtype)
        d_num_selected = cp.empty(1, dtype=np.int32)
        in_iter = self._get_composed_iterator()

        algorithms.select(d_in=in_iter, d_out=d_out, d_num_selected_out=d_num_selected, cond=pred, num_items=self.length)

        out_len = int(d_num_selected.get()[0])
        if out_len == 0:
            return self._empty_result()

        d_kept = d_out[:out_len]
        result = ParrotArray(data=d_kept, dtype=self.dtype)
        result._iterator = d_kept
        result.length = out_len
        return result

    def sign(self):
        """Get sign of each element: -1 for negative, 0 for zero, 1 for positive.

        Returns:
            A new ParrotArray with sign values
        """

        def sign_op(x):
            return 1 if x > 0 else (-1 if x < 0 else 0)

        self._transforms.append(sign_op)
        return self

    def rand(self):
        """Generate random integers between 0 and each element.

        LAZY operation - creates a transform iterator that generates random values
        on-demand, without materializing the array until collect() is called.

        For masked arrays, the mask is applied first (materializing the filter).

        For each element val[i], generates a random integer in [0, val[i]).

        Returns:
            A new ParrotArray with random integer values

        Note:
            Uses deterministic randomness based on element index and a per-call
            entropy seed. Same entropy + same indices = same random values.
        """
        # Apply mask if present before rand operation
        if self.has_mask:
            return self._apply_mask_if_needed().rand()

        global _global_rand_counter
        _global_rand_counter += 1

        # Generate entropy from multiple sources (similar to C++ rand_functor)
        # Mix: counter, time, random bits, and object address
        extra_entropy = (
            (random.getrandbits(32))
            ^ (int(time.time() * 1000000) & 0xFFFFFFFF)
            ^ (_global_rand_counter * 2654435761)
            ^ (id(self) & 0xFFFFFFFF)
        ) & 0xFFFFFFFF

        # Create the random operation with captured entropy
        rand_op = _make_rand_op(extra_entropy)

        # Create counting iterator for indices
        indices_iter = iterators.CountingIterator(np.int32(0))

        # Get current composed iterator (with all transforms applied)
        values_iter, restore_values = self._get_nestable_iterator()

        # Create zip iterator combining (index, value)
        zip_iter = iterators.ZipIterator(indices_iter, values_iter)
        restore_values()

        # Create transform iterator that applies the random functor
        rand_iter = iterators.TransformIterator(zip_iter, rand_op)

        # Return new lazy ParrotArray
        result = ParrotArray(iterator=rand_iter, dtype=self.dtype)
        result.length = self.length
        return result

    def neq(self, arg):
        """Element-wise inequality (scalar or array). Returns 1 or 0."""
        return self.map2(arg, neq_op)

    def gather(self, indices):
        """Gather elements at the specified indices (lazy permutation iterator).

        Like the C++ fusion_array::gather, this returns a new ParrotArray backed
        by a PermutationIterator so no data is materialized — the gather is
        fused into subsequent kernels.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            indices: A ParrotArray of indices to gather

        Returns:
            A new ParrotArray with elements at the specified indices
        """
        # If masked, we need contiguous data first (mask -> select)
        source = self._apply_mask_if_needed() if self.has_mask else self

        base_iter, restore_base = source._get_nestable_iterator()
        idx_iter, restore_idx = indices._get_nestable_iterator()

        perm_iter = iterators.PermutationIterator(base_iter, idx_iter)
        restore_base()
        restore_idx()

        result = ParrotArray(iterator=perm_iter, dtype=self.dtype)
        result.length = indices.length
        return result

    def reshape(self, shape):
        """Reshape the array to the specified dimensions.

        Args:
            shape: An integer or tuple/list of integers specifying the new shape.

        Returns:
            A new ParrotArray with the specified shape.

        Raises:
            ValueError: If the total size of the new shape exceeds the current size
                       or if trying to reshape to/from empty shape invalidly.
        """
        if isinstance(shape, int):
            shape = (shape,)

        # Calculate total size
        total_size = functools.reduce(lambda x, y: x * y, shape, 1)

        if self.length == 0:
            if total_size != 0:
                raise ValueError("Cannot reshape empty array to non-empty shape")
        elif total_size == 0:
            raise ValueError(
                "Cannot reshape non-empty array to empty shape (size must be > 0)"
            )

        if total_size > self.length:
            raise ValueError(
                f"reshape: total size {total_size} must be <= current size {self.length}; "
                "use cycle() for larger shapes"
            )

        new_arr = self._clone()
        new_arr.length = total_size
        new_arr._shape = tuple(shape)
        return new_arr

    def flatten(self):
        """Flatten the array to rank 1.

        Returns:
            A new ParrotArray with the same data but flattened to 1D.
        """
        return self.reshape((self.length,))

    def cycle(self, shape):
        """Cycle the array data to fill ``shape`` (lazy if expanding).

        Mirrors C++ ``fusion_array::cycle``:
          * If ``prod(shape) > current_size``, the data is cycled (``i % N``)
            to fill the new shape — backed by a lazy PermutationIterator so no
            data is copied up-front.
          * If ``prod(shape) <= current_size``, this behaves like
            :meth:`reshape` (a view truncation; no iterator composition).

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            shape: An int or tuple/list of ints specifying the target shape.

        Returns:
            A new ParrotArray with the specified shape.

        Raises:
            ValueError: If the current array is empty or ``prod(shape) == 0``.
        """
        if self.has_mask:
            return self._apply_mask_if_needed().cycle(shape)

        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(shape)

        total_size = functools.reduce(lambda x, y: x * y, shape, 1)
        current_size = self.length

        if current_size == 0 or total_size == 0:
            raise ValueError(
                "cycle: current_size and total_size must be > 0"
            )

        if total_size <= current_size:
            return self.reshape(shape)

        _n = current_size

        def cycle_idx(i):
            return i % _n

        cycle_idx.__annotations__ = {"i": np.int32, "return": np.int32}

        base_iter, restore_base = self._get_nestable_iterator()
        index_iter = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), cycle_idx
        )
        cycle_iter = iterators.PermutationIterator(base_iter, index_iter)
        restore_base()

        result = ParrotArray(iterator=cycle_iter, dtype=self.dtype)
        result.length = total_size
        result._shape = shape
        return result

    def nrows(self):
        """Get the number of rows in a 2D array.

        Returns:
            Number of rows (first dimension)

        Raises:
            ValueError: If array is not 2D
        """
        if self._shape is None or len(self._shape) != 2:
            raise ValueError("nrows() requires a 2D array")
        return self._shape[0]

    def ncols(self):
        """Get the number of columns in a 2D array.

        Returns:
            Number of columns (second dimension)

        Raises:
            ValueError: If array is not 2D
        """
        if self._shape is None or len(self._shape) != 2:
            raise ValueError("ncols() requires a 2D array")
        return self._shape[1]

    def astype(self, dtype):
        """Convert the array to a different dtype.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            dtype: The target numpy dtype (e.g., np.float32, np.int64)

        Returns:
            A new ParrotArray with the specified dtype
        """
        # Apply mask if present before type conversion
        if self.has_mask:
            return self._apply_mask_if_needed().astype(dtype)

        # LAZY: add a cast transform
        src_dtype = self.dtype

        # Decision must be made outside the function (Numba can't do runtime type checks)
        if dtype in (np.float32, np.float64):

            def cast_op(x):
                return float(x)

        else:

            def cast_op(x):
                return int(x)

        cast_op.__annotations__ = {"x": src_dtype, "return": dtype}

        result = ParrotArray(dtype=dtype)
        result._iterator = self._get_composed_iterator()
        result._transforms.append(cast_op)
        result.length = self.length
        result._shape = self._shape
        return result

    def replicate(self, arg):
        """Replicate each element by ``arg``.

        Two overloads mirroring C++ ``fusion_array::replicate``:

        * ``replicate(n: int)`` — each element is repeated ``n`` times (LAZY,
          backed by a PermutationIterator).  For ``[a, b, c].replicate(2)``
          this yields ``[a, a, b, b, c, c]``.  Useful for broadcasting
          row-wise reductions back to full matrix size.

        * ``replicate(mask: ParrotArray)`` — element ``i`` is repeated
          ``mask[i]`` times (MATERIALIZING, via ``cupy.repeat``).  For
          ``[1, 2, 3].replicate([2, 1, 3])`` this yields
          ``[1, 1, 2, 3, 3, 3]``.  cuda.compute does not expose a
          ``scatter_if``/segmented-scan primitive, so this overload uses
          CuPy's native ``repeat`` kernel to compute the gather pattern.

        For masked (this-side) arrays, the mask is applied first.

        Args:
            arg: Either a positive int (scalar count) or a ParrotArray of
                non-negative integer counts matching ``self.length``.

        Returns:
            A new ParrotArray with the replicated elements.

        Raises:
            ValueError: If ``arg`` is a non-positive int; or a mask with the
                wrong length or negative values.
            TypeError: If ``arg`` is not an int or ParrotArray.
        """
        if self.has_mask:
            return self._apply_mask_if_needed().replicate(arg)

        if isinstance(arg, ParrotArray):
            return self._replicate_by_mask(arg)

        if not isinstance(arg, (int, np.integer)):
            raise TypeError(
                "replicate: expected int or ParrotArray, got "
                f"{type(arg).__name__}"
            )
        n = int(arg)

        if n <= 0:
            raise ValueError("replicate: n must be positive")

        if n == 1:
            return self._clone()

        # Materialize so PermutationIterator has a raw array for random access
        d_data = self.collect()

        def replicate_index_op(idx):
            return idx // n

        replicate_index_op.__annotations__ = {"idx": np.int32, "return": np.int32}

        index_iter = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), replicate_index_op
        )
        replicate_iter = iterators.PermutationIterator(d_data, index_iter)

        result = ParrotArray(iterator=replicate_iter, dtype=self.dtype)
        result.length = self.length * n
        result._base_cupy = d_data
        return result

    def _replicate_by_mask(self, mask: "ParrotArray") -> "ParrotArray":
        """Mask-based replicate (materializing): element i repeated mask[i] times.

        Implementation note: cuda.compute lacks ``scatter_if``, so the
        scatter+scan-max trick used in the C++ backend is not directly
        available here.  Instead we use CuPy's ``cp.repeat`` — a single
        optimized native kernel that computes the same gather pattern.
        """
        if mask.has_mask:
            return self._replicate_by_mask(mask._apply_mask_if_needed())

        if self.length != mask.length:
            raise ValueError(
                f"replicate: mask size {mask.length} must match "
                f"array size {self.length}"
            )

        d_mask = mask.collect()
        if not isinstance(d_mask, cp.ndarray):
            d_mask = cp.asarray(d_mask)
        d_mask = d_mask.reshape(-1)

        if self.length > 0 and int(d_mask.min().get()) < 0:
            raise ValueError("replicate: mask values must be non-negative")

        d_src = self.collect()
        if not isinstance(d_src, cp.ndarray):
            d_src = cp.asarray(d_src)
        d_src = d_src.reshape(-1)

        # CuPy's repeat only accepts ``int``, ``list``, or ``tuple`` for the
        # ``repeats`` argument (neither cupy.ndarray nor numpy.ndarray work),
        # so we materialize the mask to a Python list on the host. The actual
        # gather still happens on-device via CuPy's internal kernel; since
        # replicate(mask) is a materializing op, this host copy is acceptable.
        h_mask = d_mask.tolist()
        d_out = cp.repeat(d_src, h_mask)

        result = ParrotArray(data=d_out, dtype=self.dtype)
        result._iterator = d_out
        result.length = int(d_out.size)
        return result

    def repeat(self, n: int):
        """Repeat a scalar value n times to create a 1D array.

        Only works on single-element (scalar) arrays.
        For arrays with more elements, use replicate() or cycle() instead.

        Args:
            n: Number of repetitions (must be > 0)

        Returns:
            A new ParrotArray of length n with the scalar value repeated

        Raises:
            ValueError: If the array has more than one element or n <= 0
        """
        if self.length != 1:
            raise ValueError("repeat: array must be a scalar (length 1)")
        if n <= 0:
            raise ValueError("repeat: n must be > 0")
        return constant(self.front(), n, self.dtype)

    def cross(self, other):
        """Cartesian product with another array (lazy).

        For [1, 2] and [a, b], produces a zipped array of length 4
        representing pairs: [(1,a), (1,b), (2,a), (2,b)].
        Apply .map(tuple_op) to combine the pairs.

        Mirrors C++ replicate + cycle + pairs.

        Args:
            other: Another ParrotArray

        Returns:
            A new ParrotArray backed by a ZipIterator over the expanded arrays
        """
        if self.length == 0 or other.length == 0:
            raise ValueError("cross: arrays must not be empty")

        total = self.length * other.length
        left = self.replicate(other.length)

        _other_len = other.length
        def cycle_idx(i):
            return i % _other_len
        cycle_idx.__annotations__ = {"i": np.int32, "return": np.int32}

        cycle_index_iter = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), cycle_idx
        )
        right_base, restore_right = other._get_nestable_iterator()
        right_iter = iterators.PermutationIterator(right_base, cycle_index_iter)
        restore_right()

        left_iter, restore_left = left._get_nestable_iterator()
        zip_iter = iterators.ZipIterator(left_iter, right_iter)
        restore_left()

        result = ParrotArray(iterator=zip_iter, dtype=self.dtype)
        result.length = total
        return result

    def pairs(self, other: "ParrotArray") -> "ParrotArray":
        """Pair each element with the corresponding element of ``other`` (lazy).

        For arrays ``[a, b, c]`` and ``[x, y, z]``, produces a zipped array of
        pairs ``[(a, x), (b, y), (c, z)]``.  Apply ``.map(tuple_op)`` to combine
        the pairs.

        Mirrors C++ ``fusion_array::pairs(other)``.

        Args:
            other: Another ParrotArray with the same length as ``self``.

        Returns:
            A new ParrotArray backed by a ZipIterator over ``(self, other)``.

        Raises:
            TypeError: If ``other`` is not a ParrotArray.
            ValueError: If the two arrays have different lengths.
        """
        if self.has_mask:
            return self._apply_mask_if_needed().pairs(other)
        if not isinstance(other, ParrotArray):
            raise TypeError(
                f"pairs: expected ParrotArray, got {type(other).__name__}"
            )
        if other.has_mask:
            return self.pairs(other._apply_mask_if_needed())
        if self.length != other.length:
            raise ValueError(
                f"pairs: length mismatch {self.length} vs {other.length}"
            )

        a, restore_a = self._get_nestable_iterator()
        b, restore_b = other._get_nestable_iterator()
        zip_iter = iterators.ZipIterator(a, b)
        restore_a()
        restore_b()

        result = ParrotArray(iterator=zip_iter, dtype=self.dtype)
        result.length = self.length
        result._pair_parts = (self, other)
        return result

    def enumerate(self):
        """Pair each element with its index (lazy).

        For [a, b, c], produces a zipped array of pairs:
        [(a, 0), (b, 1), (c, 2)].
        Apply .map(tuple_op) to combine the pairs.

        Mirrors C++ enumerate() which returns pairs(range(size)).

        Returns:
            A new ParrotArray backed by a ZipIterator of (value, index)
        """
        indices_iter = iterators.CountingIterator(np.int32(0))
        values_iter, restore_values = self._get_nestable_iterator()
        zip_iter = iterators.ZipIterator(values_iter, indices_iter)
        restore_values()

        result = ParrotArray(iterator=zip_iter, dtype=self.dtype)
        result.length = self.length
        result._pair_parts = (self, None)
        return result

    def transpose(self):
        """Transpose a 2D array (lazy).

        Uses a permutation iterator with transposed index mapping so no
        data is copied.  For a matrix with shape (R, C), returns a new
        array with shape (C, R).

        Returns:
            A new ParrotArray representing the transposed matrix

        Raises:
            ValueError: If the array is not rank 2
        """
        if self._shape is None or len(self._shape) != 2:
            raise ValueError("transpose: array must be rank 2 (a matrix)")

        nrows, ncols = self._shape
        total = nrows * ncols

        _nrows = nrows
        _ncols = ncols
        def transpose_idx(i):
            return (i % _nrows) * _ncols + i // _nrows
        transpose_idx.__annotations__ = {"i": np.int32, "return": np.int32}

        base_iter, restore_base = self._get_nestable_iterator()
        index_iter = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), transpose_idx
        )
        perm_iter = iterators.PermutationIterator(base_iter, index_iter)
        restore_base()

        result = ParrotArray(iterator=perm_iter, dtype=self.dtype)
        result.length = total
        result._shape = (ncols, nrows)
        return result

    def to_host(self):
        """Get the array data as a Python list.

        For masked arrays, the mask is applied first (materializing the filter).

        Returns:
            A list of Python values
        """
        d_data = self.collect()  # collect() handles mask application
        return d_data.get().tolist()

    def outer(self, other, op):
        """Compute the outer product with another array as a 2D matrix (lazy operation).

        For arrays [1, 2] and [a, b] with operation op, returns a 2×2 matrix:
            [[op(1, a), op(1, b)],
             [op(2, a), op(2, b)]]

        This is much more efficient than the cycle-based approach as it uses
        lazy iterators that compute values on-demand without materializing
        intermediate arrays.

        For masked arrays, the mask is applied first (materializing the filter).

        Args:
            other: The other ParrotArray to compute the outer product with
            op: A binary operation taking two arguments (x, y). Can be:
                - A known binary op (add_op, mul_op, sub_op, etc.)
                - A custom binary lambda: lambda x, y: x * y + 1

        Returns:
            A new ParrotArray containing the results in a 2D matrix shape
            with dimensions (self.length, other.length)

        Raises:
            ValueError: If either array is empty
            ValueError: If other is not a ParrotArray
            ValueError: If op is not a binary function

        Example:
            >>> a = array([1, 2, 3])
            >>> b = array([10, 20])
            >>> result = a.outer(b, mul_op)
            >>> # Returns 3x2 matrix: [[10, 20], [20, 40], [30, 60]]
            >>> result = a.outer(b, lambda x, y: x + y * 2)
            >>> # Custom binary operation
        """
        # Apply masks if present
        if self.has_mask:
            return self._apply_mask_if_needed().outer(other, op)
        if isinstance(other, ParrotArray) and other.has_mask:
            return self.outer(other._apply_mask_if_needed(), op)

        if not isinstance(other, ParrotArray):
            raise ValueError("outer: other must be a ParrotArray")

        this_size = self.length
        other_size = other.length

        if this_size == 0 or other_size == 0:
            raise ValueError("outer: arrays must not be empty")

        # Check if it's a known binary op with a pre-defined tuple version
        zip_op = _OP_TO_ZIP.get(op)
        if zip_op is None:
            # Must be a binary function - convert to tuple form using AST
            try:
                sig = inspect.signature(op)
                num_params = len(sig.parameters)
                if num_params != 2:
                    raise ValueError(
                        f"outer: op must be a binary function (2 arguments), got {num_params}"
                    )
            except (ValueError, TypeError):
                pass  # Can't inspect signature, try conversion anyway

            zip_op = _binary_to_tuple_op(op)

        zip_op = self._sanitize_op(zip_op)

        # Materialize both inputs so PermutationIterator has raw arrays
        d_self = self.collect()
        d_other = other.collect()

        def row_index_op(idx):
            return idx // other_size

        def col_index_op(idx):
            return idx % other_size

        row_index_op.__annotations__ = {"idx": np.int32, "return": np.int32}
        col_index_op.__annotations__ = {"idx": np.int32, "return": np.int32}

        row_indices = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), row_index_op
        )
        col_indices = iterators.TransformIterator(
            iterators.CountingIterator(np.int32(0)), col_index_op
        )

        self_gathered = iterators.PermutationIterator(d_self, row_indices)
        other_gathered = iterators.PermutationIterator(d_other, col_indices)

        zip_iter = iterators.ZipIterator(self_gathered, other_gathered)
        outer_iter = iterators.TransformIterator(zip_iter, zip_op)

        result = ParrotArray(iterator=outer_iter, dtype=self.dtype)
        result.length = this_size * other_size
        result._shape = (this_size, other_size)
        result._base_cupy = d_self
        return result


def range(n: int, dtype=np.int32):
    """Create a range of numbers starting from 0."""
    return ParrotArray().range(n, dtype)


def constant(value: Union[int, float], length: int, dtype=np.int32):
    """Create a constant sequence with given value and length."""
    return ParrotArray().constant(value, length, dtype)


def from_array(arr):
    """Create from existing array."""
    return ParrotArray(data=arr)


def array(data, dtype=np.int32):
    """Create a ParrotArray from a list or array.

    Args:
        data: A list, tuple, or array-like of values
        dtype: The data type for the array (default: np.int32)

    Returns:
        A ParrotArray containing the data on the GPU
    """
    # Convert to CuPy array on GPU
    d_data = cp.array(data, dtype=dtype)

    # Create a ParrotArray - CuPy arrays can be used directly as iterators
    result = ParrotArray(data=d_data, dtype=dtype)
    result._iterator = d_data  # CuPy array works as an iterator
    result.length = len(d_data)
    return result


def scalar(value: Union[int, float], dtype=np.int32):
    """Create a single-element array with the given value."""
    return constant(value, 1, dtype)


def matrix(rows: int, cols: int, value: Union[int, float] = 0, dtype=np.int32):
    """Create a matrix with given dimensions and value."""
    return constant(value, rows * cols, dtype).reshape((rows, cols))


# Module-level helper functions for iterator operations (enables JIT caching)
def _plus_one(idx: np.int32) -> np.int32:
    """Increment index by 1 for adjacent pair iteration."""
    return idx + 1


def _identity_int(x: np.int64) -> np.int64:
    """Identity function for integer types."""
    return int(x)


def _identity_float(x: np.float64) -> np.float64:
    """Identity function for float types."""
    return float(x)


def _keep_pred(t) -> np.uint8:
    """Predicate for keep/filter: returns 1 if mask element is truthy."""
    return np.uint8(1) if t[1] else np.uint8(0)


def _add_one(x) -> np.int64:
    """Add 1 to x - used by where() for 1-indexed results."""
    return x + 1


def _eq_two(x) -> np.int32:
    """Check if x equals 2 - commonly used for length-2 segment filtering."""
    return np.int32(1) if x == 2 else np.int32(0)


# Pre-annotated reduction ops (avoids creating closures + inspect calls per sum/max/etc.)
def _reduce_add(a, b):
    return a + b


def _reduce_mul(a, b):
    return a * b


def _reduce_max(a, b):
    return a if a > b else b


def _reduce_min(a, b):
    return a if a < b else b


def _reduce_and(a, b):
    return a and b


def _reduce_or(a, b):
    return a or b


for _rop in [
    _reduce_add,
    _reduce_mul,
    _reduce_max,
    _reduce_min,
    _reduce_and,
    _reduce_or,
]:
    _rop.__annotations__ = {"a": np.int64, "b": np.int64, "return": np.int64}


# Binary operators for map_adj and other operations
# Using operator module and concise functions
add_op = operator.add
sub_op = operator.sub
mul_op = operator.mul
div_op = operator.truediv
min_op = min
max_op = max


def eq_op(a, b):
    return int(a == b)


def neq_op(a, b):
    return int(a != b)


def lt_op(a, b):
    return int(a < b)


def gt_op(a, b):
    return int(a > b)


def gte_op(a, b):
    return int(a >= b)


def lte_op(a, b):
    return int(a <= b)


mod_op = operator.mod
idiv_op = operator.floordiv


def delta_op(a, b):
    return b - a


# Pre-annotate all binary ops
for _op in [eq_op, neq_op, lt_op, gt_op, gte_op, lte_op, delta_op]:
    _op.__annotations__ = {"a": np.int64, "b": np.int64, "return": np.int64}


# Named, pre-annotated tuple versions for _OP_TO_ZIP (avoids lambda sanitization overhead)
def _zip_add(t):
    return t[0] + t[1]


def _zip_sub(t):
    return t[0] - t[1]


def _zip_mul(t):
    return t[0] * t[1]


def _zip_div(t):
    return t[0] / t[1]


def _zip_eq(t):
    return 1 if t[0] == t[1] else 0


def _zip_min(t):
    return t[0] if t[0] < t[1] else t[1]


def _zip_max(t):
    return t[0] if t[0] > t[1] else t[1]


def _zip_delta(t):
    return t[1] - t[0]


def _zip_neq(t):
    return 1 if t[0] != t[1] else 0


def _zip_lt(t):
    return 1 if t[0] < t[1] else 0


def _zip_gt(t):
    return 1 if t[0] > t[1] else 0


def _zip_gte(t):
    return 1 if t[0] >= t[1] else 0


def _zip_lte(t):
    return 1 if t[0] <= t[1] else 0


def _zip_mod(t):
    return t[0] % t[1]


def _zip_idiv(t):
    return t[0] // t[1]


# Note: zip ops intentionally have NO annotations.
# The cuda.compute library infers input types from the ZipIterator,
# and _sanitize_op will skip them because their __name__ != "<lambda>".


_OP_TO_ZIP = {
    add_op: _zip_add,
    sub_op: _zip_sub,
    mul_op: _zip_mul,
    div_op: _zip_div,
    eq_op: _zip_eq,
    min_op: _zip_min,
    max_op: _zip_max,
    delta_op: _zip_delta,
    neq_op: _zip_neq,
    lt_op: _zip_lt,
    gt_op: _zip_gt,
    gte_op: _zip_gte,
    lte_op: _zip_lte,
    mod_op: _zip_mod,
    idiv_op: _zip_idiv,
}


# Scalar binder factories for map2's scalar path.
# Numba's CUDA JIT cannot compile closures that capture Python function
# objects as free variables.  These factories produce closures that only
# capture a scalar value, which Numba handles natively.
def _bind_add(v):
    def f(x): return x + v
    return f

def _bind_sub(v):
    def f(x): return x - v
    return f

def _bind_mul(v):
    def f(x): return x * v
    return f

def _bind_div(v):
    def f(x): return x / v
    return f

def _bind_idiv(v):
    def f(x): return x // v
    return f

def _bind_mod(v):
    def f(x): return x % v
    return f

def _bind_gt(v):
    def f(x): return 1 if x > v else 0
    return f

def _bind_gte(v):
    def f(x): return 1 if x >= v else 0
    return f

def _bind_lt(v):
    def f(x): return 1 if x < v else 0
    return f

def _bind_lte(v):
    def f(x): return 1 if x <= v else 0
    return f

def _bind_eq(v):
    def f(x): return 1 if x == v else 0
    return f

def _bind_neq(v):
    def f(x): return 1 if x != v else 0
    return f

def _bind_min(v):
    def f(x): return v if x > v else x
    return f

def _bind_max(v):
    def f(x): return v if x < v else x
    return f

_OP_TO_SCALAR_BIND = {
    add_op: _bind_add,
    sub_op: _bind_sub,
    mul_op: _bind_mul,
    div_op: _bind_div,
    idiv_op: _bind_idiv,
    mod_op: _bind_mod,
    gt_op: _bind_gt,
    gte_op: _bind_gte,
    lt_op: _bind_lt,
    lte_op: _bind_lte,
    eq_op: _bind_eq,
    neq_op: _bind_neq,
    min_op: _bind_min,
    max_op: _bind_max,
}


# Public API exports
__all__ = [
    "ParrotArray",
    "range",
    "constant",
    "from_array",
    "scalar",
    "matrix",
    "array",
    "min_op",
    "max_op",
    "lt_op",
    "gt_op",
    "eq_op",
    "neq_op",
    "add_op",
    "sub_op",
    "delta_op",
    "mul_op",
    "div_op",
]
