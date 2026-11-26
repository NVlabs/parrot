# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Guard a lazy value read before its producer is evaluated.

ZipIterator evaluates all children before a Python callback sees the tuple.
This iterator instead checks selection first. The small CCCL adapter uses the
iterator-extension interfaces shipped in cuda-cccl 1.2.1.
"""

from textwrap import dedent

from cuda.compute._cpp_compile import compile_cpp_op_code, make_variable_declaration
from cuda.compute.iterators import IteratorBase
from cuda.compute.iterators._base import compose_iterator_states
from cuda.compute.iterators._common import CUDA_PREAMBLE, ensure_iterator
from cuda.compute.op import Op, OpKind


class MaskedIterator(IteratorBase):
    """Read a value when its mask is nonzero, otherwise return zero."""

    def __init__(self, values, mask):
        self._values = ensure_iterator(values)
        self._mask = ensure_iterator(mask)
        state, alignment, offsets = compose_iterator_states(list(self.children))
        self._value_offset, self._mask_offset = offsets
        super().__init__(state, alignment, self._values.value_type)

    @property
    def children(self):
        return self._values, self._mask

    @property
    def kind(self):
        return ("ParrotMaskedIterator", self._values.kind, self._mask.kind)

    def _make_advance_op(self):
        value_op = self._values.get_advance_op()
        mask_op = self._mask.get_advance_op()
        symbol = self._make_advance_symbol()
        source = dedent(f"""
            {CUDA_PREAMBLE}
            extern "C" __device__ void {value_op.name}(void*, void*);
            extern "C" __device__ void {mask_op.name}(void*, void*);
            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                char* bytes = static_cast<char*>(state);
                {value_op.name}(bytes + {self._value_offset}, offset);
                {mask_op.name}(bytes + {self._mask_offset}, offset);
            }}
        """)
        return self._compile(symbol, source, value_op, mask_op)

    def _make_input_deref_op(self):
        value_op = self._values.get_input_deref_op()
        mask_op = self._mask.get_input_deref_op()
        symbol = self._make_input_deref_symbol()
        mask_value = make_variable_declaration(self._mask.value_type, "selected")
        source = dedent(f"""
            {CUDA_PREAMBLE}
            extern "C" __device__ void {value_op.name}(void*, void*);
            extern "C" __device__ void {mask_op.name}(void*, void*);
            extern "C" __device__ void {symbol}(void* state, void* result) {{
                char* bytes = static_cast<char*>(state);
                {mask_value}
                {mask_op.name}(bytes + {self._mask_offset}, &selected);
                if (selected != 0) {{
                    {value_op.name}(bytes + {self._value_offset}, result);
                }} else {{
                    cuda::std::memset(result, 0, {self.value_type.info.size});
                }}
            }}
        """)
        return self._compile(symbol, source, value_op, mask_op)

    def _make_output_deref_op(self):
        return None

    @staticmethod
    def _compile(symbol, source, *dependencies):
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=compile_cpp_op_code(source),
            extra_ltoirs=[
                code
                for dependency in dependencies
                for code in (dependency.code, *dependency.extra_code)
            ],
        )

    def __add__(self, offset):
        return MaskedIterator(self._values + offset, self._mask + offset)
