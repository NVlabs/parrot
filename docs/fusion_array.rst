fusion_array
=============

Overview
--------

The ``fusion_array`` class is the core of Parrot, providing fused evaluation of array operations.

.. note::
   You can create arrays in various ways:

   .. code-block:: cpp
   
      // Using template argument deduction with initializer list
      auto arr2 = parrot::array({1, 2, 3, 4});      // Deduces int 
      auto arr3 = parrot::array({1.5, 2.5, 3.5});   // Deduces double
      
      // Creating a range (1-indexed)
      auto arr4 = parrot::range(10);                // Creates [1,2,3...10]
      
      // Creating a scalar array
      auto scalar_arr = parrot::scalar(5);
      
      // Creating a matrix with all elements set to a value
      auto mat = parrot::matrix(3.14, {3, 4});      // Creates a 3x4 matrix filled with 3.14

Class Documentation
-------------------

Fused Operations
----------------

1-index Maps (Unary)
~~~~~~~~~~~~~~~~~~~~

.. _cp-fusion-array-map:

.. doxygenfunction:: parrot::fusion_array::map

.. _cp-fusion-array-abs:

.. doxygenfunction:: parrot::fusion_array::abs

.. _cp-fusion-array-dble:

.. doxygenfunction:: parrot::fusion_array::dble

.. _cp-enumerate:

.. doxygenfunction:: parrot::fusion_array::enumerate

.. _cp-fusion-array-even:

.. doxygenfunction:: parrot::fusion_array::even

.. _cp-fusion-array-half:

.. doxygenfunction:: parrot::fusion_array::half

.. _cp-fusion-array-log:

.. doxygenfunction:: parrot::fusion_array::log

.. _cp-fusion-array-exp:

.. doxygenfunction:: parrot::fusion_array::exp

.. _cp-fusion-array-neg:

.. doxygenfunction:: parrot::fusion_array::neg

.. _cp-fusion-array-odd:

.. doxygenfunction:: parrot::fusion_array::odd

.. _cp-fusion-array-rand:

.. doxygenfunction:: parrot::fusion_array::rand

.. _cp-fusion-array-sign:

.. doxygenfunction:: parrot::fusion_array::sign

.. _cp-fusion-array-sq:

.. doxygenfunction:: parrot::fusion_array::sq

.. _cp-fusion-array-sqrt:

.. doxygenfunction:: parrot::fusion_array::sqrt

1-index Maps (Binary)
~~~~~~~~~~~~~~~~~~~~~

.. _cp-fusion-array-map2:

.. doxygenfunction:: parrot::fusion_array::map2(const fusion_array<OtherIterator> &value, BinaryFunctor binary_op) const
.. doxygenfunction:: parrot::fusion_array::map2(const T &value, BinaryFunctor binary_op) const

.. _cp-fusion-array-add:

.. doxygenfunction:: parrot::fusion_array::add

.. _cp-fusion-array-div:

.. doxygenfunction:: parrot::fusion_array::div

.. _cp-fusion-array-gt:

.. doxygenfunction:: parrot::fusion_array::gt

.. _cp-fusion-array-gte:

.. doxygenfunction:: parrot::fusion_array::gte

.. _cp-fusion-array-idiv:

.. doxygenfunction:: parrot::fusion_array::idiv

.. _cp-fusion-array-lt:

.. doxygenfunction:: parrot::fusion_array::lt

.. _cp-fusion-array-lte:

.. doxygenfunction:: parrot::fusion_array::lte

.. _cp-fusion-array-max:

.. doxygenfunction:: parrot::fusion_array::max

.. _cp-fusion-array-min:

.. doxygenfunction:: parrot::fusion_array::min

.. _cp-fusion-array-minus:

.. doxygenfunction:: parrot::fusion_array::minus

.. _cp-fusion-array-times:

.. doxygenfunction:: parrot::fusion_array::times

.. _cp-fusion-array-eq:

.. doxygenfunction:: parrot::fusion_array::eq

.. _cp-pairs:

.. doxygenfunction:: parrot::fusion_array::pairs

2-index Maps
~~~~~~~~~~~~

.. _cp-fusion-array-map-adj:

.. doxygenfunction:: parrot::fusion_array::map_adj

.. _cp-fusion-array-deltas:

.. doxygenfunction:: parrot::fusion_array::deltas

.. _cp-fusion-array-differ:

.. doxygenfunction:: parrot::fusion_array::differ

Joins
~~~~~

.. _cp-fusion-array-append:

.. doxygenfunction:: parrot::fusion_array::append

.. _cp-fusion-array-prepend:

.. doxygenfunction:: parrot::fusion_array::prepend

Products
~~~~~~~~

.. _cp-fusion-array-cross:

.. doxygenfunction:: parrot::fusion_array::cross

.. note::
   The cross method computes the cartesian product of two arrays. For arrays [1, 2] and [a, b], 
   it returns pairs [(1, a), (1, b), (2, a), (2, b)]. The implementation replicates each 
   element of the first array by the size of the second array, and cycles the second array 
   to match the total size, then pairs them together.

.. _cp-fusion-array-outer:

.. doxygenfunction:: parrot::fusion_array::outer

.. note::
   The outer method computes the outer product of two arrays as a 2D matrix using a binary operation. 
   For arrays [1, 2] and [a, b] with operation `op`, it returns a 2×2 matrix:
   [op(1, a), op(1, b)]
   [op(2, a), op(2, b)]
   This is equivalent to cross(other).map(binary_op_adapter(op)).reshape({this.size(), other.size()}).

Reshapes
~~~~~~~~

.. _cp-fusion-array-take:

.. doxygenfunction:: parrot::fusion_array::take

.. _cp-fusion-array-drop:

.. doxygenfunction:: parrot::fusion_array::drop

.. _cp-fusion-array-transpose:

.. doxygenfunction:: parrot::fusion_array::transpose

.. _cp-fusion-array-reshape:

.. doxygenfunction:: parrot::fusion_array::reshape

.. note::
   The reshape method returns a new fusion_array with the specified shape.
   The total size of the new shape must be less than or equal to the current array size.
   Some data may be truncated if the new size is smaller. For cycling data to fill
   larger shapes, use cycle() instead.

.. _cp-fusion-array-cycle:

.. doxygenfunction:: parrot::fusion_array::cycle

.. note::
   The cycle method returns a new fusion_array with the specified shape.
   If the new shape's total size is greater than the array size, the current data 
   is cycled to fill the new shape. If the new shape's total size is less than or
   equal to the array size, it behaves like reshape().

.. _cp-fusion-array-repeat:

.. doxygenfunction:: parrot::fusion_array::repeat

Copying
~~~~~~~

.. _cp-fusion-array-replicate:

.. doxygenfunction:: parrot::fusion_array::replicate(int n) const

.. note::
   **Scalar overload (Fused/Lazy)**: ``replicate(int n)`` repeats each element of the array n times. 
   For an array [1, 2, 3] with n=2, it returns [1, 1, 2, 2, 3, 3]. This operation uses lazy 
   iterators for efficiency and can be fused with subsequent operations. This is different from 
   repeat() which only works on scalar arrays (rank = 0).

Permutations
~~~~~~~~~~~~

.. _cp-fusion-array-rev:

.. doxygenfunction:: parrot::fusion_array::rev

.. _cp-fusion-array-gather:

.. doxygenfunction:: parrot::fusion_array::gather

Conditionally Fused Operations
-------------------------------

Compactions
~~~~~~~~~~~

.. _cp-fusion-array-keep:

.. doxygenfunction:: parrot::fusion_array::keep

.. _cp-fusion-array-filter:

.. doxygenfunction:: parrot::fusion_array::filter

.. _cp-fusion-array-where:

.. doxygenfunction:: parrot::fusion_array::where

.. _cp-fusion-array-uniq:

.. doxygenfunction:: parrot::fusion_array::uniq

.. _cp-fusion-array-distinct:

.. doxygenfunction:: parrot::fusion_array::distinct

Materializing Operations
------------------------

Reductions
~~~~~~~~~~

.. _cp-fusion-array-reduce:

.. doxygenfunction:: parrot::fusion_array::reduce

.. _cp-fusion-array-all:

.. doxygenfunction:: parrot::fusion_array::all

.. _cp-fusion-array-any:

.. doxygenfunction:: parrot::fusion_array::any

.. _cp-fusion-array-maxr:

.. doxygenfunction:: parrot::fusion_array::maxr

.. _cp-fusion-array-max-by:

.. doxygenfunction:: parrot::fusion_array::max_by_key

.. note::
   The max_by_key method finds the maximum element based on a key extractor function.
   It's especially useful when working with pairs or complex objects where you
   want to find the maximum based on a specific field or computation.

.. _cp-fusion-array-minr:

.. doxygenfunction:: parrot::fusion_array::minr

.. _cp-fusion-array-minmax:

.. doxygenfunction:: parrot::fusion_array::minmax

.. _cp-fusion-array-prod:

.. doxygenfunction:: parrot::fusion_array::prod

.. _cp-fusion-array-sum:

.. doxygenfunction:: parrot::fusion_array::sum

Scans
~~~~~

.. _cp-fusion-array-scan:

.. doxygenfunction:: parrot::fusion_array::scan

.. _cp-fusion-array-alls:

.. doxygenfunction:: parrot::fusion_array::alls

.. _cp-fusion-array-anys:

.. doxygenfunction:: parrot::fusion_array::anys

.. _cp-fusion-array-maxs:

.. doxygenfunction:: parrot::fusion_array::maxs

.. _cp-fusion-array-mins:

.. doxygenfunction:: parrot::fusion_array::mins

.. _cp-fusion-array-prods:

.. doxygenfunction:: parrot::fusion_array::prods

.. _cp-fusion-array-sums:

.. doxygenfunction:: parrot::fusion_array::sums

Permutations
~~~~~~~~~~~~

.. _cp-fusion-array-sort:

.. doxygenfunction:: parrot::fusion_array::sort

.. _cp-fusion-array-sort-by:

.. doxygenfunction:: parrot::fusion_array::sort_by

.. _cp-fusion-array-sort-by-key:

.. doxygenfunction:: parrot::fusion_array::sort_by_key

Compactions
~~~~~~~~~~~

.. _cp-fusion-array-rle:

.. doxygenfunction:: parrot::fusion_array::rle

Copying
~~~~~~~

.. _cp-fusion-array-replicate-mask:

.. doxygenfunction:: parrot::fusion_array::replicate(const fusion_array<MaskIterType> &mask) const

.. note::
   **Mask overload (Materializing)**: ``replicate(mask_array)`` repeats each element according to 
   the corresponding value in the mask array. For an array [1, 2, 3] with mask [2, 1, 3], it 
   returns [1, 1, 2, 3, 3, 3]. This operation requires materialization due to the variable 
   repetition pattern and uses efficient index generation with permutation iterators.
   This is useful for operations like duplicating zeros: ``arr.replicate(arr.eq(0).add(1))``.

Split-Reductions
~~~~~~~~~~~~~~~~

.. _cp-fusion-array-chunk-by-reduce:

.. doxygenfunction:: parrot::fusion_array::chunk_by_reduce

Comparisons
~~~~~~~~~~~

.. _cp-fusion-array-match:

.. doxygenfunction:: parrot::fusion_array::match

Properties
----------

.. _cp-fusion-array-size:

.. doxygenfunction:: parrot::fusion_array::size

.. _cp-fusion-array-rank:

.. doxygenfunction:: parrot::fusion_array::rank

.. _cp-fusion-array-shape:

.. doxygenfunction:: parrot::fusion_array::shape

Accessors
---------

.. _cp-fusion-array-value:

.. doxygenfunction:: parrot::fusion_array::value

.. _cp-fusion-array-front:

.. doxygenfunction:: parrot::fusion_array::front

.. _cp-fusion-array-back:

.. doxygenfunction:: parrot::fusion_array::back

.. _cp-fusion-array-to-host:

.. doxygenfunction:: parrot::fusion_array::to_host

Array Creation
--------------

.. _array-creation-functions:

Parrot provides several convenient functions for creating fusion_array objects. These functions handle memory management automatically and provide efficient lazy evaluation where possible.

.. _cp-array-functions:

.. doxygenfunction:: parrot::array(std::initializer_list<T> init_list)

.. doxygenfunction:: parrot::array(const std::vector<T> &host_vec)

.. _cp-range-function:

.. doxygenfunction:: parrot::range

.. _cp-scalar-function:

.. doxygenfunction:: parrot::scalar


.. _cp-matrix-function:

.. doxygenfunction:: parrot::matrix



I/O
---

.. _cp-fusion-array-print:

.. doxygenfunction:: parrot::fusion_array::print

Function Objects
----------------

Accessors
~~~~~~~~~

.. _cp-fst:

.. code-block:: cpp

   struct parrot::fst {
       template <typename T, typename U>
       __host__ __device__
       T operator()(const thrust::pair<T, U>& p) const;
   };

The ``fst`` function extracts the first element of a pair. Useful with functions like ``max_by_key``.

**Example:**

.. code-block:: cpp

   // Find the pair with the maximum first element
   auto result = pairs.max_by_key(parrot::fst());

.. _cp-snd:

.. code-block:: cpp

   struct parrot::snd {
       template <typename T, typename U>
       __host__ __device__
       U operator()(const thrust::pair<T, U>& p) const;
   };

The ``snd`` function extracts the second element of a pair. Useful with functions like ``max_by_key``.

**Example:**

.. code-block:: cpp

   // Find the pair with the maximum second element
   auto result = pairs.max_by_key(parrot::snd());

Binary Operations
~~~~~~~~~~~~~~~~~

.. _cp-eq:

.. code-block:: cpp

   struct parrot::eq {
       template <typename T>
       __host__ __device__
       bool operator()(const T& a, const T& b) const;
   };

.. _cp-gt:

.. code-block:: cpp

   struct parrot::gt {
       template <typename T>
       __host__ __device__
       bool operator()(const T& a, const T& b) const;
   };

.. _cp-gte:

.. code-block:: cpp

   struct parrot::gte {
       template <typename T>
       __host__ __device__
       bool operator()(const T& a, const T& b) const;
   };

.. _cp-lt:

.. code-block:: cpp

   struct parrot::lt {
       template <typename T>
       __host__ __device__
       bool operator()(const T& a, const T& b) const;
   };

.. _cp-lte:

.. code-block:: cpp

   struct parrot::lte {
       template <typename T>
       __host__ __device__
       bool operator()(const T& a, const T& b) const;
   };

.. _cp-max:

.. code-block:: cpp

   struct parrot::max {
       template <typename T>
       __host__ __device__
       T operator()(const T& a, const T& b) const;
   };

.. _cp-min:

.. code-block:: cpp

   struct parrot::min {
       template <typename T>
       __host__ __device__
       T operator()(const T& a, const T& b) const;
   };

.. _cp-mul:

.. code-block:: cpp

   struct parrot::mul {
       template <typename T>
       __host__ __device__
       T operator()(const T& a, const T& b) const;
   };

.. _cp-add:

.. code-block:: cpp

   struct parrot::add {
       template <typename T>
       __host__ __device__
       T operator()(const T& a, const T& b) const;
   };

Statistical Functions
---------------------

.. _cp-stats-norm-cdf:

.. doxygenfunction:: parrot::stats::norm_cdf

.. note::
   This function computes the cumulative distribution function of the standard normal distribution.
   It's particularly useful for statistical computations and probability calculations.

.. _cp-stats-mode:

.. doxygenfunction:: parrot::stats::mode

.. note::
   This function computes the statistical mode (most frequently occurring value) of an array.
   It returns the value that appears most frequently in the array. If all values appear with 
   equal frequency, it returns the smallest value after sorting. The implementation uses 
   efficient operations: sorting, run-length encoding, and finding the maximum by count.