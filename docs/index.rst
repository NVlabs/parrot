Welcome to the Parrot Docs
--------------------------

Parrot is a C++ library for fused array operations using CUDA/Thrust. It provides efficient GPU-accelerated operations with fused evaluation semantics, allowing for chaining of operations without unnecessary intermediate materializations. 

Compare examples from the CUDA C++ library :doc:`Thrust <parrot_v_thrust>`.

Explore the code on `GitHub <https://github.com/NVlabs/parrot>`_. If you would like to contribue, please refer to the `contributing <https://github.com/NVlabs/parrot/blob/main/CONTRIBUTING.md>`_ and `building <https://github.com/NVlabs/parrot/blob/main/BUILDING.md>`_ guides.

Features
~~~~~~~~

* Implicit fusion¹ of array operations
* GPU acceleration using CUDA/Thrust
* Chainable operations with a clean API

¹ - Any operation that **can** be fused **is** fused.

Quick Start
~~~~~~~~~~~

.. code-block:: cpp

   #include "parrot.hpp"
   #include <iostream>
   
   int main() {
       // Create a matrix
       auto matrix = parrot::range(10000).as<float>().reshape({100, 100});

       // Calculate the row-wise softmax of a matrix
       auto cols = matrix.ncols();
       auto z    = matrix - matrix.maxr<2>().replicate(cols);
       auto num  = z.exp();
       auto den  = num.sum<2>();
       (num / den.replicate(cols)).print();
       
       return 0;
   }

**Performance Comparison of Softmax:**

.. raw:: html
   :file: _static/softmax.html

For a full screen version, click `here <_static/softmax.html>`_.

Find more examples :doc:`here <examples>`.

Contents
~~~~~~~~

.. toctree::
   :maxdepth: 1
   
   how_to_use
   fusion_array
   examples

.. toctree::
   :maxdepth: 1
   
   parrot_v_thrust
   

