How to Use Parrot
=================

This guide shows you how to integrate Parrot into your C++ projects using CMake.

CMake Integration
-----------------

To integrate Parrot into your CMake project, you can use FetchContent to automatically download and configure the library along with its dependencies:

.. code-block:: cmake

   # Include FetchContent module for downloading dependencies
   include(FetchContent)

   # Fetch the official parrot library from NVlabs/parrot
   FetchContent_Declare(
     parrot
     GIT_REPOSITORY https://github.com/NVlabs/parrot.git
     GIT_TAG main
     GIT_SHALLOW TRUE
   )

   # Make parrot available
   FetchContent_MakeAvailable(parrot)

   # Add parrot include directory from FetchContent
   FetchContent_GetProperties(parrot)
   if(parrot_POPULATED)
       include_directories(${parrot_SOURCE_DIR})
       message(STATUS "Using parrot library from: ${parrot_SOURCE_DIR}")
       
       # Get CCCL from parrot's build and prioritize it over system headers
       # This ensures we use Thrust 3.2.0+ instead of system Thrust 3.0.1
       FetchContent_GetProperties(cccl)
       if(cccl_POPULATED)
           include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR})
           include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/cub)
           include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/thrust)
           include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/libcudacxx/include)
           
           # Add compiler flags to prioritize CCCL headers
           set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}")
           set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/cub")
           set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/thrust")
           set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/libcudacxx/include")
           
           message(STATUS "Using CCCL (Thrust 3.2.0+) from: ${cccl_SOURCE_DIR}")
       endif()
   endif()

Why This Configuration?
-----------------------

This configuration ensures that your project uses the correct version of Thrust (3.2.0+) and other CCCL components that Parrot depends on. The key benefits are:

* **Dependency Management**: Automatically downloads and configures Parrot and its dependencies
* **Version Compatibility**: Ensures you use Thrust 3.2.0+ instead of potentially older system-installed versions
* **Header Prioritization**: Uses ``BEFORE SYSTEM`` and ``-isystem`` flags to prioritize CCCL headers over system headers
* **Shallow Cloning**: Uses ``GIT_SHALLOW TRUE`` for faster downloads

Basic Usage
-----------

Once integrated, you can use Parrot in your CUDA C++ code:

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

For more detailed examples, see the :doc:`examples` section.

Requirements
------------

* CUDA-capable GPU
* CUDA Toolkit 13.0 or later
* CMake 3.18 or later
* C++20 compatible compiler

Building Your Project
---------------------

After setting up the CMake configuration, build your project as usual:

.. code-block:: bash

   mkdir build && cd build
   cmake ..
   make -j$(nproc)

The FetchContent mechanism will automatically handle downloading and building Parrot and its dependencies during the first build.
