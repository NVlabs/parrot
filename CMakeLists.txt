cmake_minimum_required(VERSION 3.10)

# Set CUDA compiler path before project declaration
# Only do dynamic detection if CMAKE_CUDA_COMPILER is not already set
if(NOT DEFINED CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER STREQUAL "")
    # Try to find NVCC dynamically, with fallback to common locations
    find_program(NVCC_EXECUTABLE nvcc
        HINTS
            $ENV{CUDA_HOME}/bin
            $ENV{CUDA_PATH}/bin
            /usr/local/cuda/bin
            /opt/cuda/bin
        DOC "NVIDIA CUDA Compiler"
    )

    if(NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
        message(STATUS "Auto-detected NVCC: ${NVCC_EXECUTABLE}")
    else()
        # Fallback to your specific installation if dynamic detection fails
        set(FALLBACK_NVCC "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin/nvcc")
        if(EXISTS ${FALLBACK_NVCC})
            set(CMAKE_CUDA_COMPILER ${FALLBACK_NVCC})
            message(STATUS "Using fallback NVCC: ${FALLBACK_NVCC}")
        else()
            message(FATAL_ERROR "Could not find NVCC compiler. Please set CMAKE_CUDA_COMPILER manually or ensure NVCC is in your PATH.")
        endif()
    endif()
else()
    message(STATUS "Using manually specified NVCC: ${CMAKE_CUDA_COMPILER}")
endif()

# Initialize project with just CXX language first
project(parrot LANGUAGES CXX)

# Configure ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    # Set ccache as the launcher for both C++ and CUDA
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found - compile performance may be reduced")
endif()

# CUDA Architecture Configuration
# Option 1: User can specify via -DCUDA_ARCH=75,89
# Option 2: Auto-detect from current GPU
# Option 3: Build for common architectures
set(CUDA_ARCH "AUTO" CACHE STRING "CUDA architectures to build for (e.g., '75', '75,89', 'AUTO', 'ALL')")

# Now enable CUDA language
enable_language(CUDA)

# Function to detect GPU architecture using nvidia-smi
function(detect_gpu_architecture output_var)
    # Try using nvidia-smi first (faster and more reliable)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE nvidia_smi_output
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(nvidia_smi_output)
            # Take the first GPU's compute capability
            string(REGEX MATCH "^[0-9]+\\.[0-9]+" first_gpu_cap "${nvidia_smi_output}")
            if(first_gpu_cap)
                string(REPLACE "." "" arch_output "${first_gpu_cap}")
                set(${output_var} "${arch_output}" PARENT_SCOPE)
                message(STATUS "Detected GPU architecture via nvidia-smi: ${arch_output}")
                return()
            endif()
        endif()
    endif()
    
    # Fallback: try using deviceQuery if available
    find_program(DEVICE_QUERY deviceQuery PATHS
        /usr/local/cuda/extras/demo_suite
        /opt/cuda/extras/demo_suite
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../extras/demo_suite
    )
    
    if(DEVICE_QUERY)
        execute_process(
            COMMAND ${DEVICE_QUERY}
            OUTPUT_VARIABLE device_query_output
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(device_query_output)
            string(REGEX MATCH "CUDA Capability Major/Minor version number:[^0-9]*([0-9]+)\\.([0-9]+)" 
                   match_result "${device_query_output}")
            if(CMAKE_MATCH_1 AND CMAKE_MATCH_2)
                set(arch_output "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
                set(${output_var} "${arch_output}" PARENT_SCOPE)
                message(STATUS "Detected GPU architecture via deviceQuery: ${arch_output}")
                return()
            endif()
        endif()
    endif()
    
    # If all detection methods fail
    message(WARNING "Failed to detect GPU architecture - no nvidia-smi or deviceQuery found")
    set(${output_var} "" PARENT_SCOPE)
endfunction()

# Set CUDA architectures based on user choice
if(CUDA_ARCH STREQUAL "AUTO")
    detect_gpu_architecture(detected_arch)
    if(detected_arch)
        set(CMAKE_CUDA_ARCHITECTURES ${detected_arch})
        message(STATUS "Using auto-detected CUDA architecture: ${detected_arch}")
    else()
        # Fallback to common architectures if detection fails
        set(CMAKE_CUDA_ARCHITECTURES "75;89")
        message(STATUS "Auto-detection failed. Building for architectures: 75 (Turing/RTX), 89 (Ampere)")
    endif()
elseif(CUDA_ARCH STREQUAL "ALL")
    # Build for common modern architectures
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    message(STATUS "Building for all common CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    # Use user-specified architectures
    string(REPLACE "," ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler timing flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-report")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --time")

# CUDA settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Include directories - prioritize CCCL over system CUB
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})


# Enable testing
enable_testing()

# Download and configure doctest
include(FetchContent)

# Set timeout and retry options for FetchContent
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11 # Using a specific stable tag with 'v' prefix
  GIT_SHALLOW TRUE
  TIMEOUT 30
)

# Download NVIDIA CCCL for latest CUB/Thrust
FetchContent_Declare(
  cccl
  GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
  GIT_TAG main # Use latest main branch for newest features
  GIT_SHALLOW TRUE
  TIMEOUT 60
)

# Make dependencies available with timeout handling
message(STATUS "Downloading doctest... (this may take a moment)")
FetchContent_MakeAvailable(doctest)

message(STATUS "Downloading NVIDIA CCCL... (this may take a moment)")
FetchContent_MakeAvailable(cccl)

# Add CCCL include directories with highest priority to override system headers
include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR})
include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/cub)  
include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/thrust)
include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/libcudacxx/include)
message(STATUS "Added CCCL include directories (highest priority): ${cccl_SOURCE_DIR}")

# Force CCCL headers to be found first via compiler flags
# Use -isystem to suppress warnings and -idirafter to ensure system headers come after
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR} -isystem ${cccl_SOURCE_DIR}/cub -isystem ${cccl_SOURCE_DIR}/thrust -isystem ${cccl_SOURCE_DIR}/libcudacxx/include")

# Create test directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Add test executable with the new modular structure
add_executable(parrot_tests tests/test_main.cu)

# Set CUDA options for the test executable
set_target_properties(parrot_tests PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(parrot_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Link DocTest
target_link_libraries(parrot_tests PRIVATE doctest::doctest)

# Register test
add_test(NAME ParrotTests COMMAND parrot_tests)

# Optional: Add individual test executables for focused testing
# These can be useful for debugging specific functionality

add_executable(test_basic tests/test_basic_operations.cu tests/test_common.hpp)
set_target_properties(test_basic PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_basic PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_basic PRIVATE doctest::doctest)

add_executable(test_sorting tests/test_sorting.cu tests/test_common.hpp)
set_target_properties(test_sorting PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_sorting PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_sorting PRIVATE doctest::doctest)

add_executable(test_math tests/test_math_operations.cu tests/test_common.hpp)
set_target_properties(test_math PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_math PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_math PRIVATE doctest::doctest)

add_executable(test_reductions tests/test_reductions.cu tests/test_common.hpp)
set_target_properties(test_reductions PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_reductions PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_reductions PRIVATE doctest::doctest)

add_executable(test_scans tests/test_scans.cu tests/test_common.hpp)
set_target_properties(test_scans PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_scans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_scans PRIVATE doctest::doctest)

# Add new test executables for the missing categories
add_executable(test_array_ops tests/test_array_operations.cu tests/test_common.hpp)
set_target_properties(test_array_ops PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_array_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_array_ops PRIVATE doctest::doctest)

add_executable(test_advanced tests/test_advanced_operations.cu tests/test_common.hpp)
set_target_properties(test_advanced PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_advanced PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_advanced PRIVATE doctest::doctest)

add_executable(test_multidim tests/test_multidimensional.cu tests/test_common.hpp)
set_target_properties(test_multidim PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_multidim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_multidim PRIVATE doctest::doctest)

add_executable(test_integration tests/test_integration.cu tests/test_common.hpp)
set_target_properties(test_integration PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_integration PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_integration PRIVATE doctest::doctest)

add_executable(test_top10 tests/test_top10.cu tests/test_common.hpp)
set_target_properties(test_top10 PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_top10 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_top10 PRIVATE doctest::doctest)

add_executable(test_fun tests/test_fun.cu tests/test_common.hpp)
set_target_properties(test_fun PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_fun PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(test_fun PRIVATE doctest::doctest)

# Register individual tests with CTest
add_test(NAME BasicOperationsTests COMMAND test_basic)
add_test(NAME SortingTests COMMAND test_sorting)
add_test(NAME MathOperationsTests COMMAND test_math)
add_test(NAME ReductionsTests COMMAND test_reductions)
add_test(NAME ScansTests COMMAND test_scans)
add_test(NAME ArrayOperationsTests COMMAND test_array_ops)
add_test(NAME AdvancedOperationsTests COMMAND test_advanced)
add_test(NAME MultidimensionalTests COMMAND test_multidim)
add_test(NAME IntegrationTests COMMAND test_integration)
add_test(NAME Top10Tests COMMAND test_top10)
add_test(NAME FunTests COMMAND test_fun)

# Add starter examples as executables
file(GLOB STARTER_EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/examples/starter/*.cu")
foreach(example_path ${STARTER_EXAMPLES})
    get_filename_component(example_name ${example_path} NAME_WE)
    set(target_name "${example_name}_starter")
    
    add_executable(${target_name} ${example_path})
    set_target_properties(${target_name} PROPERTIES 
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endforeach()

# Add softmax example
add_executable(softmax softmax.cu)
set_target_properties(softmax PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(softmax PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    # Doxygen
    add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
    
    # Sphinx (optional)
    find_program(SPHINX_EXECUTABLE NAMES sphinx-build)
    if(SPHINX_EXECUTABLE)
        add_custom_target(docs
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/docs/build
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
            COMMAND ${SPHINX_EXECUTABLE} -b html
            ${CMAKE_CURRENT_SOURCE_DIR}/docs
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/build/html
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Building HTML documentation with Sphinx"
            VERBATIM)
    endif()
endif()
