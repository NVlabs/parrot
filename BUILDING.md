# Building and Testing parrot

This document describes how to build, test, and generate documentation for the parrot project.

## Prerequisites

- CMake (version 3.10 or higher)
- C++ compiler with C++20 support
- NVIDIA CUDA Toolkit 13.0 or later
- NVIDIA GPU with compute capability 7.0 or higher
- Python 3 with pip (for documentation, optional)

## CUDA Architecture Configuration

The project automatically detects your GPU architecture, but you can also configure it manually:

### Auto-detection (Default)
```bash
cmake ..
```
This will automatically detect the GPU architecture on your system.

### Manual Architecture Selection
```bash
# For specific architecture (e.g., RTX 8000 with sm_75)
cmake .. -DCUDA_ARCH=75

# For Ampere GPUs (e.g., RTX 30xx series, A100, laptop GPUs)
cmake .. -DCUDA_ARCH=89

# For multiple architectures (fat binary)
cmake .. -DCUDA_ARCH=75,89

# For all common modern architectures
cmake .. -DCUDA_ARCH=ALL
```

## Building the Project

1. Create a build directory:
   ```bash
   mkdir -p build
   cd build
   ```

2. Configure with CMake:
   ```bash
   cmake ..
   ```

3. Build the project:
   ```bash
   cmake --build .
   ```

## Running Tests

The project uses doctest for unit testing with both comprehensive and individual test executables.

### Run All Tests
```bash
# Run all tests via CTest
ctest

# Or run the main test executable directly
./parrot_tests
```

### Run Individual Test Categories
```bash
# Basic operations
./test_basic

# Sorting algorithms
./test_sorting

# Mathematical operations
./test_math

# Reduction operations
./test_reductions

# Scan operations
./test_scans

# Array operations
./test_array_ops

# Advanced operations
./test_advanced

# Multidimensional operations
./test_multidim

# Integration tests
./test_integration
```

## Building Documentation

The project uses Doxygen and optionally Sphinx for documentation.

### Prerequisites

Install required Python packages (optional, for Sphinx):
```bash
pip install sphinx sphinx-rtd-theme breathe
```

## Generate Documentation

1. Generate Doxygen XML documentation:
   ```bash
   doxygen Doxyfile
   ```

2. Build Sphinx documentation:
   ```bash
   cd docs
   sphinx-build -b html . build/html
   ```

   The generated HTML documentation will be available in `docs/build/html/`.
 
## Viewing Documentation
 
Open the HTML documentation in your browser:
```bash
open docs/build/html/index.html
