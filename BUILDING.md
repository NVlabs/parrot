# Building and Testing parrot

This document describes how to build, test, and generate documentation for the parrot project.

## Prerequisites

- CMake (version 3.10 or higher)
- C++ compiler with C++20 support
- NVIDIA CUDA Toolkit 13.0 or later
- NVIDIA GPU with compute capability 7.0 or higher
- Python 3 with pip (for documentation, optional)

<details>
<summary><strong>CUDA Installation</strong> (Click to expand if you need to install CUDA)</summary>

The most common setup issue is the missing NVCC compiler. Here's how to install CUDA properly:

### Option 1: NVIDIA CUDA Toolkit (Recommended)

1. Download the CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Follow the [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for your platform

### Option 2: NVIDIA HPC SDK

1. Download from [NVIDIA HPC SDK Downloads](https://developer.nvidia.com/hpc-sdk-downloads)
2. Follow the [HPC SDK Installation Guide](https://docs.nvidia.com/hpc-sdk/hpc-sdk-install-guide/index.html)

### Environment Configuration

Add these to your `~/.bashrc`, `~/.zshrc`, or equivalent:

```bash
# CUDA installation paths (adjust based on your installation)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# For NVIDIA HPC SDK installations
# export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda
# export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin:$PATH
```

Reload your shell configuration:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Verification

Check that CUDA is properly installed:

```bash
# Check NVCC compiler
nvcc --version

# Check NVIDIA driver
nvidia-smi
```

### Troubleshooting "Could not find NVCC compiler"

If CMake cannot find the NVCC compiler:

1. **Set CMAKE_CUDA_COMPILER manually**:
   ```bash
   cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
   # or for HPC SDK:
   cmake -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvcc ..
   ```

2. **Check NVCC is in PATH**:
   ```bash
   which nvcc
   ```

3. **Install NVIDIA drivers** if `nvidia-smi` doesn't work:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install nvidia-driver-580  # or latest version
   
   # CentOS/RHEL/Fedora
   sudo dnf install akmod-nvidia
   ```

</details>

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
