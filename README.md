<div align="center">
  <img src="docs/_static/logo.png" alt="Parrot C++" width="300">
  <h1><b>Parrot</b></h1>
</div>

<p align="center">
    <a href="https://github.com/nvlabs/parrot/issues" alt="contributions welcome">
        <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat" /></a>
    <a href="https://en.cppreference.com/w/cpp/compiler_support/11">
        <img src="https://img.shields.io/badge/C++%20-20-ff69b4.svg"/></a>
    <a href="https://nvlabs.github.io/parrot/" alt="Documentation">
        <img src="https://img.shields.io/badge/docs-latest-blue.svg" /></a>
</p>

**Parrot** is a C++ library for fused array operations using CUDA/Thrust. It provides efficient GPU-accelerated operations with lazy evaluation semantics, allowing for chaining of operations without unnecessary intermediate materializations.

## ✨ Features

- **Fused Operations** - Operations are fused when possible
- **GPU Acceleration** - Built on CUDA/Thrust for high performance
- **Chainable API** - Clean, expressive syntax for complex operations
- **Header-Only** - Simple integration with `#include "parrot.hpp"`

¹ *Lazy-ish means that any operation that **can** be lazily evaluated **is** lazily evaluated.*

## 🚀 Quick Start

```cpp
#include "parrot.hpp"

int main() {
    // Create arrays
    auto A = parrot::array({3, 4, 0, 8, 2});
    auto B = parrot::array({6, 7, 2, 1, 8});
    auto C = parrot::array({2, 5, 7, 4, 3});
    
    // Chain operations
    (B * C + A).print();  // Output: 15 39 14 12 26
}
```

### Advanced Example: Row-wise Softmax

```cpp
// Create a matrix
auto matrix = parrot::range(10000).as<float>().reshape({100, 100});

// Calculate the row-wise softmax of a matrix
auto cols = matrix.shape()[1];
auto z    = matrix - matrix.maxr<2>().replicate(cols);
auto num  = z.exp();
auto den  = num.sum<2>();
(num / den.replicate(cols)).print();
```

## 🏗️ Building

### Prerequisites

- CMake (3.10+)
- C++ compiler with C++20 support
- NVIDIA CUDA Toolkit 13.0 or later
- NVIDIA GPU with compute capability 7.0+

### Build Steps

```bash
# Clone the repository
git clone <repository-url>
cd parrot

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
cmake --build . -j$(nproc)

# Run tests
ctest
```

For detailed build instructions, see [`BUILDING.md`](BUILDING.md).

## 📊 Comparisons

Parrot provides significant code reduction compared to other CUDA libraries:

| Library    | Code Reduction     |
| ---------- | ------------------ |
| **Thrust** | **~10x less code** |

See detailed comparisons in our [documentation](https://nvlabs.github.io/parrot/).

## 📚 Documentation

- **[Full Documentation](https://nvlabs.github.io/parrot/)** - Complete API reference and examples
- **[Examples](docs/examples.rst)** - Standalone Parrot examples
- **[vs Thrust](docs/parrot_v_thrust.rst)** - Side-by-side comparisons with Thrust


## 🧪 Examples

The [`examples/`](examples/) directory contains:

- **`getting_started/`** - Simple examples to get started
- **`thrust/`** - Parrot implementations of Thrust examples  


## 🛠️ Development

### Running Tests

```bash
# All tests
ctest

# Individual test categories
./test_basic      # Basic operations
./test_sorting    # Sorting algorithms
./test_math       # Mathematical operations
./test_reductions # Reduction operations
```

### Code Quality

```bash
# Run clang-tidy
./scripts/run-clang-tidy.sh

# Auto-fix issues
./scripts/run-clang-tidy.sh --fix
```

### Building Documentation

```bash
# Install dependencies
pip install -r requirements.txt

# Build docs
cd script && ./build-docs.sh
```

## 📁 Repository Structure

```
parrot/
├── parrot.hpp              # Main header (single-file library)
├── thrustx.hpp             # Extended Thrust utilities
├── examples/               # Example code
│   ├── getting_started/    # Simple getting-started examples
│   ├── thrust/             # Parrot versions of Thrust examples
│   ├── real_world/         # Parrot versions of examples from open source projects
├── tests/                  # Unit tests
├── docs/                   # Documentation source
├── scripts/                # Development scripts
```

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guide for details on:

- Developer Certificate of Origin (DCO) requirements
- Setting up the development environment  
- Code standards and review process
- Running tests and code quality checks
- Building documentation

All contributions must be signed off under the DCO and comply with the Apache License 2.0.

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project includes third-party software components. See [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) for complete license information and attributions.

## 🙏 Acknowledgments

Built on top of [NVIDIA Thrust](https://github.com/NVIDIA/thrust) and [CUDA](https://developer.nvidia.com/cuda-zone).

---

**[📖 Read the Full Documentation →](https://nvlabs.github.io/parrot/)**