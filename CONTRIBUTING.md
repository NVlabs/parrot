# Contributing to parrot

We welcome contributions to Parrot! This document outlines the process for contributing to the project and the requirements for submitting code.

## Table of Contents

- [Issue Tracking](#issue-tracking)
- [Getting Started](#getting-started)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
- [Contribution Process](#contribution-process)
- [Code Standards](#code-standards)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [License](#license)

## Issue Tracking

All enhancement, bugfix, or change requests must begin with the creation of a GitHub Issue.

- **Bug Reports**: Use the bug report template to describe the issue, including steps to reproduce, expected behavior, and system information.
- **Feature Requests**: Use the feature request template to describe the proposed enhancement and its use case.
- **Questions**: Use GitHub Discussions for questions about usage or implementation details.

The issue must be reviewed by parrot engineers and discussed prior to code implementation for significant changes.

## Getting Started

1. **Fork the Repository**: Create a fork of the parrot repository on your GitHub account.

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/parrot.git
   cd parrot
   ```

3. **Set Up Development Environment**: Follow the instructions in [BUILDING.md](BUILDING.md) to set up your development environment.

4. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Developer Certificate of Origin (DCO)

All contributions to parrot must be made under the Developer Certificate of Origin (DCO). The DCO is a simple statement that you, as a contributor, have the legal right to make the contribution.

### DCO Text

By contributing to this project, you certify that:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### Signing Off Commits

To indicate your acceptance of the DCO, you must sign off on each commit using the `-s` flag:

```bash
git commit -s -m "Your commit message"
```

This adds a `Signed-off-by` line to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

**Important**: All commits in your pull request must be signed off. Commits without proper sign-off will not be accepted.

### Automatic Sign-off

You can configure Git to automatically sign off commits:

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
git config format.signoff true
```

## Contribution Process

1. **Create an Issue**: For significant changes, create an issue first to discuss the proposed changes with maintainers.

2. **Make Your Changes**: 
   - Follow the [code standards](#code-standards) outlined below
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all commits are signed off with DCO

3. **Test Your Changes**:
   ```bash
   # Build the project
   mkdir build && cd build
   cmake ..
   cmake --build . -j$(nproc)
   
   # Run tests
   ctest
   
   # Run code quality checks
   cd .. && ./scripts/run-clang-tidy.sh
   ```

4. **Submit a Pull Request**:
   - Push your changes to your fork
   - Create a pull request against the main branch
   - Provide a clear description of your changes
   - Reference any related issues

5. **Code Review**: Maintainers will review your pull request and may request changes.

6. **Merge**: Once approved, your pull request will be merged by a maintainer.

## Code Standards

### Licensing

- All new source files must include the SPDX license identifier:
  ```cpp
  /*
   * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
   * All rights reserved. SPDX-License-Identifier: Apache-2.0
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   * http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  ```

### C++ Standards

- Use C++20 features and standards
- Follow modern C++ best practices
- Use `auto` for type deduction where appropriate
- Prefer `constexpr` and `const` where possible
- Use RAII principles

### Code Style and Formatting

All source code contributions must strictly adhere to the parrot coding guidelines:

- **clang-format**: Use clang-format for consistent code formatting (configuration provided in `.clang-format`)
- **Naming Conventions**: Follow existing conventions in the relevant file, submodule, module, and project
- **Comments**: Use clear, descriptive comments for complex algorithms or non-obvious code
- **Consistency**: Follow existing patterns in the codebase

#### Formatting Commands

Format git changes:
```bash
# Commit ID is optional - if unspecified, run format on staged changes
git-clang-format --style file [commit ID/reference]
```

Format individual source files:
```bash
# -style=file: Obtain formatting rules from .clang-format
# -i: In-place modification of the processed file
clang-format -style=file -i -fallback-style=none <file(s) to process>
```

Format entire codebase (for maintainers only):
```bash
find . -name "*.hpp" -o -name "*.cu" -o -name "*.h" | grep -v build | grep -v _deps \
| xargs clang-format -style=file -i -fallback-style=none
```

#### Code Quality Guidelines

- Avoid introducing unnecessary complexity into existing code to preserve maintainability and readability
- Avoid committing commented-out code
- Ensure the build log is clean (no warnings or errors)
- Use meaningful variable and function names
- Add comments for complex algorithms or non-obvious code

### CUDA Specific

- Use `__host__ __device__` annotations appropriately
- Ensure CUDA kernels are efficient and follow CUDA best practices
- Test on multiple GPU architectures when possible

### Adding or Modifying Functionality

When adding or disabling functionality:

- **CMake Options**: Add a CMake option with a default value that matches existing behavior
- **File Inclusion**: Where entire files can be included/excluded, modify `CMakeLists.txt` rather than using `#if` guards around entire file bodies
- **Minor Changes**: For minor changes to existing files, use `#if` guards appropriately
- **Backward Compatibility**: Ensure changes maintain backward compatibility unless explicitly breaking

Example CMake option:
```cmake
option(PARROT_ENABLE_FEATURE "Enable new feature" ON)
```

## Pull Request Guidelines

### PR Best Practices

Try to keep pull requests (PRs) as concise as possible:

- **Single Concern**: Each PR should address a single concern when possible. If there are several unrelated fixes needed, consider opening separate PRs and indicating dependencies in the description.
- **Clean Code**: Avoid committing commented-out code or debug statements.
- **Descriptive Titles**: Write commit titles using imperative mood and reference the Issue number.

### Commit Message Format

Follow this recommended format for commit messages:

```
#<Issue Number> - <Commit Title>

<Commit Body>

Signed-off-by: Your Name <your.email@example.com>
```

Example:
```
#123 - Add support for custom reduction operations

This commit adds support for user-defined reduction operations
in the fusion_array class, enabling more flexible data processing
workflows.

Signed-off-by: John Doe <john.doe@example.com>
```

### PR Workflow

1. **Fork and Clone**: Fork the upstream parrot repository and clone your fork
2. **Create Branch**: Create a feature branch for your changes
3. **Make Changes**: Implement your changes following the coding guidelines
4. **Test Thoroughly**: Ensure all tests pass and the build is clean
5. **Sign Commits**: All commits must be signed off with DCO
6. **Push to Fork**: Push your changes to your forked repository
7. **Create PR**: Open a pull request against the appropriate upstream branch
8. **Mark WIP**: If still working on the PR, prefix the title with `[WIP]`
9. **Address Reviews**: Respond to reviewer feedback and make necessary changes
10. **Final Review**: Wait for final approval from parrot engineers

### PR Requirements

Before submitting a PR, ensure:

- [ ] All commits are signed off (`git commit -s`)
- [ ] Code follows formatting guidelines (`clang-format` applied)
- [ ] Build is clean with no warnings or errors
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] SPDX license headers included in new files
- [ ] Related GitHub issue exists and is referenced

### Review Process

- At least one parrot engineer will be assigned for review
- PRs will only be accepted after adequate testing has been completed
- The corresponding issue will be closed when the PR is merged
- Complex changes may require multiple review cycles

## Testing

### Unit Tests

- Add unit tests for all new functionality
- Tests are located in the `tests/` directory
- Use the doctest framework for testing
- Ensure tests pass on multiple GPU architectures

### Test Categories

- `test_basic`: Basic operations and functionality
- `test_math`: Mathematical operations
- `test_reductions`: Reduction operations
- `test_scans`: Scan operations
- `test_sorting`: Sorting algorithms
- `test_array_ops`: Array manipulation operations
- `test_advanced`: Advanced operations
- `test_multidim`: Multi-dimensional operations
- `test_integration`: Integration tests

### Running Tests

```bash
# Run all tests
ctest

# Run specific test categories
./test_basic
./test_math
# etc.
```

## Documentation

All new components must contain accompanying documentation describing the functionality, dependencies, and known issues.

### Documentation Requirements

- **New Features**: All new functionality must include comprehensive documentation
- **API Documentation**: Use Doxygen-style comments for all public APIs
- **Examples**: Provide usage examples for new features or complex functionality
- **README Updates**: Update README.md for significant changes or new components
- **Known Issues**: Document any limitations or known issues

### Code Documentation Standards

- Use Doxygen-style comments for public APIs:
  ```cpp
  /**
   * @brief Brief description of the function
   * @param param1 Description of parameter 1
   * @param param2 Description of parameter 2
   * @return Description of return value
   * @throws std::exception Description of when exceptions are thrown
   */
  ```
- Document function parameters and return values
- Provide usage examples for complex functions
- Include performance considerations where relevant

### User Documentation

- Update relevant documentation in the `docs/` directory
- Add examples to the `examples/` directory for new features
- Update the README.md for user-facing changes
- Ensure documentation builds without warnings

### Building Documentation

```bash
# Install documentation dependencies
pip install -r requirements.txt

# Build documentation
cd docs
sphinx-build -b html . build/html

# Check for documentation warnings
sphinx-build -W -b html . build/html
```

### Documentation Testing

- Verify that all code examples in documentation compile and run correctly
- Test documentation builds on clean environments
- Ensure all links and references are valid

## Legal Compliance

### Contribution Rights

Make sure that you can contribute your work to open source:

- **No License Conflicts**: Ensure no license conflicts are introduced by your code
- **No Patent Conflicts**: Ensure no patent conflicts are introduced by your code  
- **Original Work**: Certify that your contribution is your original work or you have rights to submit it
- **Third-Party Code**: If including third-party code, ensure proper licensing and attribution

### Intellectual Property

- All contributions must comply with NVIDIA's intellectual property policies
- Contributors must have the legal right to submit their contributions
- Any third-party dependencies must be properly documented and licensed

## License

By contributing to parrot, you agree that your contributions will be licensed under the Apache License 2.0, the same license as the project.

### License Requirements

- All new source files must include proper SPDX license headers
- Contributions must not introduce incompatible license dependencies
- Third-party code must be properly attributed in THIRD_PARTY_LICENSES

## Questions and Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Security**: For security-related issues, please follow responsible disclosure practices

## Code of Conduct

This project follows the NVIDIA Code of Conduct. By participating, you are expected to uphold this code.

---

Thank you for contributing to Parrot! Your contributions help make GPU computing more accessible and efficient for everyone.
