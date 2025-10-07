#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Build documentation for Parrot

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

QUICK=0

# Parse args
for arg in "$@"; do
    case "$arg" in
        --quick)
            QUICK=1
            ;;
        -q)
            QUICK=1
            ;;
        *)
            ;;
    esac
done

if [ "$QUICK" -eq 1 ]; then
    echo "Building Parrot documentation (quick mode)..."
else
    echo "Building Parrot documentation..."
fi

# Change to project root
cd "$PROJECT_ROOT"

# Check if clang-format is available
if ! command -v clang-format &> /dev/null; then
    echo "Note: clang-format not found. Code formatting will be skipped."
    echo "To install clang-format:"
    echo "  Ubuntu/Debian: sudo apt-get install clang-format"
    echo "  macOS: brew install clang-format"
    echo ""
fi

# Generate comparison documentation (unless quick)
if [ "$QUICK" -eq 1 ]; then
    echo "Skipping example/comparison generation (--quick)."
else
    echo "Generating comparison documentation..."
    uv run docs/scripts/generate_comparisons.py
fi

# Check if uv is available and build with Sphinx
if command -v uv &> /dev/null; then
    echo "Building Sphinx documentation..."
    cd docs
    uv run make html
    echo "Documentation built successfully in docs/_build/html/"
else
    echo "uv not found. Please install uv or run: pip install -r requirements.txt && cd docs && make html"
    echo "Documentation comparison files generated successfully."
fi

echo "Documentation build complete!"
