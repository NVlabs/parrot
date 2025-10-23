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

# Exit on error
set -e

# Directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Navigate to project root (parent of scripts directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Default settings
FIX_MODE=0
JOBS=$(nproc)  # Default to number of CPU cores

# Help function
show_help() {
    echo "Usage: ./run-clang-tidy.sh [options] [file1] [file2] ..."
    echo "If no file is specified, all files will be checked (parrot.hpp and all test files)."
    echo
    echo "Options:"
    echo "  --fix             Automatically fix issues where possible"
    echo "  -j, --jobs N      Run N clang-tidy jobs in parallel (default: $(nproc))"
    echo "  -h, --help        Show this help message"
    echo
    echo "Examples:"
    echo "  ./run-clang-tidy.sh                        # Check all files (parallel)"
    echo "  ./run-clang-tidy.sh --fix                  # Check and fix all files (parallel)"
    echo "  ./run-clang-tidy.sh -j1                    # Run sequentially (single job)"
    echo "  ./run-clang-tidy.sh -j8 --fix              # Run 8 parallel jobs with fixes"
    echo "  ./run-clang-tidy.sh parrot.hpp             # Check only parrot.hpp"
    echo "  ./run-clang-tidy.sh --fix parrot.hpp       # Check and fix parrot.hpp"
    echo "  ./run-clang-tidy.sh tests/test_main.cu     # Check specific test file"
    exit 0
}

# Parse arguments
FILES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_MODE=1
            shift
            ;;
        -j|--jobs)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                JOBS=$2
                shift 2
            else
                echo "Error: -j/--jobs requires a positive integer"
                exit 1
            fi
            ;;
        -j*)
            # Handle -j8 style (no space)
            JOBS_VALUE=${1#-j}
            if [[ $JOBS_VALUE =~ ^[0-9]+$ ]]; then
                JOBS=$JOBS_VALUE
                shift
            else
                echo "Error: Invalid jobs value: $JOBS_VALUE"
                exit 1
            fi
            ;;
        -h|--help)
            show_help
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Check if clang-tidy is installed
if ! command -v clang-tidy &> /dev/null; then
    echo "Error: clang-tidy is not installed. Please install it first."
    echo "On Ubuntu/Debian: sudo apt-get install clang-tidy"
    echo "On macOS: brew install llvm (and make sure it's in your PATH)"
    exit 1
fi

# Find CUDA and Thrust installations
THRUST_INCLUDE_DIR=""
CUDA_PATH=""
CCCL_INCLUDE_DIR=""

# First priority: Use CCCL from build directory (has the correct Thrust version)
if [ -d "$PROJECT_ROOT/build/_deps/cccl-src" ]; then
    CCCL_INCLUDE_DIR="$PROJECT_ROOT/build/_deps/cccl-src"
    echo "Using CCCL (Thrust/CUB/libcudacxx) from build directory"
fi

# Find CUDA installation (prefer HPC SDK for CUDA runtime/stdlib)
if [ -d "/opt/nvidia/hpc_sdk" ]; then
    # Find the most recent HPC SDK version
    HPC_SDK_VERSION=$(ls -1 /opt/nvidia/hpc_sdk/Linux_x86_64/ 2>/dev/null | sort -V | tail -n1)
    if [ -n "$HPC_SDK_VERSION" ]; then
        # Check for cuda directory in HPC SDK
        if [ -d "/opt/nvidia/hpc_sdk/Linux_x86_64/$HPC_SDK_VERSION/cuda" ]; then
            CUDA_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/$HPC_SDK_VERSION/cuda"
            # Only use HPC SDK's include if we don't have CCCL
            if [ -z "$CCCL_INCLUDE_DIR" ]; then
                THRUST_INCLUDE_DIR="$CUDA_PATH/include"
            fi
            echo "Using HPC SDK CUDA at $CUDA_PATH"
        fi
    fi
fi

# Fallback to standard CUDA installation
if [ -z "$CUDA_PATH" ] && [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
    if [ -z "$CCCL_INCLUDE_DIR" ]; then
        THRUST_INCLUDE_DIR="/usr/local/cuda/include"
    fi
    echo "Using standard CUDA at $CUDA_PATH"
fi

# Last resort: try to find nvcc in PATH
if [ -z "$CUDA_PATH" ] && command -v nvcc &> /dev/null; then
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
    if [ -z "$CCCL_INCLUDE_DIR" ]; then
        THRUST_INCLUDE_DIR="$CUDA_PATH/include"
    fi
    echo "Using CUDA from PATH at $CUDA_PATH"
fi

if [ -z "$CUDA_PATH" ]; then
    echo "Warning: Could not find CUDA installation."
fi

# Find doctest in the project directory
DOCTEST_DIR=""
# First try to find doctest in build/_deps directory
if [ -d "$PROJECT_ROOT/build/_deps" ]; then
    DOCTEST_DIRS=$(find "$PROJECT_ROOT/build/_deps" -name "doctest-src" -type d 2>/dev/null)
    if [ -n "$DOCTEST_DIRS" ]; then
        # Use the first matching directory
        DOCTEST_DIR=$(echo "$DOCTEST_DIRS" | head -n 1)
        echo "Found Doctest at $DOCTEST_DIR"
    fi
fi

# If not found in build/_deps, check other common locations
if [ -z "$DOCTEST_DIR" ]; then
    if [ -f "$PROJECT_ROOT/doctest/doctest.h" ]; then
        DOCTEST_DIR="$PROJECT_ROOT"
        echo "Found Doctest at $DOCTEST_DIR/doctest"
    elif [ -f "$PROJECT_ROOT/include/doctest/doctest.h" ]; then
        DOCTEST_DIR="$PROJECT_ROOT/include"
        echo "Found Doctest at $DOCTEST_DIR"
    elif [ -f "$PROJECT_ROOT/external/doctest/doctest.h" ]; then
        DOCTEST_DIR="$PROJECT_ROOT/external"
        echo "Found Doctest at $DOCTEST_DIR"
    elif [ -f "$PROJECT_ROOT/tests/doctest/doctest.h" ]; then
        DOCTEST_DIR="$PROJECT_ROOT/tests"
        echo "Found Doctest at $DOCTEST_DIR"
    else
        echo "Warning: Could not find doctest.h. Doctest headers might not be found."
    fi
fi

# Define include paths
INCLUDE_ARGS=""
if [ -n "$PROJECT_ROOT" ]; then
    INCLUDE_ARGS="$INCLUDE_ARGS -I$PROJECT_ROOT"
fi
# Add CCCL includes (thrust, cub, libcudacxx) - these take priority
if [ -n "$CCCL_INCLUDE_DIR" ]; then
    INCLUDE_ARGS="$INCLUDE_ARGS -I$CCCL_INCLUDE_DIR/thrust"
    INCLUDE_ARGS="$INCLUDE_ARGS -I$CCCL_INCLUDE_DIR/cub"
    INCLUDE_ARGS="$INCLUDE_ARGS -I$CCCL_INCLUDE_DIR/libcudacxx/include"
fi
# Add CUDA includes (for CUDA runtime, etc.)
if [ -n "$CUDA_PATH" ]; then
    INCLUDE_ARGS="$INCLUDE_ARGS -I$CUDA_PATH/include"
fi
# Fallback to legacy THRUST_INCLUDE_DIR if no CCCL
if [ -n "$THRUST_INCLUDE_DIR" ] && [ -z "$CCCL_INCLUDE_DIR" ]; then
    INCLUDE_ARGS="$INCLUDE_ARGS -I$THRUST_INCLUDE_DIR"
fi
if [ -n "$DOCTEST_DIR" ]; then
    # Include the parent directory to support #include <doctest/doctest.h>
    INCLUDE_ARGS="$INCLUDE_ARGS -I$DOCTEST_DIR"
fi

# Get list of test files
TEST_FILES=$(find tests/ -name "*.cu" -type f 2>/dev/null | sort)

# Define compilation database
# Since we might not have CMake, we'll create a simple compilation database
cat > compile_commands.json << EOF
[
  {
    "directory": "$PROJECT_ROOT",
    "command": "clang++ -std=c++20 $INCLUDE_ARGS -x cuda --no-cuda-version-check parrot.hpp",
    "file": "parrot.hpp"
  }$(for test_file in $TEST_FILES; do
    echo ","
    echo "  {"
    echo "    \"directory\": \"$PROJECT_ROOT\","
    echo "    \"command\": \"clang++ -std=c++20 $INCLUDE_ARGS -x cuda --no-cuda-version-check $test_file\","
    echo "    \"file\": \"$test_file\""
    echo "  }"
done)
]
EOF

# Job control for parallel execution
declare -a RUNNING_JOBS=()
declare -a JOB_FILES=()

# Function to wait for a job slot to become available
wait_for_job_slot() {
    while [ ${#RUNNING_JOBS[@]} -ge $JOBS ]; do
        # Check if any jobs have completed
        local new_jobs=()
        local new_files=()
        for i in "${!RUNNING_JOBS[@]}"; do
            local pid=${RUNNING_JOBS[$i]}
            local file=${JOB_FILES[$i]}
            if ! kill -0 $pid 2>/dev/null; then
                # Job completed, wait for it to collect exit status
                wait $pid
                local exit_code=$?
                if [ $exit_code -ne 0 ] && [ $exit_code -ne 1 ]; then
                    echo "Warning: clang-tidy failed for $file with exit code $exit_code"
                fi
                echo "Completed: $file"
            else
                # Job still running
                new_jobs+=($pid)
                new_files+=("$file")
            fi
        done
        RUNNING_JOBS=("${new_jobs[@]}")
        JOB_FILES=("${new_files[@]}")
        
        # Small sleep to avoid busy waiting
        if [ ${#RUNNING_JOBS[@]} -ge $JOBS ]; then
            sleep 0.1
        fi
    done
}

# Function to wait for all remaining jobs to complete
wait_for_all_jobs() {
    echo "Waiting for remaining jobs to complete..."
    for pid in "${RUNNING_JOBS[@]}"; do
        wait $pid
        local exit_code=$?
        if [ $exit_code -ne 0 ] && [ $exit_code -ne 1 ]; then
            echo "Warning: clang-tidy job failed with exit code $exit_code"
        fi
    done
    RUNNING_JOBS=()
    JOB_FILES=()
}

# Function to run clang-tidy on a file
run_clang_tidy() {
    file=$1
    echo "Running clang-tidy on $file..."
    
    # Determine line count
    line_count=$(wc -l < "$file")
    
    # Build extra args string
    EXTRA_ARGS_STRING=""
    
    # Add include paths to extra args
    for include_path in $INCLUDE_ARGS; do
        EXTRA_ARGS_STRING="$EXTRA_ARGS_STRING -extra-arg=$include_path"
    done
    
    # Add CUDA-specific flags to avoid false positive errors
    EXTRA_ARGS_STRING="$EXTRA_ARGS_STRING -extra-arg=--no-cuda-version-check"
    EXTRA_ARGS_STRING="$EXTRA_ARGS_STRING -extra-arg=-Xclang"
    EXTRA_ARGS_STRING="$EXTRA_ARGS_STRING -extra-arg=-fcuda-allow-variadic-functions"
    EXTRA_ARGS_STRING="$EXTRA_ARGS_STRING -extra-arg=-Wno-unknown-cuda-version"
    
    # Add fix mode if enabled
    FIX_ARGS=""
    if [ $FIX_MODE -eq 1 ]; then
        echo "Auto-fix mode enabled"
        FIX_ARGS="--fix"
    fi
    
    # Run clang-tidy, excluding system, CCCL, and doctest headers from analysis
    # Only analyze headers in the project root
    clang-tidy -config-file="$PROJECT_ROOT/.clang-tidy-cuda" \
               -header-filter="^$PROJECT_ROOT/(?!build/).*\.(hpp|h|cuh)$" \
               -line-filter="[{'name':'$file','lines':[[1,$line_count]]}]" \
               $EXTRA_ARGS_STRING \
               $FIX_ARGS \
               "$file"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 1 ]; then
        echo "Error: clang-tidy failed with exit code $exit_code"
        return $exit_code
    fi
    
    if [ $FIX_MODE -eq 1 ]; then
        echo "Fixes applied to $file (if any were possible)"
    fi
    
    echo "Clang-tidy completed for $file."
    return 0
}

# Function to start a clang-tidy job in the background
start_clang_tidy_job() {
    local file=$1
    echo "Starting clang-tidy on $file..."
    
    # Run clang-tidy in background and capture PID
    run_clang_tidy "$file" &
    local pid=$!
    
    # Add to tracking arrays
    RUNNING_JOBS+=($pid)
    JOB_FILES+=("$file")
    
    return 0
}

# Prepare list of all files to process
ALL_FILES=()
if [ ${#FILES[@]} -eq 0 ]; then
    # If no files specified, run on all files
    ALL_FILES+=("parrot.hpp")
    for test_file in $TEST_FILES; do
        ALL_FILES+=("$test_file")
    done
else
    # Run on specified files
    ALL_FILES=("${FILES[@]}")
fi

echo "Running clang-tidy with $JOBS parallel jobs on ${#ALL_FILES[@]} files..."

# Process all files with parallel execution
if [ $JOBS -eq 1 ]; then
    # Sequential execution for single job
    for file in "${ALL_FILES[@]}"; do
        run_clang_tidy "$file"
        echo
    done
else
    # Parallel execution
    for file in "${ALL_FILES[@]}"; do
        # Wait for a job slot to become available
        wait_for_job_slot
        
        # Start the job
        start_clang_tidy_job "$file"
    done
    
    # Wait for all remaining jobs to complete
    wait_for_all_jobs
fi

# Cleanup
rm -f compile_commands.json

echo "All clang-tidy checks completed." 