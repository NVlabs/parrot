#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Script to compile and run all examples in the examples/ folder
# (excluding real_world examples as requested)
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # Go up one level from scripts/
EXAMPLES_DIR="$PROJECT_ROOT/examples"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUTS_DIR="$SCRIPT_DIR/expected_outputs"  # Expected outputs stay in scripts/

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_EXAMPLES=0
SUCCESSFUL_COMPILES=0
SUCCESSFUL_RUNS=0
FAILED_COMPILES=0
FAILED_RUNS=0
FAILED_COMPARISONS=0
RAND_EXAMPLES=0  # Count of examples using .rand()

# Script modes
MODE="run"  # Default mode: run examples
CAPTURE_MODE=false
TEST_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --capture)
            MODE="capture"
            CAPTURE_MODE=true
            shift
            ;;
        --test)
            MODE="test"
            TEST_MODE=true
            shift
            ;;
        --run)
            MODE="run"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Compile and run all Parrot examples (excluding real_world)"
            echo ""
            echo "Options:"
            echo "  --run      Run examples normally (default)"
            echo "  --capture  Run examples and capture their output as expected results"
            echo "  --test     Run examples and compare output against saved expected results"
            echo "  -h, --help Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Run all examples"
            echo "  $0 --capture      # Run examples and save outputs"
            echo "  $0 --test         # Run examples and test against saved outputs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 Parrot Examples Runner${NC}"
case $MODE in
    "capture")
        echo -e "${YELLOW}📸 CAPTURE MODE: Saving example outputs${NC}"
        ;;
    "test")
        echo -e "${BLUE}🧪 TEST MODE: Comparing against expected outputs${NC}"
        ;;
    "run")
        echo -e "${GREEN}▶️  RUN MODE: Running examples${NC}"
        ;;
esac
echo "=================================="

# Function to print status
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "error")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "info")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
    esac
}

# Function to check if an example uses .rand()
uses_rand() {
    local example_file=$1
    if grep -q "\.rand()" "$example_file"; then
        return 0  # Uses rand
    else
        return 1  # Does not use rand
    fi
}

# Function to ensure project is built with dependencies
ensure_project_built() {
    if [ ! -d "$BUILD_DIR" ]; then
        print_status "info" "Build directory not found. Building project to fetch dependencies..."
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        
        # Configure with CMake
        if ! cmake .. >/dev/null 2>&1; then
            print_status "error" "Failed to configure project with CMake"
            return 1
        fi
        
        # Build just enough to get dependencies (no need for full build)
        if ! cmake --build . --target doctest >/dev/null 2>&1; then
            print_status "warning" "Could not build doctest target, but continuing..."
        fi
        
        cd "$PROJECT_ROOT"
        print_status "success" "Project configured and dependencies fetched"
    else
        print_status "info" "Build directory exists, assuming dependencies are available"
    fi
    
    return 0
}

# Function to get CUDA compiler flags based on CMakeLists.txt
get_cuda_flags() {
    local flags=""
    
    # Basic flags from CMakeLists.txt (matching the build output)
    flags="--extended-lambda -std=c++20 -rdc=true"
    
    # Include directories - prioritize CCCL if available
    flags="$flags -I$PROJECT_ROOT"
    
    # Check if CCCL was downloaded by CMake
    local cccl_dir="$BUILD_DIR/_deps/cccl-src"
    if [ -d "$cccl_dir" ]; then
        flags="$flags -isystem $cccl_dir"
        flags="$flags -isystem $cccl_dir/cub"
        flags="$flags -isystem $cccl_dir/thrust"
        flags="$flags -isystem $cccl_dir/libcudacxx/include"
    fi
    
    # Try to detect GPU architecture (matching CMake output shows sm_89)
    if command -v nvidia-smi &> /dev/null; then
        local arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')
        if [[ ! -z "$arch" ]]; then
            flags="$flags --generate-code=arch=compute_$arch,code=[compute_$arch,sm_$arch]"
        fi
    else
        # Fallback to sm_89 as shown in CMake output
        flags="$flags --generate-code=arch=compute_89,code=[compute_89,sm_89]"
    fi
    
    echo "$flags"
}

# Function to compile and run a single example
compile_and_run_example() {
    local example_file=$1
    local example_name=$(basename "$example_file" .cu)
    local example_dir=$(dirname "$example_file")
    local relative_path=${example_file#$EXAMPLES_DIR/}
    
    echo ""
    echo -e "${BLUE}📁 Processing: $relative_path${NC}"
    echo "----------------------------------------"
    
    TOTAL_EXAMPLES=$((TOTAL_EXAMPLES + 1))
    
    # Create temporary build directory for this example
    local temp_build_dir="/tmp/parrot_example_$example_name"
    mkdir -p "$temp_build_dir"
    
    # Get CUDA flags
    local cuda_flags=$(get_cuda_flags)
    
    # Print status messages after getting flags (only once per run)
    if [ $TOTAL_EXAMPLES -eq 1 ]; then
        local cccl_dir="$BUILD_DIR/_deps/cccl-src"
        if [ -d "$cccl_dir" ]; then
            print_status "info" "Using CCCL headers from: $cccl_dir"
        else
            print_status "warning" "CCCL headers not found, may cause compilation issues"
        fi
        
        if command -v nvidia-smi &> /dev/null; then
            local arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')
            if [[ ! -z "$arch" ]]; then
                print_status "info" "Auto-detected GPU architecture: sm_$arch"
            fi
        else
            print_status "warning" "Could not detect GPU architecture, using sm_89"
        fi
    fi
    
    # Compile the example
    print_status "info" "Compiling $example_name..."
    
    # Use ccache if available (like the main project does)
    local nvcc_cmd="$NVCC_PATH"
    if command -v ccache &> /dev/null; then
        nvcc_cmd="ccache $NVCC_PATH"
    fi
    
    local compile_cmd="$nvcc_cmd $cuda_flags -o $temp_build_dir/$example_name $example_file"
    
    if eval "$compile_cmd" 2>/dev/null; then
        print_status "success" "Compilation successful"
        SUCCESSFUL_COMPILES=$((SUCCESSFUL_COMPILES + 1))
        
        # Run the example and handle different modes
        print_status "info" "Running $example_name..."
        local output_file=""
        local expected_file=""
        
        if [ "$CAPTURE_MODE" = true ] || [ "$TEST_MODE" = true ]; then
            # Create expected outputs directory structure
            local output_subdir="$OUTPUTS_DIR/$(dirname "$relative_path")"
            mkdir -p "$output_subdir"
            expected_file="$output_subdir/$example_name.out"
        fi
        
        # Run the example and capture output
        local run_output=""
        local run_exit_code=0
        
        if run_output=$(timeout 30s "$temp_build_dir/$example_name" 2>&1); then
            run_exit_code=0
        else
            run_exit_code=$?
        fi
        
        if [ $run_exit_code -eq 0 ]; then
            if [ "$CAPTURE_MODE" = true ]; then
                # Save output to expected results file
                echo "$run_output" > "$expected_file"
                print_status "success" "Execution successful - Output saved to $expected_file"
                SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
            elif [ "$TEST_MODE" = true ]; then
                # Check if this example uses .rand()
                if uses_rand "$example_file"; then
                    # Handle rand examples specially
                    echo -e "${YELLOW}🟡 Output differs because of .rand() usage in example${NC}"
                    SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                    RAND_EXAMPLES=$((RAND_EXAMPLES + 1))
                else
                    # Compare output against expected results for deterministic examples
                    if [ -f "$expected_file" ]; then
                        local expected_output=$(cat "$expected_file")
                        if [ "$run_output" = "$expected_output" ]; then
                            print_status "success" "Execution successful - Output matches expected result"
                            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                        else
                            print_status "error" "Output differs from expected result"
                            echo -e "${RED}Expected:${NC}"
                            echo "$expected_output" | sed 's/^/  /'
                            echo -e "${RED}Actual:${NC}"
                            echo "$run_output" | sed 's/^/  /'
                            FAILED_COMPARISONS=$((FAILED_COMPARISONS + 1))
                            FAILED_RUNS=$((FAILED_RUNS + 1))
                        fi
                    else
                        print_status "warning" "No expected output file found: $expected_file"
                        print_status "info" "Run with --capture first to create expected outputs"
                        echo -e "${BLUE}Actual output:${NC}"
                        echo "$run_output" | sed 's/^/  /'
                        FAILED_RUNS=$((FAILED_RUNS + 1))
                    fi
                fi
            else
                # Normal run mode - just show output
                print_status "success" "Execution successful"
                echo "$run_output"
                SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
            fi
        else
            if [ $run_exit_code -eq 124 ]; then
                print_status "error" "Execution timed out (30s limit)"
            else
                print_status "error" "Execution failed (exit code: $run_exit_code)"
            fi
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
    else
        print_status "error" "Compilation failed"
        FAILED_COMPILES=$((FAILED_COMPILES + 1))
        
        # Show compilation error for debugging
        echo -e "${RED}Compilation error details:${NC}"
        eval "$compile_cmd"
    fi
    
    # Clean up
    rm -rf "$temp_build_dir"
}

# Function to process examples in a directory
process_directory() {
    local dir=$1
    local dir_name=$(basename "$dir")
    
    echo ""
    echo -e "${YELLOW}📂 Processing directory: $dir_name${NC}"
    echo "========================================"
    
    # Find all .cu files in the directory
    local cu_files=($(find "$dir" -name "*.cu" -type f))
    
    if [ ${#cu_files[@]} -eq 0 ]; then
        print_status "warning" "No .cu files found in $dir_name"
        return
    fi
    
    # Process each .cu file
    for cu_file in "${cu_files[@]}"; do
        compile_and_run_example "$cu_file"
    done
}

# Function to find nvcc
find_nvcc() {
    # First check if nvcc is in PATH
    if command -v nvcc &> /dev/null; then
        echo "nvcc"
        return 0
    fi
    
    # Check the same nvcc that CMake found (from the build output)
    local cmake_nvcc="/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvcc"
    if [[ -x "$cmake_nvcc" ]]; then
        echo "$cmake_nvcc"
        return 0
    fi
    
    # Search in common CUDA installation locations
    local search_dirs=(
        "/usr/local/cuda/bin"
        "/usr/local/cuda-*/bin"
        "/opt/cuda/bin"
        "/opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin"
    )
    
    for dir_pattern in "${search_dirs[@]}"; do
        # Use find for patterns with wildcards
        if [[ "$dir_pattern" == *"*"* ]]; then
            local found_nvcc=$(find /usr/local -name "nvcc" -type f 2>/dev/null | head -1)
            if [[ -n "$found_nvcc" && -x "$found_nvcc" ]]; then
                echo "$found_nvcc"
                return 0
            fi
        else
            # Direct path check
            local nvcc_path="$dir_pattern/nvcc"
            if [[ -x "$nvcc_path" ]]; then
                echo "$nvcc_path"
                return 0
            fi
        fi
    done
    
    return 1
}

# Check prerequisites
print_status "info" "Checking prerequisites..."

NVCC_PATH=$(find_nvcc)
if [[ -z "$NVCC_PATH" ]]; then
    print_status "error" "nvcc (CUDA compiler) not found in PATH or common locations"
    print_status "info" "Searched locations:"
    print_status "info" "  - PATH"
    print_status "info" "  - /usr/local/cuda/bin/"
    print_status "info" "  - /usr/local/cuda-*/bin/"
    print_status "info" "  - /opt/cuda/bin/"
    print_status "info" "  - /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin/"
    exit 1
fi

print_status "success" "Found nvcc at: $NVCC_PATH"

if ! command -v nvidia-smi &> /dev/null; then
    print_status "warning" "nvidia-smi not found - GPU detection may be limited"
fi

# Check if parrot.hpp exists
if [ ! -f "$PROJECT_ROOT/parrot.hpp" ]; then
    print_status "error" "parrot.hpp not found in project root"
    exit 1
fi

print_status "success" "Prerequisites check passed"

# Ensure project is built and dependencies are available
if ! ensure_project_built; then
    print_status "error" "Failed to build project and fetch dependencies"
    exit 1
fi

# Process each examples subdirectory (excluding real_world)
cd "$EXAMPLES_DIR"

for dir in */; do
    dir_path="$EXAMPLES_DIR/$dir"
    dir_name=$(basename "$dir_path")
    
    # Skip real_world directory as requested
    if [ "$dir_name" = "real_world" ]; then
        print_status "info" "Skipping real_world directory as requested"
        continue
    fi
    
    if [ -d "$dir_path" ]; then
        process_directory "$dir_path"
    fi
done

# Print final summary
echo ""
echo -e "${BLUE}📊 Final Summary${NC}"
echo "=================================="
echo -e "Total examples processed: ${BLUE}$TOTAL_EXAMPLES${NC}"
echo -e "Successful compilations:  ${GREEN}$SUCCESSFUL_COMPILES${NC}"
echo -e "Failed compilations:      ${RED}$FAILED_COMPILES${NC}"
echo -e "Successful runs:          ${GREEN}$SUCCESSFUL_RUNS${NC}"
echo -e "Failed runs:              ${RED}$FAILED_RUNS${NC}"

if [ "$TEST_MODE" = true ]; then
    echo -e "Failed comparisons:       ${RED}$FAILED_COMPARISONS${NC}"
    echo -e "Random examples:          ${YELLOW}$RAND_EXAMPLES${NC}"
fi

# Calculate success rates
if [ $TOTAL_EXAMPLES -gt 0 ]; then
    compile_rate=$((SUCCESSFUL_COMPILES * 100 / TOTAL_EXAMPLES))
    run_rate=$((SUCCESSFUL_RUNS * 100 / TOTAL_EXAMPLES))
    echo -e "Compilation success rate: ${GREEN}$compile_rate%${NC}"
    echo -e "Execution success rate:   ${GREEN}$run_rate%${NC}"
fi

echo ""
if [ "$CAPTURE_MODE" = true ]; then
    if [ $FAILED_COMPILES -eq 0 ] && [ $FAILED_RUNS -eq 0 ]; then
        print_status "success" "All examples compiled and outputs captured successfully! 📸"
        print_status "info" "Expected outputs saved to: $OUTPUTS_DIR"
        exit 0
    else
        print_status "error" "Some examples failed during capture"
        exit 1
    fi
elif [ "$TEST_MODE" = true ]; then
    if [ $FAILED_COMPILES -eq 0 ] && [ $FAILED_COMPARISONS -eq 0 ]; then
        print_status "success" "All examples passed regression tests! 🧪✅"
        if [ $RAND_EXAMPLES -gt 0 ]; then
            print_status "info" "$RAND_EXAMPLES examples use .rand() and were skipped for comparison"
        fi
        exit 0
    else
        print_status "error" "Some examples failed regression tests"
        exit 1
    fi
else
    if [ $FAILED_COMPILES -eq 0 ] && [ $FAILED_RUNS -eq 0 ]; then
        print_status "success" "All examples compiled and ran successfully! 🎉"
        exit 0
    elif [ $FAILED_COMPILES -eq 0 ]; then
        print_status "warning" "All examples compiled successfully, but some failed to run"
        exit 1
    else
        print_status "error" "Some examples failed to compile or run"
        exit 1
    fi
fi
