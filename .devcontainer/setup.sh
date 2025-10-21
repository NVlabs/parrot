#!/bin/bash

# NVIDIA Parrot Development Container Setup Script
# This script runs after the dev container is created

set -e

echo "🦜 Setting up NVIDIA Parrot development environment..."

# Colors for output - only use if terminal supports colors
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && tput colors >/dev/null 2>&1 && [[ $(tput colors) -ge 8 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    # No color support or output is redirected
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verify CUDA installation
print_status "Verifying CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_success "NVCC found - CUDA version: $CUDA_VERSION"
    
    # Check if CUDA version meets requirements
    if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l) -eq 1 ]]; then
        print_success "CUDA version meets requirements (>= 12.0)"
    else
        print_warning "CUDA version $CUDA_VERSION may not meet requirements (>= 12.0)"
    fi
else
    print_error "NVCC not found in PATH"
fi

# Check GPU availability
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $GPU_INFO"
    else
        print_warning "nvidia-smi command failed - GPU may not be available"
    fi
else
    print_warning "nvidia-smi not found"
fi

# Verify CMake version
print_status "Verifying CMake installation..."
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | sed 's/cmake version //')
    print_success "CMake found - version: $CMAKE_VERSION"
else
    print_error "CMake not found"
fi

# Set up ccache
print_status "Configuring ccache..."
if command -v ccache &> /dev/null; then
    ccache --set-config=max_size=5G
    ccache --set-config=compression=true
    ccache --set-config=compression_level=6
    print_success "ccache configured with 5GB cache size"
else
    print_warning "ccache not found"
fi

# Create build directory
print_status "Creating build directory..."
if [ ! -d "/workspace/build" ]; then
    mkdir -p /workspace/build
    print_success "Build directory created"
else
    print_status "Build directory already exists"
fi

# Install additional Python packages if needed
print_status "Checking Python environment..."
python3 -c "import sphinx, breathe, furo" 2>/dev/null && \
    print_success "Python documentation packages available" || \
    print_warning "Some Python packages may be missing"

# Set up git hooks (optional)
print_status "Setting up git configuration..."
cd /workspace
if [ -d ".git" ]; then
    # Set up some useful git aliases
    git config --global alias.st status
    git config --global alias.co checkout
    git config --global alias.br branch
    git config --global alias.ci commit
    git config --global alias.unstage 'reset HEAD --'
    git config --global alias.last 'log -1 HEAD'
    git config --global alias.visual '!gitk'
    print_success "Git aliases configured"
else
    print_status "Not in a git repository - skipping git setup"
fi

# Display system information
print_status "System Information:"
echo "  OS: $(lsb_release -d | cut -f2)"
echo "  Kernel: $(uname -r)"
echo "  Architecture: $(uname -m)"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"

# Display build instructions
echo ""
echo -e "🚀 ${GREEN}Setup complete!${NC} Here's how to get started:"
echo ""
echo -e "1. ${BLUE}Build Parrot:${NC}"
echo "   cd /workspace"
echo "   mkdir -p build && cd build"
echo "   cmake .."
echo "   cmake --build . -j\$(nproc)"
echo ""
echo -e "2. ${BLUE}Run tests:${NC}"
echo "   ctest"
echo "   # or run individual tests:"
echo "   ./test_basic"
echo ""
echo -e "3. ${BLUE}Build documentation:${NC}"
echo "   cd /workspace/docs"
echo "   make html"
echo ""
echo -e "4. ${BLUE}Try examples:${NC}"
echo "   cd /workspace/examples/getting_started"
echo "   nvcc -I../.. sum_of_squares.cu -o sum_of_squares"
echo "   ./sum_of_squares"
echo ""
echo "📚 For more information, see:"
echo "   - BUILDING.md for detailed build instructions"
echo "   - README.md for project overview"
echo "   - docs/ for full documentation"
echo ""

print_success "NVIDIA Parrot development environment is ready! 🦜"
