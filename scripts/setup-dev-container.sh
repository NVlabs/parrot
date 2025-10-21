#!/bin/bash

# NVIDIA Parrot Development Container Setup for Beginners
# This script helps beginners set up the development environment

set -e

# Colors for output - only use if terminal supports colors
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && tput colors >/dev/null 2>&1 && [[ $(tput colors) -ge 8 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    # No color support or output is redirected
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    PURPLE=''
    CYAN=''
    NC=''
fi

# Function to print colored output
print_header() {
    echo -e "\n${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}\n"
}

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

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Welcome message
print_header "🦜 NVIDIA Parrot Development Container Setup"
echo "This script will help you set up a complete development environment for NVIDIA Parrot."
echo "The environment includes CUDA, CMake, and all necessary tools in a Docker container."
echo ""

# Check prerequisites
print_header "📋 Checking Prerequisites"

# Check if Docker is installed
print_step "Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    print_success "Docker found - version: $DOCKER_VERSION"
else
    print_error "Docker is not installed!"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if user has Docker permissions
print_step "Checking Docker permissions..."
if docker ps &> /dev/null; then
    print_success "Docker is accessible"
else
    print_error "Cannot access Docker daemon!"
    echo ""
    echo "Your user is not in the 'docker' group. To fix this, run:"
    echo ""
    echo -e "  ${CYAN}sudo usermod -aG docker \$USER${NC}"
    echo -e "  ${CYAN}newgrp docker${NC}"
    echo ""
    echo "Or log out and log back in for the group change to take effect."
    echo ""
    echo "Then run this script again (without sudo)."
    exit 1
fi

# Check if Docker Compose is available
print_step "Checking Docker Compose..."
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short)
    print_success "Docker Compose found - version: $COMPOSE_VERSION"
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
    print_success "Docker Compose found - version: $COMPOSE_VERSION"
    COMPOSE_CMD="docker-compose"
else
    print_error "Docker Compose is not installed!"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
print_step "Checking NVIDIA Docker support..."
if docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_success "NVIDIA Docker runtime is working"
elif docker run --rm --runtime=nvidia nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_success "NVIDIA Docker runtime is working (legacy mode)"
else
    print_warning "NVIDIA Docker runtime may not be properly configured"
    echo "You may need to install nvidia-docker2 or nvidia-container-toolkit"
    echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if we're in the right directory and navigate to project root
print_step "Verifying project directory..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ ! -f "parrot.hpp" ] || [ ! -f "CMakeLists.txt" ]; then
    print_error "Cannot find parrot project files!"
    echo "Expected to find parrot.hpp and CMakeLists.txt in: $PROJECT_ROOT"
    exit 1
fi
print_success "Found Parrot project files in: $PROJECT_ROOT"

# Setup options
print_header "🛠️ Setup Options"
echo "Choose your development environment:"
echo "1. VS Code/Cursor Dev Container (recommended for VS Code or Cursor users)"
echo "2. Standalone Docker Container (for terminal/other editors)"
echo "3. Both (complete setup)"
echo ""
read -p "Enter your choice (1-3): " -n 1 -r SETUP_CHOICE
echo ""

case $SETUP_CHOICE in
    1)
        SETUP_VSCODE=true
        SETUP_STANDALONE=false
        ;;
    2)
        SETUP_VSCODE=false
        SETUP_STANDALONE=true
        ;;
    3)
        SETUP_VSCODE=true
        SETUP_STANDALONE=true
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

# VS Code/Cursor Dev Container setup
if [ "$SETUP_VSCODE" = true ]; then
    print_header "🔧 Setting up VS Code/Cursor Dev Container"
    
    # Check if VS Code or Cursor is installed
    EDITOR_FOUND=false
    if command -v code &> /dev/null; then
        print_success "VS Code found"
        EDITOR_FOUND=true
        
        # Check for Dev Containers extension
        print_step "Checking for Dev Containers extension..."
        if code --list-extensions | grep -q "ms-vscode-remote.remote-containers"; then
            print_success "Dev Containers extension is installed"
        else
            print_warning "Dev Containers extension not found"
            echo "Installing Dev Containers extension..."
            code --install-extension ms-vscode-remote.remote-containers
            print_success "Dev Containers extension installed"
        fi
    fi
    
    if command -v cursor &> /dev/null; then
        print_success "Cursor found"
        EDITOR_FOUND=true
        
        # Check for Dev Containers extension in Cursor
        print_step "Checking for Dev Containers extension in Cursor..."
        if cursor --list-extensions | grep -q "ms-vscode-remote.remote-containers"; then
            print_success "Dev Containers extension is installed in Cursor"
        else
            print_warning "Dev Containers extension not found in Cursor"
            echo "Installing Dev Containers extension in Cursor..."
            cursor --install-extension ms-vscode-remote.remote-containers
            print_success "Dev Containers extension installed in Cursor"
        fi
    fi
    
    if [ "$EDITOR_FOUND" = false ]; then
        print_warning "Neither VS Code nor Cursor found in PATH"
        echo "Please install VS Code or Cursor and the Dev Containers extension"
        echo "VS Code: https://code.visualstudio.com/"
        echo "Cursor: https://cursor.sh/"
        echo "Extension: ms-vscode-remote.remote-containers"
    fi
    
    print_step "Building dev container image..."
    $COMPOSE_CMD -f .devcontainer/docker-compose.yml build
    print_success "Dev container image built successfully"
fi

# Standalone Docker setup
if [ "$SETUP_STANDALONE" = true ]; then
    print_header "🐳 Setting up Standalone Docker Container"
    
    print_step "Building Docker image..."
    $COMPOSE_CMD build
    print_success "Docker image built successfully"
    
    print_step "Creating and starting container..."
    $COMPOSE_CMD up -d parrot-dev
    print_success "Container started successfully"
    
    # Test the container
    print_step "Testing container..."
    if $COMPOSE_CMD exec parrot-dev nvcc --version &> /dev/null; then
        print_success "Container is working correctly"
    else
        print_warning "Container may have issues - check logs with: $COMPOSE_CMD logs parrot-dev"
    fi
fi

# Final instructions
print_header "🎉 Setup Complete!"

if [ "$SETUP_VSCODE" = true ]; then
    echo -e "${GREEN}VS Code/Cursor Dev Container:${NC}"
    echo -e "1. Open in your editor:"
    echo -e "   • VS Code: ${CYAN}code .${NC}"
    echo -e "   • Cursor: ${CYAN}cursor .${NC}"
    echo "2. When prompted, click 'Reopen in Container'"
    echo "3. Or use Command Palette (Ctrl+Shift+P): 'Dev Containers: Reopen in Container'"
    echo ""
fi

if [ "$SETUP_STANDALONE" = true ]; then
    echo -e "${GREEN}Standalone Docker Container:${NC}"
    echo -e "• Start development: ${CYAN}$COMPOSE_CMD exec parrot-dev bash${NC}"
    echo -e "• Stop container: ${CYAN}$COMPOSE_CMD down${NC}"
    echo -e "• View logs: ${CYAN}$COMPOSE_CMD logs parrot-dev${NC}"
    echo -e "• Rebuild: ${CYAN}$COMPOSE_CMD build${NC}"
    echo ""
fi

echo -e "${GREEN}Quick Start Commands (inside container):${NC}"
echo -e "• Build project: ${CYAN}cd /workspace && mkdir build && cd build && cmake .. && make -j\$(nproc)${NC}"
echo -e "• Run tests: ${CYAN}ctest${NC}"
echo -e "• Try example: ${CYAN}cd /workspace/examples/getting_started && nvcc -I../.. sum_of_squares.cu -o sum_of_squares && ./sum_of_squares${NC}"
echo ""

echo -e "${GREEN}Documentation:${NC}"
echo -e "• Build docs: ${CYAN}cd /workspace/docs && make html${NC}"
echo -e "• View docs: Open ${CYAN}docs/build/html/index.html${NC} in browser"
echo ""

echo -e "${GREEN}Useful Files:${NC}"
echo -e "• ${CYAN}BUILDING.md${NC} - Detailed build instructions"
echo -e "• ${CYAN}README.md${NC} - Project overview"
echo -e "• ${CYAN}examples/${NC} - Example code"
echo ""

print_success "NVIDIA Parrot development environment is ready! 🦜"
echo "Happy coding! 🚀"
