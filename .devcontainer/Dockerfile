# NVIDIA Parrot Development Container
# Based on NVIDIA CUDA development image with Ubuntu

ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create a non-root user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    ccache \
    pkg-config \
    # Version control
    git \
    git-lfs \
    # Python and documentation tools
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Utilities
    curl \
    wget \
    unzip \
    zip \
    tree \
    htop \
    vim \
    nano \
    bc \
    # Development tools
    gdb \
    valgrind \
    clang-tidy \
    clang-format \
    # Documentation tools
    doxygen \
    graphviz \
    # Network tools
    openssh-client \
    # Repository management
    software-properties-common \
    lsb-release \
    gnupg \
    # User management
    sudo \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install modern CMake (3.23+ required for CUDA C++20 support)
# First, try from Kitware repository for latest version
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | \
    tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* || \
    # If Kitware fails, install from official CMake releases
    (cd /tmp && \
     wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh && \
     sh cmake-3.28.3-linux-x86_64.sh --prefix=/usr/local --skip-license && \
     rm cmake-3.28.3-linux-x86_64.sh)

# Create workspace directory and set permissions
RUN mkdir -p /workspace

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace

# Switch to the developer user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install Python packages for documentation
RUN python3 -m pip install --user --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY --chown=$USERNAME:$USERNAME requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --user -r /tmp/requirements.txt

# Set up ccache for faster builds
RUN mkdir -p /home/$USERNAME/.ccache && \
    echo "max_size = 5.0G" > /home/$USERNAME/.ccache/ccache.conf && \
    echo "compression = true" >> /home/$USERNAME/.ccache/ccache.conf && \
    echo "compression_level = 6" >> /home/$USERNAME/.ccache/ccache.conf

# Configure git (users can override these)
RUN git config --global init.defaultBranch main && \
    git config --global pull.rebase false && \
    git config --global user.name "Developer" && \
    git config --global user.email "developer@example.com"

# Set up shell environment
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/$USERNAME/.bashrc && \
    echo 'export CUDA_HOME=/usr/local/cuda' >> /home/$USERNAME/.bashrc && \
    echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> /home/$USERNAME/.bashrc && \
    echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"' >> /home/$USERNAME/.bashrc && \
    echo 'export CCACHE_DIR="$HOME/.ccache"' >> /home/$USERNAME/.bashrc && \
    echo 'alias ll="ls -alF"' >> /home/$USERNAME/.bashrc && \
    echo 'alias la="ls -A"' >> /home/$USERNAME/.bashrc && \
    echo 'alias l="ls -CF"' >> /home/$USERNAME/.bashrc

# Set workspace as working directory
WORKDIR /workspace

# Verify CUDA installation
RUN nvcc --version && nvidia-smi || echo "GPU not available during build"

# Set the default command
CMD ["/bin/bash"]

# Labels for better container management
LABEL maintainer="NVIDIA Parrot Development Team"
LABEL description="Development container for NVIDIA Parrot with CUDA support"
LABEL version="1.0"
LABEL cuda.version="${CUDA_VERSION}"
LABEL ubuntu.version="${UBUNTU_VERSION}"
