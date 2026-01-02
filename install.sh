#!/bin/bash
# install.sh - Auto-detect CUDA and install appropriate dependencies

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the uv environment to make it available in this session
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        echo "Error: Failed to install uv. Please install it manually from https://github.com/astral-sh/uv"
        exit 1
    fi
    echo "uv installed successfully."
else
    echo "uv is already installed."
fi

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt | grep -oP '\d+\.\d+')
elif command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
fi

CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

echo "Detected CUDA version: $CUDA_VERSION"

# Map to PyTorch CUDA version
if [[ "$CUDA_MAJOR" -ge 13 ]]; then
    EXTRA="cu130"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 9 ]]; then
    EXTRA="cu129"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 8 ]]; then
    EXTRA="cu128"
else
    echo "Warning: CUDA $CUDA_VERSION may not be fully compatible. Using cu128."
    EXTRA="cu128"
fi

echo "Installing with PyTorch for CUDA $EXTRA..."
uv sync --extra dev --extra $EXTRA
