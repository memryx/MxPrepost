#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path to the directory where build.sh is located
PROJECT_ROOT=$(pwd)
NUM_CORES=$(nproc)

echo "Starting build process..."

# 1. build cpp shared library: libmxprepost.so
echo "\n\n--- Building libmxprepost.so ---"
cd "$PROJECT_ROOT"
mkdir -p build && cd build
cmake ..
make -j$NUM_CORES

# 2. build pymodule: mxprepost.so
echo "\n\n--- Building mxprepost pymodule ---"
cd "$PROJECT_ROOT/pymodule"
mkdir -p build && cd build
cmake ..
make -j$NUM_CORES


# --- Create Symlinks ---
echo "\n\n--- Creating Symbolic Links ---"
# Navigate to the target directory
cd "$PROJECT_ROOT/samples/python"

# 1. Link mxprepost (using wildcard to handle any python version)
# Use 'ln -sf' to overwrite existing links if they exist
ln -sfv "$PROJECT_ROOT/pymodule/build"/mxprepost.cpython-*.so .

echo "Build completed successfully!"