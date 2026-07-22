# Exit immediately if a command exits with a non-zero status
# set -e

# Get the absolute path to the directory where build.sh is located


PROJECT_PYTHON="$(pwd)/../.."
NUM_CORES=$(nproc)

# 3. build cpp shared library: libmxpipe.so
echo "\n\n--- Building libmxpipe.so ---"
cd "$PROJECT_PYTHON"
[ -d "build" ] && rm -rf "build" && echo "Folder deleted" || echo "Folder does not exist"
mkdir -p build && cd build
cmake ..
make -j$NUM_CORES

# 4. build pymodule: mxpipe.so
echo "\n\n--- Building mxpipe pymodule ---"
cd "$PROJECT_PYTHON/pymodule"
[ -d "build" ] && rm -rf "build" && echo "Folder deleted" || echo "Folder does not exist"
mkdir -p build && cd build
cmake ..
make -j$NUM_CORES

cd "$PROJECT_PYTHON/samples/python"