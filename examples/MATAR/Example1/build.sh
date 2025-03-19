#!/bin/bash

# Guard against sourcing
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed
    : # continue with script
else
    echo "This script should be executed, not sourced"
    echo "Please run: ./build.sh -t <build_type>"
    return 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [-t build_type] [-d]"
    echo "build_type options: serial, openmp, pthreads, cuda, hip"
    echo "  -t    Specify build type (required)"
    echo "  -d    Enable debug build (optional)"
    exit 1
}

# Parse command line arguments
while getopts "t:d" opt; do
    case ${opt} in
        t )
            build_type=$OPTARG
            ;;
        d )
            debug=true
            ;;
        \? )
            usage
            ;;
    esac
done

# Validate build type
if [ -z "$build_type" ]; then
    echo "Error: Build type (-t) is required"
    usage
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KOKKOS_INSTALL_SCRIPT="../kokkos-install.sh"
MATAR_INSTALL_SCRIPT="../matar-install.sh"
BUILD_DIR="${SCRIPT_DIR}/build_${build_type}"
INSTALL_DIR="${SCRIPT_DIR}/install"

# Create build and install directories
mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_DIR}"

# First, install Kokkos with the specified build type
echo "Installing Kokkos with ${build_type} backend..."
cd "${SCRIPT_DIR}"
if [ "$debug" = "true" ]; then
    bash "${KOKKOS_INSTALL_SCRIPT}" -t "${build_type}" -d -p "${INSTALL_DIR}/kokkos"
else
    bash "${KOKKOS_INSTALL_SCRIPT}" -t "${build_type}" -p "${INSTALL_DIR}/kokkos"
fi

# Set up environment variables for MATAR installation
export KOKKOS_INSTALL_DIR="${INSTALL_DIR}/kokkos"
export MATAR_INSTALL_DIR="${INSTALL_DIR}/matar"

# Install MATAR
echo "Installing MATAR..."
cd "${SCRIPT_DIR}"
if [ "$debug" = "true" ]; then
    bash "${MATAR_INSTALL_SCRIPT}" "${build_type}" "true"
else
    bash "${MATAR_INSTALL_SCRIPT}" "${build_type}"
fi

# Source the environment setup for both Kokkos and MATAR
source "${INSTALL_DIR}/kokkos/setup_env.sh"
source "${INSTALL_DIR}/matar/setup_env.sh"

# Create CMakeLists.txt for the example
cat > "${SCRIPT_DIR}/CMakeLists.txt" << EOF
cmake_minimum_required(VERSION 3.16)
project(MatarExample1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CMake module path to find MATAR and Kokkos
list(APPEND CMAKE_PREFIX_PATH "${INSTALL_DIR}/matar")
list(APPEND CMAKE_PREFIX_PATH "${INSTALL_DIR}/kokkos")

# Find packages
find_package(Kokkos REQUIRED)
find_package(MATAR REQUIRED PATHS "${INSTALL_DIR}/matar" NO_DEFAULT_PATH)

# Create the executable
add_executable(matar_matmul matar_matmul.cpp)
target_link_libraries(matar_matmul 
    Kokkos::kokkos
    MATAR::matar
)

# Add include directories explicitly if needed
target_include_directories(matar_matmul PRIVATE 
    "${INSTALL_DIR}/matar/include"
    "${INSTALL_DIR}/kokkos/include"
)
EOF

# Build the example
echo "Building matar_matmul example..."
cd "${BUILD_DIR}"

# Configure with CMake
cmake -DCMAKE_PREFIX_PATH="${INSTALL_DIR}/kokkos;${INSTALL_DIR}/matar" \
      -DMATAR_DIR="${INSTALL_DIR}/matar" \
      -DCMAKE_MODULE_PATH="${INSTALL_DIR}/matar/cmake" \
      ..

# Build
make -j$(nproc)

echo "Build completed!"
echo "The executable can be found at: ${BUILD_DIR}/matar_matmul"
echo ""
echo "To run the example:"
echo "cd ${BUILD_DIR}"
echo "./matar_matmul"
