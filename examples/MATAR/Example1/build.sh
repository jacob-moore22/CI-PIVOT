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
BUILD_DIR="${SCRIPT_DIR}/build_${build_type}"

# Create the build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"

# Get the directory where the script is called from
CURRENT_DIR=$(pwd)
MATAR_SOURCE_DIR="${CURRENT_DIR}/MATAR"
MATAR_BUILD_DIR="${CURRENT_DIR}/build_tmp/matar"
MATAR_INSTALL_DIR="${CURRENT_DIR}/MATAR/install/matar"
MATAR_BUILD_CORES=$(nproc)

# Clone MATAR repository if it doesn't exist
if [ ! -d "${MATAR_SOURCE_DIR}" ]; then
    echo "Cloning MATAR repository..."
    git clone https://github.com/lanl/MATAR.git
fi

# Verify the repository was cloned successfully
if [ ! -f "${MATAR_SOURCE_DIR}/CMakeLists.txt" ]; then
    echo "Error: Failed to clone MATAR repository or repository is invalid"
    exit 1
fi

# Verify the build script exists
if [ ! -f "${MATAR_SOURCE_DIR}/scripts/build-matar.sh" ]; then
    echo "Error: build-matar.sh not found at ${MATAR_SOURCE_DIR}/scripts/build-matar.sh"
    exit 1
fi

# Source the build script with proper arguments
echo "Building Kokkos..."
source "${MATAR_SOURCE_DIR}/scripts/build-matar.sh" --kokkos_build_type=${build_type} --build_action=install-kokkos

echo "Building MATAR..."
source "${MATAR_SOURCE_DIR}/scripts/build-matar.sh" --kokkos_build_type=${build_type} --build_action=install-matar

echo "Matar installation complete"

# Get the actual Kokkos install directory from MATAR's build
KOKKOS_INSTALL_DIR="${CURRENT_DIR}/MATAR/install/kokkos"

# Create CMakeLists.txt for the example
echo "Creating CMakeLists.txt for matar_matmul example..."
cat > "${SCRIPT_DIR}/CMakeLists.txt" << EOF
cmake_minimum_required(VERSION 3.16)
project(MatarExample1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the namespace explicitly
add_definitions(-DMTR_NAMESPACE=mtr)
add_definitions(-DHAVE_KOKKOS=1)

# Set CMake module path to find MATAR and Kokkos
list(APPEND CMAKE_PREFIX_PATH "${MATAR_INSTALL_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${KOKKOS_INSTALL_DIR}")

# Find packages
find_package(Kokkos REQUIRED PATHS "${KOKKOS_INSTALL_DIR}" NO_DEFAULT_PATH)

# Create the executable
add_executable(matar_matmul matar_matmul.cpp)

# Add include directories for the target
target_include_directories(matar_matmul PRIVATE 
    "${MATAR_SOURCE_DIR}/src"
    "${MATAR_SOURCE_DIR}/src/include"
    "${MATAR_INSTALL_DIR}/include"
    "${KOKKOS_INSTALL_DIR}/include"
)

# Link libraries
target_link_libraries(matar_matmul 
    Kokkos::kokkos
)

# Print include directories for debugging
message(STATUS "MATAR source dir: ${MATAR_SOURCE_DIR}/src")
message(STATUS "MATAR source matar dir: ${MATAR_SOURCE_DIR}/src/matar")
message(STATUS "MATAR source kokkos dir: ${MATAR_SOURCE_DIR}/src/matar/kokkos")
message(STATUS "MATAR install include dir: ${MATAR_INSTALL_DIR}/include")
message(STATUS "MATAR install matar dir: ${MATAR_INSTALL_DIR}/include/matar")
message(STATUS "MATAR install kokkos dir: ${MATAR_INSTALL_DIR}/include/matar/kokkos")
message(STATUS "Kokkos include dir: ${KOKKOS_INSTALL_DIR}/include")
EOF

# Build the example
echo "Building matar_matmul example..."

# Create the build directory if it doesn't exist (just to be sure)
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" || {
    echo "Error: Failed to change to directory ${BUILD_DIR}"
    exit 1
}

# Configure with CMake
cmake -DCMAKE_PREFIX_PATH="${MATAR_INSTALL_DIR};${KOKKOS_INSTALL_DIR}" \
      -DMATAR_DIR="${MATAR_INSTALL_DIR}" \
      -DKokkos_DIR="${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos" \
      -DCMAKE_MODULE_PATH="${MATAR_INSTALL_DIR}/cmake" \
      "${SCRIPT_DIR}"

# Build
make -j$(nproc)

echo "Build completed!"
echo "The executable can be found at: ${BUILD_DIR}/matar_matmul"
echo ""
echo "To run the example:"
echo "cd ${BUILD_DIR}"
echo "./matar_matmul"
