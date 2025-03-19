#!/bin/bash -e

build_type=${1}
debug=${2}

# Get the directory where the script is called from
CURRENT_DIR=$(pwd)
MATAR_SOURCE_DIR="${CURRENT_DIR}/MATAR"
MATAR_BUILD_DIR="${CURRENT_DIR}/build_tmp/matar"
MATAR_INSTALL_DIR="${CURRENT_DIR}/install/matar"
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

# Create necessary directories
mkdir -p "${MATAR_BUILD_DIR}"
mkdir -p "${MATAR_INSTALL_DIR}"

# Clean previous installation
rm -rf "${MATAR_INSTALL_DIR}"

cmake_options=(
    -D CMAKE_INSTALL_PREFIX="${MATAR_INSTALL_DIR}"
    -D CMAKE_CXX_STANDARD=17
    -D BUILD_SHARED_LIBS=ON
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON
    -D Matar_ENABLE_INSTALL=ON
    -D Matar_ENABLE_EXAMPLES=OFF
    -D Matar_ENABLE_TESTS=OFF
)

if [ "$debug" = "true" ]; then
    echo "Setting debug to true for CMAKE build type"
    cmake_options+=(
        -DCMAKE_BUILD_TYPE=Debug
    )
else
    cmake_options+=(
        -DCMAKE_BUILD_TYPE=Release
    )
fi

# Always enable Kokkos
cmake_options+=(
    -D Matar_ENABLE_KOKKOS=ON
    -D CMAKE_PREFIX_PATH="${KOKKOS_INSTALL_DIR}"
)

# Handle different build types
case ${build_type} in
    "cuda")
        cmake_options+=(-D Matar_CUDA_BUILD=ON)
        ;;
    "openmp")
        # OpenMP is handled by Kokkos
        ;;
    "hip")
        # HIP support if needed
        ;;
esac

# Print CMake options for reference
echo "CMake Options: ${cmake_options[@]}"

# Configure Matar
echo "Configuring Matar..."
cmake "${cmake_options[@]}" -B "${MATAR_BUILD_DIR}" -S "${MATAR_SOURCE_DIR}"

# Build Matar
echo "Building Matar..."
make -C "${MATAR_BUILD_DIR}" -j${MATAR_BUILD_CORES}

# Install Matar
echo "Installing Matar..."
make -C "${MATAR_BUILD_DIR}" install

# Create a setup script for MATAR environment
cat > "${MATAR_INSTALL_DIR}/setup_env.sh" << EOF
#!/bin/bash
export CMAKE_PREFIX_PATH=\${CMAKE_PREFIX_PATH}:${MATAR_INSTALL_DIR}
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${MATAR_INSTALL_DIR}/lib
export MATAR_DIR="${MATAR_INSTALL_DIR}/include/lib/cmake/matar"
EOF
chmod +x "${MATAR_INSTALL_DIR}/setup_env.sh"

source "${MATAR_INSTALL_DIR}/setup_env.sh"

echo "Matar installation complete!"
echo "Installation location: ${MATAR_INSTALL_DIR}"
echo "To set up the environment variables, run:"
echo "source ${MATAR_INSTALL_DIR}/setup_env.sh"