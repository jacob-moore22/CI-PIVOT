#!/bin/bash -e

build_type=${1}
debug=${2}

rm -rf ${MATAR_INSTALL_DIR}
mkdir -p ${MATAR_BUILD_DIR} 

cmake_options=(
    -D CMAKE_INSTALL_PREFIX="${MATAR_INSTALL_DIR}"
    -D CMAKE_CXX_STANDARD=17
    # Add option to generate CMake config files
    -D BUILD_SHARED_LIBS=ON
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON
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
cmake "${cmake_options[@]}" -B "${MATAR_BUILD_DIR}" -S "${MATAR_SOURCE_DIR}"

# Build Matar
echo "Building Matar..."
make -C ${MATAR_BUILD_DIR} -j${MATAR_BUILD_CORES}

# Install Matar
echo "Installing Matar..."
make -C ${MATAR_BUILD_DIR} install

# Create a setup script for MATAR environment
cat > ${MATAR_INSTALL_DIR}/setup_env.sh << EOF
#!/bin/bash
export CMAKE_PREFIX_PATH=\${CMAKE_PREFIX_PATH}:${MATAR_INSTALL_DIR}
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${MATAR_INSTALL_DIR}/lib
EOF
chmod +x ${MATAR_INSTALL_DIR}/setup_env.sh

echo "Matar installation complete"