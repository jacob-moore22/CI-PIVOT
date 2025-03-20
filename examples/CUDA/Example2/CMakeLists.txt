cmake_minimum_required(VERSION 3.8)

# Enable CUDA
enable_language(CUDA)

# Set project name
project(MatrixMultiply CUDA CXX)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Add executable
add_executable(matrix_multiply matrix_multiply.cu)

# Set CUDA specific flags
set_target_properties(matrix_multiply PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(matrix_multiply PRIVATE -O3)
endif()

# Add CUDA specific flags
target_compile_options(matrix_multiply PRIVATE
    -arch=sm_80
    -std=c++14
)

# Print build information
message(STATUS "Building matrix_multiply with CUDA")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}") 