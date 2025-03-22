
project(matar_matmul)

# Set C++ standard
# Usage of C++17 standard requires CMake version >= 3.8
cmake_minimum_required(VERSION 3.8)

# Current usage of shared_ptr in MATAR requires C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add option for build type
# To set this option when calling cmake, use:
#   cmake -DBUILD_TYPE=serial|cuda|openmp|pthreads|hip ..
option(BUILD_TYPE "Type of build (serial, cuda, etc.)" "serial")

# Set paths using the build type
set(Kokkos_DIR ${CMAKE_SOURCE_DIR}/MATAR/build-matar-${BUILD_TYPE}/kokkos)
set(Matar_DIR ${CMAKE_SOURCE_DIR}/MATAR/build-matar-${BUILD_TYPE}/matar)

# Check if BUILD_TYPE is set
if(NOT BUILD_TYPE)
    message(FATAL_ERROR "BUILD_TYPE not set. Please specify build type using:
    cmake -DBUILD_TYPE=<type> ..
    where <type> is one of: serial, cuda, openmp, pthreads, hip")
endif()

# Add definitions based on build type
add_definitions(-DHAVE_KOKKOS=1)

if(BUILD_TYPE STREQUAL "openmp")
    add_definitions(-DHAVE_OPENMP=1)
endif()

if(BUILD_TYPE STREQUAL "cuda") 
    add_definitions(-DHAVE_CUDA=1)
endif()

# Add Kokkos and MATAR to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH 
    ${CMAKE_SOURCE_DIR}/MATAR/build-matar-${BUILD_TYPE}/kokkos
    ${CMAKE_SOURCE_DIR}/MATAR/build-matar-${BUILD_TYPE}/matar
)

# Find required packages
find_package(Kokkos REQUIRED)
find_package(Matar REQUIRED)

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

message(STATUS "Matar headers: ${CMAKE_SOURCE_DIR}/MATAR/src/include")

# Add executable
add_executable(matmul matar_matmul.cpp)

# target_include_directories(matmul PRIVATE /path/to/headers)

# Link against required libraries
target_link_libraries(matmul PRIVATE 
    Kokkos::kokkos
)

# Add include directories
target_include_directories(matmul PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/MATAR/src/include
)

# Print configuration information
message(STATUS "Building MATAR matrix multiplication example")
message(STATUS "Build type: ${BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Kokkos found: ${Kokkos_FOUND}")
message(STATUS "MATAR found: ${MATAR_FOUND}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}") 