cmake_minimum_required(VERSION 3.10)
project(matar_matmul)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add option for build type
# To set this option when calling cmake, use:
#   cmake -DBUILD_TYPE=serial|cuda|openmp|pthreads|hip ..
option(BUILD_TYPE "Type of build (serial, cuda, etc.)" "serial")

# Set paths using the build type
set(Kokkos_DIR ${CMAKE_SOURCE_DIR}/MATAR/build-matar-${BUILD_TYPE}/kokkos)
set(MATAR_DIR ${CMAKE_SOURCE_DIR}/MATAR/build-matar-${BUILD_TYPE}/matar)

# Add Kokkos and MATAR to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH 
    ${CMAKE_SOURCE_DIR}/MATAR/build-matar-${BUILD_TYPE}/kokkos
    ${CMAKE_SOURCE_DIR}/MATAR/build-matar-${BUILD_TYPE}/matar
)

# Find required packages
find_package(Kokkos REQUIRED)

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

message(STATUS "Matar headers: ${CMAKE_SOURCE_DIR}../MATAR/src/include")

# Add executable
add_executable(matmul matar_matmul.cpp)

target_include_directories(MyExecutable PRIVATE /path/to/headers)

# Link against required libraries
target_link_libraries(matmul PRIVATE 
    Kokkos::kokkos
)

# Add include directories
target_include_directories(matmul PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}../MATAR/src/include
)

# Print configuration information
message(STATUS "Building MATAR matrix multiplication example")
message(STATUS "Build type: ${BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Kokkos found: ${Kokkos_FOUND}")
message(STATUS "MATAR found: ${MATAR_FOUND}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}") 