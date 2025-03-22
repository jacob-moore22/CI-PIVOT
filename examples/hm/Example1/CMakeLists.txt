cmake_minimum_required(VERSION 3.16)
project(KokkosExample1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Kokkos
find_package(Kokkos REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/../MATAR)

# Create the executable
add_executable(matmul matmul.cpp)

target_link_libraries(matmul Kokkos::kokkos)
