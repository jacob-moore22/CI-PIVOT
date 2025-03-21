cmake_minimum_required(VERSION 3.1)
project(VectorMatrixMPI)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") # using Clang
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -march=native -mtune=native")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") # using GCC
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -march=native -mtune=native")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel") # using Intel C
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi-alias -xHOST")

elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")

elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qalias=ansi -qhot -qarch=pwr9 -qtune=pwr9")

elseif (CMAKE_CXX_COMPILER_ID MATCHES "Cray")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -h restrict=a")

endif()

# Require MPI for this project:
find_package(MPI REQUIRED)

# Add the vecmat executable
add_executable(vecmat vector_matrix_mpi.cpp)
set_target_properties(vecmat PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
set_target_properties(vecmat PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
set_target_properties(vecmat PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
target_link_libraries(vecmat PRIVATE "${MPI_CXX_LIBRARIES}")

# Add a test for vecmat
enable_testing()
add_test(VectorMatrixTest ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/vecmat ${MPIEXEC_POSTFLAGS})

# Cleanup
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake CTestTestfile.cmake Testing)