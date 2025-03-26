#!/bin/bash

# Exit on any error
set -e

# Settings - customize as needed
KOKKOS_DIR=${KOKKOS_DIR:-$HOME/kokkos-install}
PARMETIS_DIR=${PARMETIS_DIR:-$HOME/parmetis-install}
MATAR_DIR=${MATAR_DIR:-../MATAR}

# Check if Kokkos is available
if [ ! -d "$KOKKOS_DIR" ]; then
    echo "Error: Kokkos directory not found at $KOKKOS_DIR"
    echo "Please install Kokkos or set KOKKOS_DIR to your Kokkos installation directory"
    exit 1
fi

# Check if ParMETIS is available
if [ ! -d "$PARMETIS_DIR" ]; then
    echo "Error: ParMETIS directory not found at $PARMETIS_DIR"
    echo "Please install ParMETIS or set PARMETIS_DIR to your ParMETIS installation directory"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Compilation settings
CXX=${CXX:-mpicxx}
CXXFLAGS="-O3 -std=c++14 -fopenmp -DHAVE_MPI -DHAVE_KOKKOS -DHAVE_OPENMP"

# Include paths
INCLUDES="-I$KOKKOS_DIR/include -I$PARMETIS_DIR/include -I$MATAR_DIR"

# Library paths
LIBS="-L$KOKKOS_DIR/lib -L$PARMETIS_DIR/lib -lparmetis -lmetis -lkokkos -lkokkoscore -lkokkoscontainers"

echo "Compiling ParMETIS example..."
$CXX $CXXFLAGS $INCLUDES ../parmetis.cpp -o parmetis_example $LIBS

echo "Build completed successfully!"
echo "Run the example with:"
echo "mpirun -np <num_procs> ./parmetis_example"

cd ..
