#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv)
{
    // MPI_Init is required for all MPI programs
    // this call generates the default MPI_COM_WORLD communicator
    MPI_Init(&argc, &argv);

    int rank, nprocs;

    // Get the rank ID of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the number of ranks in the program defined by the MPI run command
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    printf("Rank %d of %d\n", rank, nprocs);

    // MPI_Finalize is required at the end of all MPI codes, it manages
    // teardown and deallocation of memory for MPI communication
    // If you dont call this, your program will likley segfault and/or have large
    // memory leaks. 
    MPI_Finalize();
    return 0;
}

/*
To build this code, use the following commands at this directory level
cmake .
make
make test

Or, call it directly using mpiexec: 
mpiexec -n 1  ./MinWorkExampleMPI


To cleanup after the run, use the following commands
make clean
make distclean

*/