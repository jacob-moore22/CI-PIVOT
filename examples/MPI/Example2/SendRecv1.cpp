#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    // Initialize the MPI environment
    // This must be called before any other MPI function
    // This call generates the default MPI_COM_WORLD communicator
    MPI_Init(&argc, &argv);

    // Define the size of our data arrays
    int count = 10;
    // Arrays to hold data we want to send and receive
    double xsend[count], xrecv[count];
    // Initialize the send array with some values (0 to 9)
    for (int i=0; i<count; i++){
        xsend[i] = (double)i;
    }

    // Get the rank (process ID) of the current process
    // Each process in MPI has a unique rank starting from 0
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get the total number of processes in the communicator
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Check if we have an even number of processes
    // This is important for our pairing strategy
    if (nprocs%2 == 1){
        if (rank == 0){
            printf("Error: Must be called with an even number of processes\n");
        }
        exit(1);
    }

    // Calculate the tag and partner rank for communication
    // tag: Used to identify different types of messages
    //      In this case, we use rank/2 to group processes into pairs
    int tag = rank/2;
    // partner_rank: Calculate which process we should communicate with
    //               For each pair of processes, they communicate with each other
    //               Example: Process 0 pairs with 1, 2 pairs with 3, etc.
    int partner_rank = (rank/2)*2 + (rank+1)%2;
    // Define the communicator (in this case, using the default MPI_COMM_WORLD)
    MPI_Comm comm = MPI_COMM_WORLD;

    // MPI_Recv: Receive data from our partner process
    // Parameters:
    //   xrecv: Buffer to store received data
    //   count: Number of elements to receive
    //   MPI_DOUBLE: Data type of the elements
    //   partner_rank: Rank of the process to receive from
    //   tag: Message tag to match with corresponding send
    //   comm: Communicator to use
    //   MPI_STATUS_IGNORE: We don't need status information
    MPI_Recv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, MPI_STATUS_IGNORE);
    
    // MPI_Send: Send data to our partner process
    // Parameters:
    //   xsend: Data to send
    //   count: Number of elements to send
    //   MPI_DOUBLE: Data type of the elements
    //   partner_rank: Rank of the process to send to
    //   tag: Message tag to match with corresponding receive
    //   comm: Communicator to use
    MPI_Send(xsend, count, MPI_DOUBLE, partner_rank, tag, comm);

    // Only process 0 prints the completion message
    if (rank == 0) printf("SendRecv successfully completed\n");

    // Clean up the MPI environment
    // This must be called before the program exits
    MPI_Finalize();
    return 0;
}