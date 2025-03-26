#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
#include <parmetis.h>
#include <Kokkos_Core.hpp>
#include "matar.h"

// Required for MATAR data structures
using namespace mtr;

/**
 * DistributedDCArray: A class that extends DCArrayKokkos with MPI communication capabilities
 * for distributed graph partitioning using ParMETIS
 * 
 * This class handles:
 * - Wrapping ParMETIS to decompose a graph
 * - Building the connectivity for HALO communications 
 * - Automating the communication process via a simple .comm() command
 */
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class DistributedDCArray {
private:
    using DataArray = Kokkos::DualView<T*, Layout, ExecSpace, MemoryTraits>;
    using IndexArray = Kokkos::DualView<idx_t*, Layout, ExecSpace, MemoryTraits>;
    using CommunicationMap = CArrayKokkos<int>;

    // MPI related members
    int processRank_;
    int totalProcesses_;
    MPI_Comm communicator_;
    
    // ParMETIS related members
   
   
    // Array storing the distribution of vertices across processors
    // vertexDistribution_[i] contains the first global vertex number owned by processor i
    // Size is number of processors + 1, where vertexDistribution_[p+1] - vertexDistribution_[p] gives
    // number of vertices owned by processor p
    IndexArray vertexDistribution_;
    
         
    // Array storing indices into adjacencyList_ array for each vertex's adjacency list
    // For vertex i, its adjacent vertices are stored in adjacencyList_[adjacencyPointers_[i]] through
    // adjacencyList_[adjacencyPointers_[i+1]-1]. Size is number of local vertices + 1
    IndexArray adjacencyPointers_;   
    
    // Array storing adjacent vertices in compressed format
    // Contains concatenated lists of adjacent vertices for each local vertex
    // Vertices are stored using global numbering
    IndexArray adjacencyList_;      
    
    // Array storing the partition assignment for each vertex
    // vertexPartition_[i] contains the partition number (0 to numPartitions-1) that vertex i
    // is assigned to after partitioning
    IndexArray vertexPartition_;        
    
    // Local-to-global and global-to-local mappings
    CommunicationMap localToGlobalMap_;  // Maps local indices to global indices
    CommunicationMap globalToLocalMap_;  // Maps global indices to local indices
    
    // HALO communication data
    CommunicationMap neighborProcesses_;         // List of neighbor processes
    CommunicationMap indicesToSend_;      // Indices to send to each neighbor
    CommunicationMap indicesToReceive_;      // Indices to receive from each neighbor
    CommunicationMap sendCounts_;       // Number of elements to send to each neighbor
    CommunicationMap receiveCounts_;       // Number of elements to receive from each neighbor
    CommunicationMap sendOffsets_; // Displacements for sends
    CommunicationMap receiveOffsets_; // Displacements for receives
    
    // Data array containing both owned and halo elements
    DCArrayKokkos<T, Layout, ExecSpace, MemoryTraits> meshData_;
    
    // Count of owned elements vs. total (owned + halo)
    size_t localElementCount_;
    size_t totalElementCount_;
    
    /**
     * Sets up the HALO (ghost) region communication patterns between processes
     * 
     * This function:
     * 1. Analyzes the adjacency graph to identify neighboring processes
     * 2. Determines which vertices need to be sent/received between processes
     * 3. Sets up the communication buffers and patterns for efficient data exchange
     * 
     * Key data structures initialized:
     * - neighborProcesses_: Array storing the ranks of neighboring processes
     * - sendCounts_: Number of vertices to send to each neighbor
     * - receiveCounts_: Number of vertices to receive from each neighbor
     * - sendOffsets_: Offset positions for sending data
     * - receiveOffsets_: Offset positions for receiving data
     * - sendIndices: Temporary vectors storing which local vertices to send
     */
    void setup_halo_communications() {
        // First, determine which processes own adjacent vertices
        int neighborCount = 0;
        std::vector<int> tempNeighbors;
        
        // For each boundary vertex, find which process owns it
        for (size_t i = 0; i < adjacencyList_.extent(0); i++) {
            idx_t globalVertexIndex = adjacencyList_.h_view(i);
            
            // Find process that owns this vertex
            int ownerProcess = -1;
            for (int p = 0; p < totalProcesses_; p++) {
                if (globalVertexIndex >= vertexDistribution_.h_view(p) && globalVertexIndex < vertexDistribution_.h_view(p+1)) {
                    ownerProcess = p;
                    break;
                }
            }
            
            // If it's not owned by current process, add to neighbor list
            if (ownerProcess != processRank_ && ownerProcess != -1) {
                // Check if this neighbor is already in our list
                bool alreadyFound = false;
                for (int n : tempNeighbors) {
                    if (n == ownerProcess) {
                        alreadyFound = true;
                        break;
                    }
                }
                
                if (!alreadyFound) {
                    tempNeighbors.push_back(ownerProcess);
                    neighborCount++;
                }
            }
        }
        
        // Setup neighbor arrays
        neighborProcesses_ = CommunicationMap(neighborCount);
        sendCounts_ = CommunicationMap(neighborCount);
        receiveCounts_ = CommunicationMap(neighborCount);
        sendOffsets_ = CommunicationMap(neighborCount);
        receiveOffsets_ = CommunicationMap(neighborCount);
        
        // Copy neighbors to Kokkos array
        for (int i = 0; i < neighborCount; i++) {
            neighborProcesses_.host(i) = tempNeighbors[i];
        }
        
        // For each neighbor, determine which vertices to send/receive
        std::vector<std::vector<int>> indicesToSendByNeighbor(neighborCount);
        
        for (size_t i = 0; i < adjacencyList_.extent(0); i++) {
            idx_t globalVertexIndex = adjacencyList_.h_view(i);
            
            // Find process that owns this vertex
            int ownerProcess = -1;
            for (int p = 0; p < totalProcesses_; p++) {
                if (globalVertexIndex >= vertexDistribution_.h_view(p) && globalVertexIndex < vertexDistribution_.h_view(p+1)) {
                    ownerProcess = p;
                    break;
                }
            }
            
            // If it's owned by a neighbor, add to the send list for that neighbor
            if (ownerProcess != processRank_ && ownerProcess != -1) {
                // Find index of ownerProcess in neighbor list
                int neighborIndex = -1;
                for (int n = 0; n < neighborCount; n++) {
                    if (neighborProcesses_.host(n) == ownerProcess) {
                        neighborIndex = n;
                        break;
                    }
                }
                
                if (neighborIndex != -1) {
                    // Convert global index to local index
                    size_t localVertexIndex = globalVertexIndex - vertexDistribution_.h_view(processRank_);
                    indicesToSendByNeighbor[neighborIndex].push_back(localVertexIndex);
                }
            }
        }
        
        // Set send counts and allocate send indices arrays
        for (int i = 0; i < neighborCount; i++) {
            sendCounts_.host(i) = indicesToSendByNeighbor[i].size();
        }
        
        // Communicate send counts to determine receive counts
        receiveCounts_ = CommunicationMap(neighborCount);
        
        for (int i = 0; i < neighborCount; i++) {
            int destinationRank = neighborProcesses_.host(i);
            int elementsToSend = sendCounts_.host(i);
            int elementsToReceive;
            
            MPI_Sendrecv(&elementsToSend, 1, MPI_INT, destinationRank, 0,
                         &elementsToReceive, 1, MPI_INT, destinationRank, 0,
                         communicator_, MPI_STATUS_IGNORE);
            
            receiveCounts_.host(i) = elementsToReceive;
        }
        
        // Calculate displacements
        int sendOffset = 0;
        int receiveOffset = 0;
        
        for (int i = 0; i < neighborCount; i++) {
            sendOffsets_.host(i) = sendOffset;
            sendOffset += sendCounts_.host(i);
            
            receiveOffsets_.host(i) = receiveOffset;
            receiveOffset += receiveCounts_.host(i);
        }
        
        // Allocate and set send indices
        int totalSendCount = sendOffset;
        indicesToSend_ = CommunicationMap(totalSendCount);
        
        int idx = 0;
        for (int i = 0; i < neighborCount; i++) {
            for (size_t j = 0; j < indicesToSendByNeighbor[i].size(); j++) {
                indicesToSend_.host(idx++) = indicesToSendByNeighbor[i][j];
            }
        }
        
        // Allocate receive indices
        int totalReceiveCount = receiveOffset;
        indicesToReceive_ = CommunicationMap(totalReceiveCount);
        
        // Update total count to include halo elements
        totalElementCount_ = localElementCount_ + totalReceiveCount;
        
        // Resize data array to accommodate both owned and halo elements
        meshData_ = DCArrayKokkos<T, Layout, ExecSpace, MemoryTraits>(totalElementCount_);
    }

public:
    // Constructors
    DistributedDCArray() : processRank_(0), totalProcesses_(1), localElementCount_(0), totalElementCount_(0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &processRank_);
        MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses_);
        communicator_ = MPI_COMM_WORLD;
    }
    
    DistributedDCArray(MPI_Comm comm) : localElementCount_(0), totalElementCount_(0) {
        communicator_ = comm;
        MPI_Comm_rank(comm, &processRank_);
        MPI_Comm_size(comm, &totalProcesses_);
    }
    
    /**
     * Initialize the graph for partitioning
     * 
     * @param vertexDistData Distribution of vertices among processors
     * @param adjacencyPtrData Adjacency structure indices
     * @param adjacencyListData Adjacent vertices
     */
    void init_graph(idx_t* vertexDistData, size_t vertexDistSize,
                   idx_t* adjacencyPtrData, size_t adjacencyPtrSize,
                   idx_t* adjacencyListData, size_t adjacencyListSize) {
        // Allocate arrays
        vertexDistribution_ = IndexArray("vertexDistribution", vertexDistSize);
        adjacencyPointers_ = IndexArray("adjacencyPointers", adjacencyPtrSize);
        adjacencyList_ = IndexArray("adjacencyList", adjacencyListSize);
        
        // Copy data to host views
        for (size_t i = 0; i < vertexDistSize; i++) {
            vertexDistribution_.h_view(i) = vertexDistData[i];
        }
        
        for (size_t i = 0; i < adjacencyPtrSize; i++) {
            adjacencyPointers_.h_view(i) = adjacencyPtrData[i];
        }
        
        for (size_t i = 0; i < adjacencyListSize; i++) {
            adjacencyList_.h_view(i) = adjacencyListData[i];
        }
        
        // Update device views
        // The .template syntax is required when calling a template member function on a dependent type
        // IndexArray::host_mirror_space is a dependent type since IndexArray is a template parameter
        // modify() marks the host view as modified so the next sync will copy data to device
        vertexDistribution_.template modify<typename IndexArray::host_mirror_space>();
        vertexDistribution_.template sync<typename IndexArray::execution_space>();
        
        adjacencyPointers_.template modify<typename IndexArray::host_mirror_space>();
        adjacencyPointers_.template sync<typename IndexArray::execution_space>();
        
        adjacencyList_.template modify<typename IndexArray::host_mirror_space>();
        adjacencyList_.template sync<typename IndexArray::execution_space>();
        
        // Calculate owned count
        localElementCount_ = vertexDistribution_.h_view(processRank_ + 1) - vertexDistribution_.h_view(processRank_);
        totalElementCount_ = localElementCount_;
        
        // Initialize data array
        meshData_ = DCArrayKokkos<T, Layout, ExecSpace, MemoryTraits>(totalElementCount_);
    }
    
    /**
     * Partition the graph using ParMETIS
     * 
     * @param numPartitions Number of partitions
     * @param partitioningMethod ParMETIS partitioning method
     * @return Error code from ParMETIS
     */
    int partition(int numPartitions, int partitioningMethod = PARMETIS_PSR_UNCOUPLED) {
        // Initialize partition array
        vertexPartition_ = IndexArray("vertexPartition", localElementCount_);
        
        // ParMETIS parameters
        idx_t useWeights = 0;  // No weights
        idx_t zeroBasedIndexing = 0;  // 0-based indexing
        idx_t numConstraints = 1;     // Number of balancing constraints
        real_t* targetPartitionWeights = new real_t[numConstraints * numPartitions];
        real_t* imbalanceTolerance = new real_t[numConstraints];
        idx_t parmetisOptions[METIS_NOPTIONS];
        idx_t cutEdgeCount;
        
        // Set balanced partitioning
        for (int i = 0; i < numConstraints * numPartitions; i++) {
            targetPartitionWeights[i] = 1.0 / numPartitions;
        }
        
        // Set maximum allowed imbalance
        for (int i = 0; i < numConstraints; i++) {
            imbalanceTolerance[i] = 1.05;  // 5% imbalance tolerance
        }
        
        // Set default options
        parmetisOptions[0] = 0;
        
        // Get number of vertices for this processor
        idx_t localVertexCount = vertexDistribution_.h_view(processRank_ + 1) - vertexDistribution_.h_view(processRank_);
        
        // Call ParMETIS to partition the graph
        int result;
        if (partitioningMethod == PARMETIS_PSR_UNCOUPLED) {
            result = ParMETIS_V3_PartKway(vertexDistribution_.h_view.data(), adjacencyPointers_.h_view.data(), adjacencyList_.h_view.data(),
                                         NULL, NULL, &useWeights, &zeroBasedIndexing, &numConstraints, &numPartitions,
                                         targetPartitionWeights, imbalanceTolerance, parmetisOptions, &cutEdgeCount, vertexPartition_.h_view.data(), &communicator_);
        } else {
            result = ParMETIS_V3_PartGeomKway(vertexDistribution_.h_view.data(), adjacencyPointers_.h_view.data(), adjacencyList_.h_view.data(),
                                            NULL, NULL, &useWeights, &zeroBasedIndexing, &numConstraints, &numPartitions,
                                            targetPartitionWeights, imbalanceTolerance, parmetisOptions, &cutEdgeCount, vertexPartition_.h_view.data(), &communicator_);
        }
        
        if (result == METIS_OK) {
            // Update device view
            vertexPartition_.template modify<typename IndexArray::host_mirror_space>();
            vertexPartition_.template sync<typename IndexArray::execution_space>();
            
            // Setup HALO communications
            setup_halo_communications();
            
            // Print partition info on rank 0
            if (processRank_ == 0) {
                std::cout << "ParMETIS partitioning completed successfully!" << std::endl;
                std::cout << "Edge-cut: " << cutEdgeCount << std::endl;
            }
        } else {
            if (processRank_ == 0) {
                std::cout << "ParMETIS partitioning failed with error code: " << result << std::endl;
            }
        }
        
        // Clean up
        delete[] targetPartitionWeights;
        delete[] imbalanceTolerance;
        
        return result;
    }
    
    /**
     * Get the partition vector
     * 
     * @return Partition vector
     */
    idx_t* get_partition() {
        return vertexPartition_.h_view.data();
    }
    
    /**
     * Access data element (both owned and halo)
     * 
     * @param i Index
     * @return Reference to data element
     */
    T& operator()(size_t i) const {
        return meshData_(i);
    }
    
    /**
     * Perform HALO communications to sync ghost regions
     */
    void comm() {
        // Need to communicate data for ghost regions
        meshData_.update_host();
        
        // Allocate send and receive buffers
        int totalSendCount = 0;
        for (size_t i = 0; i < sendCounts_.extent(0); i++) {
            totalSendCount += sendCounts_.host(i);
        }
        
        int totalReceiveCount = 0;
        for (size_t i = 0; i < receiveCounts_.extent(0); i++) {
            totalReceiveCount += receiveCounts_.host(i);
        }
        
        T* sendBuffer = new T[totalSendCount];
        T* receiveBuffer = new T[totalReceiveCount];
        
        // Pack send buffer
        for (int i = 0; i < totalSendCount; i++) {
            sendBuffer[i] = meshData_.host(indicesToSend_.host(i));
        }
        
        // Use MPI_Alltoallv to exchange data
        MPI_Alltoallv(sendBuffer, sendCounts_.host.data(), sendOffsets_.host.data(), MPI_BYTE,
                     receiveBuffer, receiveCounts_.host.data(), receiveOffsets_.host.data(), MPI_BYTE,
                     communicator_);
        
        // Unpack receive buffer
        for (int i = 0; i < totalReceiveCount; i++) {
            // Halo elements are stored after owned elements
            meshData_.host(localElementCount_ + i) = receiveBuffer[i];
        }
        
        // Clean up
        delete[] sendBuffer;
        delete[] receiveBuffer;
        
        // Update device view
        meshData_.update_device();
    }
    
    /**
     * Get the number of owned elements
     * 
     * @return Number of owned elements
     */
    size_t get_owned_count() const {
        return localElementCount_;
    }
    
    /**
     * Get the total number of elements (owned + halo)
     * 
     * @return Total number of elements
     */
    size_t get_total_count() const {
        return totalElementCount_;
    }
    
    /**
     * Get the MPI rank
     * 
     * @return MPI rank
     */
    int get_rank() const {
        return processRank_;
    }
    
    /**
     * Get the MPI communicator size
     * 
     * @return MPI communicator size
     */
    int get_size() const {
        return totalProcesses_;
    }
    
    /**
     * Get the MPI communicator
     * 
     * @return MPI communicator
     */
    MPI_Comm get_comm() const {
        return communicator_;
    }
    
    /**
     * Set data for local elements
     * 
     * @param value Value to set
     */
    void set_values(T value) {
        // Set all elements to value
        Kokkos::parallel_for("SetValues_DistributedDCArray", totalElementCount_, KOKKOS_CLASS_LAMBDA(const int i) {
            meshData_.d_view(i) = value;
        });
    }
    
    /**
     * Destructor
     */
    virtual ~DistributedDCArray() {}
};

/**
 * Main function to demonstrate ParMETIS graph partitioning with MATAR
 */
int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get MPI process info
    int processRank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // Create a simple mesh graph for demonstration
        // For this example, we'll create a 2D grid mesh
        
        // Grid dimensions
        int gridWidth = 10;  // number of vertices in x direction
        int gridHeight = 10;  // number of vertices in y direction
        int numPartitions = numProcesses;  // number of partitions (one per process)
        
        // Total number of vertices
        int totalVertices = gridWidth * gridHeight;
        
        // Distribute vertices evenly among processors
        std::vector<idx_t> vertexDistribution(numProcesses + 1);
        for (int i = 0; i <= numProcesses; i++) {
            vertexDistribution[i] = (totalVertices * i) / numProcesses;
        }
        
        // Number of vertices for this processor
        int localVertexCount = vertexDistribution[processRank + 1] - vertexDistribution[processRank];
        
        // Build adjacency structure for a 2D grid
        std::vector<idx_t> adjacencyPointers(localVertexCount + 1);
        std::vector<idx_t> adjacencyList;
        
        adjacencyPointers[0] = 0;
        
        for (int i = 0; i < localVertexCount; i++) {
            int globalVertexIndex = vertexDistribution[processRank] + i;
            int xCoord = globalVertexIndex % gridWidth;
            int yCoord = globalVertexIndex / gridWidth;
            
            // Add neighbors (up to 4 for a 2D grid)
            // Left neighbor
            if (xCoord > 0) {
                adjacencyList.push_back(globalVertexIndex - 1);
            }
            
            // Right neighbor
            if (xCoord < gridWidth - 1) {
                adjacencyList.push_back(globalVertexIndex + 1);
            }
            
            // Top neighbor
            if (yCoord > 0) {
                adjacencyList.push_back(globalVertexIndex - gridWidth);
            }
            
            // Bottom neighbor
            if (yCoord < gridHeight - 1) {
                adjacencyList.push_back(globalVertexIndex + gridWidth);
            }
            
            adjacencyPointers[i + 1] = adjacencyList.size();
        }
        
        // Create DistributedDCArray object
        DistributedDCArray<double> mesh;
        
        // Initialize the graph
        mesh.init_graph(vertexDistribution.data(), vertexDistribution.size(),
                       adjacencyPointers.data(), adjacencyPointers.size(),
                       adjacencyList.data(), adjacencyList.size());
        
        // Partition the graph
        mesh.partition(numPartitions);
        
        // Set values based on rank for demonstration
        mesh.set_values(static_cast<double>(processRank));
        
        // Perform HALO communications
        mesh.comm();
        
        // Check some values after communication
        if (processRank == 0) {
            std::cout << "After communication on rank " << processRank << ":" << std::endl;
            std::cout << "Owned elements: " << mesh.get_owned_count() << std::endl;
            std::cout << "Total elements (owned + halo): " << mesh.get_total_count() << std::endl;
            
            // Print some values from halo regions
            if (mesh.get_total_count() > mesh.get_owned_count()) {
                std::cout << "First halo element: " << mesh(mesh.get_owned_count()) << std::endl;
            }
        }
        
        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
