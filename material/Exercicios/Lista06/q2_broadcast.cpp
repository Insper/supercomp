#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int valor;
    if (rank == 0) {
        valor = 42;
        // versão manual
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(&valor, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&valor, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    std::cout << "[Manual] Processo " << rank << " recebeu valor = " << valor << "\n";

    // versão com MPI_Bcast
    if (rank == 0) valor = 99;
    MPI_Bcast(&valor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << "[Bcast] Processo " << rank << " recebeu valor = " << valor << "\n";

    MPI_Finalize();
}
