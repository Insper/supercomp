#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int meu_valor = rank + 1; // só para ter valores diferentes
    int soma = 0;

    if (rank == 0) {
        soma = meu_valor;
        for (int src = 1; src < size; src++) {
            int temp;
            MPI_Recv(&temp, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            soma += temp;
        }
        std::cout << "[Manual] Soma total = " << soma << "\n";
    } else {
        MPI_Send(&meu_valor, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // versão com MPI_Reduce
    int soma_reduce = 0;
    MPI_Reduce(&meu_valor, &soma_reduce, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "[Reduce] Soma total = " << soma_reduce << "\n";
    }

    MPI_Finalize();
}
