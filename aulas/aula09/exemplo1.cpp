#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Inicializa o MPI.
    MPI_Init(&argc, &argv);

    char nome[MPI_MAX_PROCESSOR_NAME];
    int tamanho, rank, total;
    // Pega informações relevantes do ecossistema MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Get_processor_name(nome, &tamanho);

    std::cout << "Rank " << rank << " | Nó: " << nome << std::endl;

    if (rank == 0) {
        // O rank 0 apresenta a tarefa.
        std::cout << "Tarefa: calcular o quadrado de cada ID.\n"
                  << "Temos " << total << " nós.\n";
    } else {
        // Cada nó calcula usando seu próprio ID.
        int resultado = rank * rank;

        std::cout << ": Minha computação é = " << resultado << std::endl;
    }

    // Finaliza o MPI.
    MPI_Finalize();
    return 0;
}