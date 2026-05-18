#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 16;
    std::vector<int> dados;
    if (rank == 0) {
        dados.resize(N);
        for (int i = 0; i < N; i++) dados[i] = i+1;
    }

    int tam_local = N / size;
    std::vector<int> local(tam_local);

    MPI_Scatter(dados.data(), tam_local, MPI_INT,
                local.data(), tam_local, MPI_INT,
                0, MPI_COMM_WORLD);

    int soma_local = 0;
    for (int x : local) soma_local += x;

    std::vector<int> somas;
    if (rank == 0) somas.resize(size);

    MPI_Gather(&soma_local, 1, MPI_INT,
               somas.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        int soma_total = 0;
        for (int s : somas) soma_total += s;
        std::cout << "Soma total = " << soma_total << "\n";
    }

    MPI_Finalize();
}
