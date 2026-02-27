#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024;
    std::vector<int> A;
    if (rank == 0) {
        A.resize(N*N);
        for (int i = 0; i < N*N; i++) A[i] = rand() % 10;
    }

    int rows_per_proc = N / size;
    std::vector<int> local(rows_per_proc * N);

    MPI_Scatter(A.data(), rows_per_proc * N, MPI_INT,
                local.data(), rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);

    long long soma_local = 0;
    #pragma omp parallel for reduction(+:soma_local)
    for (int i = 0; i < rows_per_proc * N; i++) {
        soma_local += local[i];
    }

    long long soma_total = 0;
    MPI_Reduce(&soma_local, &soma_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Soma total = " << soma_total << std::endl;
    }

    MPI_Finalize();
}
