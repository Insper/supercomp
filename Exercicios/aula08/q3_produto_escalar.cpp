#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1<<20;
    std::vector<float> v, w;
    if (rank == 0) {
        v.resize(N, 1.0f);
        w.resize(N, 2.0f);
    }

    int elems_per_proc = N / size;
    std::vector<float> v_local(elems_per_proc), w_local(elems_per_proc);

    MPI_Scatter(v.data(), elems_per_proc, MPI_FLOAT,
                v_local.data(), elems_per_proc, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    MPI_Scatter(w.data(), elems_per_proc, MPI_FLOAT,
                w_local.data(), elems_per_proc, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    double soma_local = 0.0;
    #pragma omp parallel for reduction(+:soma_local)
    for (int i = 0; i < elems_per_proc; i++) {
        soma_local += v_local[i] * w_local[i];
    }

    double soma_total = 0.0;
    MPI_Reduce(&soma_local, &soma_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Produto escalar = " << soma_total << std::endl;
    }

    MPI_Finalize();
}
