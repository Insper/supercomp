#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024;
    std::vector<float> v, out;
    if (rank == 0) {
        v.resize(N, 1.0f);
        out.resize(N);
    }

    int elems_per_proc = N / size;
    std::vector<float> local(elems_per_proc), local_out(elems_per_proc);

    MPI_Scatter(v.data(), elems_per_proc, MPI_FLOAT,
                local.data(), elems_per_proc, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    float left_halo = local[0], right_halo = local[elems_per_proc-1];
    if (rank > 0) {
        MPI_Sendrecv(&local[0], 1, MPI_FLOAT, rank-1, 0,
                     &left_halo, 1, MPI_FLOAT, rank-1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size-1) {
        MPI_Sendrecv(&local[elems_per_proc-1], 1, MPI_FLOAT, rank+1, 0,
                     &right_halo, 1, MPI_FLOAT, rank+1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    #pragma omp parallel for
    for (int i = 0; i < elems_per_proc; i++) {
        float left = (i == 0) ? left_halo : local[i-1];
        float right = (i == elems_per_proc-1) ? right_halo : local[i+1];
        local_out[i] = (left + local[i] + right) / 3.0f;
    }

    MPI_Gather(local_out.data(), elems_per_proc, MPI_FLOAT,
               out.data(), elems_per_proc, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Filtro 1D concluído." << std::endl;
    }

    MPI_Finalize();
}
