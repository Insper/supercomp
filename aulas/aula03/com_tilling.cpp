#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define N 1000

/*
    Matriz armazenada como vetor 1D contíguo.
    Elemento (i, j):
        M[i*N + j]
*/

void versaoIngenua(vector<double>& A,
                   vector<double>& B,
                   vector<double>& C) {

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {

            double a_ik = A[i * N + k];

            for (int j = 0; j < N; j++) {
                C[i * N + j] +=
                    a_ik * B[k * N + j];
            }
        }
    }
}

void versaoTiling(vector<double>& A,
                  vector<double>& B,
                  vector<double>& C,
                  int Bsize) {

    for (int ii = 0; ii < N; ii += Bsize) {
        for (int kk = 0; kk < N; kk += Bsize) {
            for (int jj = 0; jj < N; jj += Bsize) {

                // multiplicação do bloco
                for (int i = ii; i < min(ii + Bsize, N); i++) {
                    for (int k = kk; k < min(kk + Bsize, N); k++) {

                        double a_ik = A[i * N + k];

                        for (int j = jj; j < min(jj + Bsize, N); j++) {

                            C[i * N + j] +=
                                a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {

    int Bsize = 0;

    if (argc > 1) {
        Bsize = atoi(argv[1]);
    }

    vector<double> A(N * N, 1.0);
    vector<double> B(N * N, 2.0);
    vector<double> C(N * N, 0.0);

    auto start = chrono::high_resolution_clock::now();

    if (Bsize <= 0) {
        versaoIngenua(A, B, C);
    } else {
        versaoTiling(A, B, C, Bsize);
    }

    auto end = chrono::high_resolution_clock::now();

    cout << "Tempo ("
         << (Bsize <= 0 ? "ingenua" : "tiling B=" + to_string(Bsize))
         << "): "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    cout << "Check C[0]: " << C[0] << endl;

    return 0;
}
