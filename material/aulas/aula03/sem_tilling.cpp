#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

#define N 1000

/*
    Matriz armazenada como vetor 1D contíguo.
    Elemento (i, j) acessado por:
        M[i*N + j]
*/

int main() {

    vector<double> A(N * N, 1.0);
    vector<double> B(N * N, 2.0);
    vector<double> C(N * N, 0.0);

    auto start = chrono::high_resolution_clock::now();

    // Ordem otimizada: i -> k -> j
    for (int i = 0; i < N; i++) {

        for (int k = 0; k < N; k++) {

            // Guarda A[i][k] em registrador
            double a_ik = A[i * N + k];

            for (int j = 0; j < N; j++) {

                // Acesso sequencial em B e C
                C[i * N + j] +=
                    a_ik * B[k * N + j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();

    cout << "Tempo (versao otimizada i-k-j): "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    cout << "Valor C[0][0]: " << C[0] << endl;

    return 0;
}
