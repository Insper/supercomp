#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define N 1500   // Tamanho da matriz N x N

// ----------------------------------------------------------
// Multiplicação INGÊNUA (ordem CORRETA: i -> k -> j)
// ----------------------------------------------------------
void multiplicacaoIngenua(vector<double>& A,
                          vector<double>& B,
                          vector<double>& C)
{
    for (int i = 0; i < N; i++) {           // linhas

        for (int k = 0; k < N; k++) {       // dimensão interna

            double a_ik = A[i * N + k];     // guarda em registrador

            for (int j = 0; j < N; j++) {   // colunas

                C[i * N + j] +=
                    a_ik * B[k * N + j];
            }
        }
    }
}


// ----------------------------------------------------------
// Multiplicação com TILING (ordem CORRETA: ii -> kk -> jj)
// Dentro do bloco: i -> k -> j
// ----------------------------------------------------------
void multiplicacaoTiling(vector<double>& A,
                         vector<double>& B,
                         vector<double>& C,
                         int bloco)
{
    for (int ii = 0; ii < N; ii += bloco) {
        for (int kk = 0; kk < N; kk += bloco) {
            for (int jj = 0; jj < N; jj += bloco) {

                for (int i = ii; i < min(ii + bloco, N); i++) {

                    for (int k = kk; k < min(kk + bloco, N); k++) {

                        double a_ik = A[i * N + k];

                        for (int j = jj; j < min(jj + bloco, N); j++) {

                            C[i * N + j] +=
                                a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}


// ----------------------------------------------------------
// Função principal
// ----------------------------------------------------------
int main(int argc, char* argv[])
{
    int tamanhoBloco = 0;

    if (argc > 1) {
        tamanhoBloco = atoi(argv[1]);
    }

    vector<double> A(N * N, 1.0);
    vector<double> B(N * N, 2.0);
    vector<double> C(N * N, 0.0);

    auto inicio = chrono::high_resolution_clock::now();

    if (tamanhoBloco <= 0) {
        multiplicacaoIngenua(A, B, C);
    }
    else {
        multiplicacaoTiling(A, B, C, tamanhoBloco);
    }

    auto fim = chrono::high_resolution_clock::now();

    cout << "Tempo ("
         << (tamanhoBloco <= 0 ? "Ingenua"
                               : "Tiling bloco=" + to_string(tamanhoBloco))
         << "): "
         << chrono::duration_cast<chrono::milliseconds>(fim - inicio).count()
         << " s" << endl;

    cout << "Valor C[0][0] = " << C[0] << endl;

    return 0;
}
