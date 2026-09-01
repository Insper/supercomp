#include <iostream>
#include <vector>
#include <omp.h>

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::stoi(argv[1]) : 1000000);
    int K = 5; // tamanho do kernel

    std::vector<float> a(N, 1.0f), kernel(K, 0.2f), c(N-K+1, 0.0f);

    double t0 = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N - K + 1; i++) {
        float soma = 0.0f;
        for (int j = 0; j < K; j++) {
            soma += a[i + j] * kernel[j];
        }
        c[i] = soma;
    }

    double t1 = omp_get_wtime();

    std::cout << "Tempo convolução = " << (t1 - t0) << "s\n";
}
