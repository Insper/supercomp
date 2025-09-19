#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

int main() {
    int N = 1000000;
    std::vector<float> v(N);

    std::mt19937 rng(123);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (int i = 0; i < N; i++) v[i] = dist(rng);

    double soma = 0.0;

    // Versão errada (sem reduction, resultado inconsistente)
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        soma += v[i];
    }
    std::cout << "[Errada] Media = " << soma / N << "\n";

    soma = 0.0;
    // Versão correta com reduction
    #pragma omp parallel for reduction(+:soma)
    for (int i = 0; i < N; i++) {
        soma += v[i];
    }
    std::cout << "[Correta] Media = " << soma / N << "\n";
}
