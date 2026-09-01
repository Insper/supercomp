#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::stoi(argv[1]) : 10000000);

    std::vector<int> v(N);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> U(0, 1000);
    for (int i = 0; i < N; i++) v[i] = U(rng);

    long long contador = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel for reduction(+:contador)
    for (int i = 0; i < N; i++) {
        if (v[i] % 2 == 0) contador++;
    }

    double t1 = omp_get_wtime();

    std::cout << "Total pares = " << contador << "\n";
    std::cout << "Tempo = " << (t1 - t0) << "s\n";
}
