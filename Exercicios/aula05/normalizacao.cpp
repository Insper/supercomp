#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::stoi(argv[1]) : 10000000);

    std::vector<float> a(N), b(N);

    std::mt19937 rng(123);
    std::uniform_real_distribution<> U(0.0, 1.0);
    for (int i = 0; i < N; i++) a[i] = static_cast<float>(U(rng));

    float max_val = 0.0f;

    double t0 = omp_get_wtime();

    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < N; i++) {
        if (a[i] > max_val) max_val = a[i];
    }

    double t1 = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        b[i] = a[i] / max_val;
    }

    double t2 = omp_get_wtime();

    std::cout << "Tempo max = " << (t1 - t0) << "s\n";
    std::cout << "Tempo normalização = " << (t2 - t1) << "s\n";
}
