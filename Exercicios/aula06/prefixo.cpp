#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    int N = 20;
    std::vector<int> a(N), p(N);

    for (int i = 0; i < N; i++) a[i] = 1;

    // Versão reformulada: progressão
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        p[i] = (i+1) * a[0];  // eliminada a dependência
    }

    for (int i = 0; i < N; i++) std::cout << p[i] << " ";
    std::cout << "\n";
}
