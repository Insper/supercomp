#include <vector>
using namespace std;

void produtoReordenado(vector<double>& A, vector<double>& B, vector<double>& C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = 0.0;
    }

    for (int k = 0; k < N; k++) {
        double tempB = B[k];
        for (int i = 0; i < N; i++) {
            C[i] += A[i] * tempB;
        }
    }
}
