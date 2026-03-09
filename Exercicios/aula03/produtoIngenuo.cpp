#include <vector>
using namespace std;

void produtoIngenuo(vector<double>& A, vector<double>& B, vector<double>& C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = 0.0;
        for (int k = 0; k < N; k++) {
            C[i] += A[i] * B[k];
        }
    }
}
