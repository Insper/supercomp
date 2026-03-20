#include <vector>
#include <algorithm>
using namespace std;

void produtoTiling(vector<double>& A, vector<double>& B, vector<double>& C, int N, int Bsize) {
    for (int i = 0; i < N; i++) {
        C[i] = 0.0;
    }

    for (int kk = 0; kk < N; kk += Bsize) {
        int kend = min(kk + Bsize, N);
        for (int i = 0; i < N; i++) {
            for (int k = kk; k < kend; k++) {
                C[i] += A[i] * B[k];
            }
        }
    }
}
