#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

#define N 1000   
int main() {

    vector<vector<double>> A(N, vector<double>(N, 1.0));
    vector<vector<double>> B(N, vector<double>(N, 2.0));
    vector<vector<double>> C(N, vector<double>(N, 0.0));

    auto start = chrono::high_resolution_clock::now();

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {

                C.at(i).at(j) +=
                    A.at(i).at(k) * B.at(k).at(j);
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();

    cout << "Tempo (versao ingenua): "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    cout << "Valor C[0][0]: " << C[0][0] << endl;

    return 0;
}
