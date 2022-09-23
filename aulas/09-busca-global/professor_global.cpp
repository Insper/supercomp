/* knapsack */
#include <iostream>

using namespace std;
 
// A utility function that returns
// maximum of two integers
int max(int a, int b) { return (a > b) ? a : b; }
 
int knapSack(int W, int wt[], int val[], int n) {
 
    if (n == 0 || W == 0)
        return 0;
 
    if (wt[n - 1] > W)
        return knapSack(W, wt, val, n - 1);
 
    else
        return max(
            val[n - 1] + knapSack(W - wt[n - 1], wt, val, n - 1),
            knapSack(W, wt, val, n - 1));
}
 
int main() {

    int n, W;

    std::cin >> n; // quantidade de elementos
    std::cin >> W; // capacidade da mochila

    int *wt = new int[n]; // pesos
    int *val = new int[n]; // valores

    for(int i=0; i < n; i++) {
        std::cin >> wt[i];
        std::cin >> val[i];
    }

    std::cout << knapSack(W, wt, val, n);
    std::cout << std::endl;

    return 0;
}
 