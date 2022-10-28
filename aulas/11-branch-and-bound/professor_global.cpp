/* knapsack */
#include <iostream>

using namespace std;

int num_leaf = 0;
int num_copy = 0;
int num_bounds = 0;
int level_bounds[200] = { 0 };

int melhor_solucao = 0;

int max(int a, int b) { return (a > b) ? a : b; }
 
int knap_sack(int W, int wt[], int val[], int n, int N, int valor, int estimativa) {
 
    // W eh a capacidade
    // N eh a quantidade de elementos

    // std::cout << "n = " << n;
    // std::cout << ", Estimativa = " << estimativa << std::endl;

    if(n == N || W == 0) {
        num_leaf++;
        // std::cout << "novo valor = " << valor << std::endl;
        if(valor > melhor_solucao) {
            melhor_solucao = valor;
            // std::cout << "atualizado melhor valor = " << melhor_solucao << std::endl;
            num_copy++;
        }
        return 0;
    }

    if(estimativa <= melhor_solucao) {
        // std::cout << "valor pior que estimativa! " << estimativa << std::endl;
        num_bounds++;
        level_bounds[n]++;
        return 0;
    }
 
    if(wt[n] > W) {

        return knap_sack(W, wt, val, n + 1, N, valor, estimativa - val[n]);

    } else { 

        return max(
            knap_sack(W - wt[n], wt, val, n + 1, N, valor + val[n], estimativa),
            knap_sack(W, wt, val, n + 1, N, valor, estimativa - val[n])
        );

    }
}
 
int main() {

    int N, W;

    std::cin >> N; // quantidade de elementos
    std::cin >> W; // capacidade da mochila

    int *wt = new int[N]; // pesos
    int *val = new int[N]; // valores

    int estimativa = 0;

    for(int i=0; i < N; i++) {
        std::cin >> wt[i];
        std::cin >> val[i];
        estimativa += val[i];
    }

    knap_sack(W, wt, val, 0, N, 0, estimativa);
    std::cout << "Melhor solução = " << melhor_solucao << std::endl;

    std::cout << "num_leaf " << num_leaf << std::endl;
    std::cout << "num_copy " << num_copy << std::endl;
    std::cout << "num_bounds " << num_bounds << std::endl;
    
    for(int i=0; i < N; i++) {
        std::cout << "\tbounds[" << i << "] = " << level_bounds[i] << std::endl;
    }

    return 0;
}
 