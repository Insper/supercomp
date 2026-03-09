#include <iostream>
#include <omp.h>

struct No {
    int valor;
    No* esq;
    No* dir;
    No(int v) : valor(v), esq(nullptr), dir(nullptr) {}
};

int soma_arvore(No* raiz) {
    if (!raiz) return 0;
    int soma_esq = 0, soma_dir = 0;

    #pragma omp task shared(soma_esq)
    soma_esq = soma_arvore(raiz->esq);

    #pragma omp task shared(soma_dir)
    soma_dir = soma_arvore(raiz->dir);

    #pragma omp taskwait
    return raiz->valor + soma_esq + soma_dir;
}

int main() {
    No* raiz = new No(1);
    raiz->esq = new No(2);
    raiz->dir = new No(3);
    raiz->esq->esq = new No(4);
    raiz->esq->dir = new No(5);

    double t0 = omp_get_wtime();
    int soma = 0;
    #pragma omp parallel
    {
        #pragma omp single
        soma = soma_arvore(raiz);
    }
    double t1 = omp_get_wtime();

    std::cout << "Soma dos nós = " << soma << " tempo = " << (t1 - t0) << "s\n";
}
