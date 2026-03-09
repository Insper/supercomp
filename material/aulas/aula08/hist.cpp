//hist.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

// tamanho do vetor de dados
#define N 20000000

// quantidade de categorias do histograma
#define BINS 2

/*
Função que preenche o vetor com valores aleatórios
entre 0 e BINS-1
*/
void fill_data(std::vector<int> &data) {

    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, BINS-1);

    for (int i = 0; i < N; i++) {
        data[i] = dist(gen);
    }
}

/*
Zera o histograma antes de cada execução
*/
void reset_histogram(std::vector<int> &hist) {
    for (int i = 0; i < BINS; i++)
        hist[i] = 0;
}

/*
Versão usando ATOMIC
Cada incremento no histograma é protegido
por uma operação atômica.

Isso garante proteção contra condição de corrida, mas cria contenção
quando muitas threads tentam atualizar
o mesmo contador.
*/
double run_atomic(std::vector<int> &data, std::vector<int> &hist) {

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {

        int bin = data[i];


        #pragma omp atomic
        hist[bin]++;
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end-start).count();
}

/*
Versão usando CRITICAL

A região critical permite que apenas
uma thread por vez execute o bloco.

Isso resolve a condição de corrida,
mas pode criar um gargalo forte,
pois o acesso fica praticamente sequencial.
*/
double run_critical(std::vector<int> &data, std::vector<int> &hist) {

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {

        int bin = data[i];

        // apenas uma thread pode executar
        #pragma omp critical
        {
            hist[bin]++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end-start).count();
}

/*
Versão usando histogramas privados

Cada thread cria um histograma local,
evitando concorrência durante o processamento.

No final, os resultados são combinados.

Essa abordagem reduz drasticamente
a necessidade de sincronização.
*/
double run_private_merge(std::vector<int> &data, std::vector<int> &hist) {

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {

        // cada thread cria seu próprio histograma
        std::vector<int> local_hist(BINS, 0);

        // divide as iterações entre as threads
        #pragma omp for
        for (int i = 0; i < N; i++) {

            int bin = data[i];

            local_hist[bin]++;
        }

        // combinação final dos resultados
        #pragma omp critical
        {
            for (int i = 0; i < BINS; i++)
                hist[i] += local_hist[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end-start).count();
}

int main() {

    std::vector<int> data(N);
    std::vector<int> histogram(BINS);

    fill_data(data);

    // diferentes quantidades de threads para testar
    std::vector<int> thread_tests = {1,2,4,8,16};

    for (int threads : thread_tests) {

        omp_set_num_threads(threads);

        std::cout << "\nThreads: " << threads << std::endl;

        reset_histogram(histogram);
        double t_atomic = run_atomic(data, histogram);

        reset_histogram(histogram);
        double t_critical = run_critical(data, histogram);

        reset_histogram(histogram);
        double t_private = run_private_merge(data, histogram);

        std::cout << "atomic:   " << t_atomic << " s\n";
        std::cout << "critical: " << t_critical << " s\n";
        std::cout << "private:  " << t_private << " s\n";
    }

    return 0;
}