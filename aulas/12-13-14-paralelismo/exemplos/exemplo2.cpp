#include <iostream>
#include <algorithm>
#include <omp.h>
#include <chrono>


#define ARRAY_SIZE 400000000
#define ARRAY_VALUE 7

int main() {

    double *array = new double[ARRAY_SIZE];

    std::fill_n(array, ARRAY_SIZE, ARRAY_VALUE);

    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for(int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = array[i] / ( array[i] * array[i] );
    }
    auto stop = std::chrono::steady_clock::now();

    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000;

    std::cout << "Tempo em paralelo = " << delta << " ms" << std::endl;

    return 0;

}