#include <iostream>
#include <algorithm>
#include <omp.h>
#include <chrono>

#define ARRAY_SIZE 100000000

int main() {

    double *array = new double[ARRAY_SIZE];

    #pragma omp parallel for
    for(int i=0; i<ARRAY_SIZE; i++){
        array[i] = i;
    }
    
    double soma = 0;
    #pragma omp parallel for reduction(+:soma)
    for(int i=1; i<ARRAY_SIZE; i++){
      soma = soma + array[i];
    }

    std::cout << std::fixed << "Soma = " << soma << std::endl;

    return 0;

}