#include <iostream>
#include <omp.h>
#include <unistd.h>


int main() {

    // Define o número máximo de threads
    omp_set_num_threads(4);

    // Executa código em paralelo
    #pragma omp parallel schedulle(static)
    {
        #pragma omp for
        for(int i = 0; i < 30; i++) {
            usleep(10 * omp_get_thread_num());
            std::cout << "Hello World (" << i << "," << omp_get_thread_num() << ")" << std::endl;
        }
    }

    return 0;

}