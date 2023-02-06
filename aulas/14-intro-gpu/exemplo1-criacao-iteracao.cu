#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

int main() {
    thrust::host_vector<double> host(5, 0);
    host[4] = 35;

    /* na linha abaixo os dados são copiados
       para GPU */
    thrust::device_vector<double> dev(host);    
    /* a linha abaixo só muda o vetor na CPU */
    host[2] = 12;
    
    printf("Host vector: ");
    for (auto i = host.begin(); i != host.end(); i++) {
        std::cout << *i << " "; // este acesso é rápido -- CPU
    }
    printf("\n");
    
    printf("Device vector: ");
    for (auto i = dev.begin(); i != dev.end(); i++) {
        std::cout << *i << " "; // este acesso é lento! -- GPU
    }
    printf("\n");

}
