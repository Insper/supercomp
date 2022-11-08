#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <iostream>

int main() {
    // Alocação do vetor na CPU e leitura de dados da entrada-padrão
    thrust::host_vector<double> vcpu(2518);
    for(int i=0;i<2518;i++)
       std::cin>>vcpu[i];

    // Alocação do vetor na GPU e inicialização de dados
    thrust::device_vector<double> vgpu(vcpu); 

    // Percurso do vetor na CPU - RÁPIDO
    std::cout <<"CPU: ";
    for (thrust::host_vector<double>::iterator elem = vcpu.begin(); elem != vcpu.end(); elem++) {
        std::cout << *elem << " ";
    }

    // Maneira mais compacta de percorrer o vetor
    std::cout <<"\nCPU: ";
    thrust::copy(vcpu.begin(), vcpu.end(), std::ostream_iterator<double>(std::cout, " "));

   
    // Percurso do vetor na GPU - LENTO
    std::cout <<"\nGPU: ";
    for (thrust::device_vector<double>::iterator elem = vgpu.begin(); elem != vgpu.end(); elem++) {
        std::cout << *elem << " ";
    } 

    // Soma todos os elementos do vetor na GPU
    double soma=thrust::reduce(vgpu.begin(), vgpu.end(), (double) 0, thrust::plus<double>());
    std::cout<<"\nSoma: "<< soma;
}
