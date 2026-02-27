#include <iostream>
#include <vector>

int main() {
    std::vector<int> dados;

    dados.reserve(10000);  // pré-aloca espaço para 10.000 elementos

    for (int i = 0; i < 1000; ++i) {
        dados.push_back(i);
    }

    std::cout << "Tamanho: " << dados.size() << std::endl;
    std::cout << "Capacidade: " << dados.capacity() << std::endl;

    return 0;
}