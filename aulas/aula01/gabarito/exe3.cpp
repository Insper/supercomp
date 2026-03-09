#include <iostream>
#include <vector>
#include <chrono>  // biblioteca para medir tempo

/**
 * Versão otimizada
 * Os vetores são passados por REFERÊNCIA constante.
 * Isso evita cópia e garante que não serão modificados.
 */
std::vector<int> soma_vetorial(const std::vector<int>& a,
                               const std::vector<int>& b){

    std::vector<int> resultado;

    for(size_t i = 0; i < a.size(); i++){
        resultado.push_back(a[i] + b[i]);
    }

    return resultado;
}

int main(){

    // Criamos dois vetores com 1000 posições
    std::vector<int> a;
    std::vector<int> b;

    for(int i = 0; i < 1000; i++){
        a.push_back(i);
        b.push_back(i * 2);
    }

    // ===== INÍCIO DA MEDIÇÃO =====
    auto inicio = std::chrono::high_resolution_clock::now();

    std::vector<int> resultado = soma_vetorial(a, b);

    auto fim = std::chrono::high_resolution_clock::now();
    // ===== FIM DA MEDIÇÃO =====

    std::chrono::duration<double, std::milli> tempo = fim - inicio;

    std::cout << "Tempo de execucao: "
              << tempo.count()
              << " ms\n";

    // Imprimimos apenas os 100 primeiros elementos
    for(int i = 0; i < 100; i++){
        std::cout << resultado[i] << " ";
    }

    std::cout << std::endl;

    return 0;
}
