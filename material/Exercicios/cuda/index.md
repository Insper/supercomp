
### Exercício: Scan  

Implemente uma função de Scan em CUDA que calcula o scan do vetor de entrada, usando as fases *up-sweep* e *down-sweep*.


1. `scan(int *entrada, int *saida, int N)`
   

#### Exemplo de saída esperada:

```
[Tempo] Scan (prefix sum): 0.016908 ms
Scan parcial (20 primeiros valores):
0 1 3 6 10 15  21 28 36 45 55  66 78  91 105 120  136 153 171 190
Soma total com Scan: 524800
```

## Template de código

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>

// ========================================
// TODO: implementar kernel de scan (up-sweep + down-sweep)
// ========================================

__global__ void kernel_scan_exclusivo(int *entrada, int *saida, int N) {
    // TODO: implementar as fases up-sweep e down-sweep aqui
    // Utilize memória compartilhada (extern __shared__)
    // e sincronizações adequadas ( __syncthreads() )
}


// ========================================
// Função auxiliar: gera vetor [1..N]
// ========================================
std::vector<int> gerar_dados(int N) {
    std::vector<int> dados(N);
    std::iota(dados.begin(), dados.end(), 1);
    return dados;
}

// ========================================
// Função auxiliar: imprime primeiros elementos
// ========================================
void imprimir_prefixos(const std::vector<int> &v, int n = 20) {
    for (int i = 0; i < n && i < (int)v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "...\n";
}

// ========================================
// Função principal
// ========================================
int main(int argc, char **argv) {
    // --- Parâmetro N ---
    int N = 1024;  // valor padrão
    if (argc > 1) N = std::stoi(argv[1]); // permite alteração externa pelo autograder

    std::vector<int> entrada = gerar_dados(N);

    // --- Ponteiros GPU ---
    int *d_entrada = nullptr;
    int *d_saida   = nullptr;

    // TODO: alocar memória na GPU
    // TODO: copiar dados de entrada para a GPU

    // --- Medição de tempo ---
    auto inicio = std::chrono::high_resolution_clock::now();

    // TODO: chamada do kernel kernel_scan_exclusivo<<<...>>>(...);
    // TODO: usar cudaDeviceSynchronize();

    auto fim = std::chrono::high_resolution_clock::now();

    // --- Copia resultados para CPU ---
    std::vector<int> saida(N);
    // TODO: copiar resultados da GPU para a CPU

    // --- Impressão dos resultados ---
    std::chrono::duration<double, std::milli> tempo = fim - inicio;
    std::cout << "[Tempo] Scan (prefix sum): " << tempo.count() << " ms\n";
    std::cout << "Scan parcial (20 primeiros valores):\n";
    imprimir_prefixos(saida);

    // TODO: calcular e imprimir o valor total da soma com base no último prefixo
    // Dica: soma_total = saida[N-1] + entrada[N-1];

    // TODO: liberar memória da GPU
    return 0;
}

```

