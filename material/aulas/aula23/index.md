
Exercícios de aquecimento para a PF

### Multiplicação de Matrizes 

Reduza os acessos à memória global, aplicando a otimização com tilling e shared memory.

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

#define N 2048

__global__ void matMul(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {

        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main() {

    size_t size = N * N * sizeof(float);

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((N+15)/16, (N+15)/16);

    matMul<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Resultado: %f\n", C[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
```



### Estimativa de pi com Monte Carlo em CUDA

Aplique técnicas de otimização em GPU para reduzir em pelo menos 2x o tempo de execução deste algoritmo sem comprometer a precisão do resultado:

```cpp 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define THREADS 256
#define BLOCKS 256
#define SAMPLES_PER_THREAD 70000

int main() {

    /*
        Contador de pontos que ficaram dentro
        do círculo unitário.
    */
    unsigned long long count = 0;

    /*
        Calcula o total de amostras que serão geradas.
    */
    unsigned long long total_samples =
        (unsigned long long)THREADS *
        BLOCKS *
        SAMPLES_PER_THREAD;

    /*
        Inicializa o gerador de números aleatórios.
    */
    srand(time(NULL));

    // Marca o tempo inicial
    clock_t start = clock();

    for (unsigned long long i = 0; i < total_samples; i++) {

        /*
            Gera coordenadas aleatórias entre 0 e 1.
        */
        float x =
            (float)rand() / (float)RAND_MAX;

        float y =
            (float)rand() / (float)RAND_MAX;

        /*
            Calcula a distância até a origem.

            Uso de sqrt() é desnecessário,
            pois poderíamos comparar apenas
            a distância ao quadrado.

            Isso torna o código mais lento.
        */
        float dist =
            sqrt((x * x) + (y * y));

        /*
            Verifica se o ponto está dentro
            do círculo unitário.
        */
        if (dist <= 1.0f) {
            count++;
        }
    }

    // Marca o tempo final
    clock_t end = clock();

    /*
        Estimativa de PI usando Monte Carlo.

            pi ≈ 4 * pontos_dentro / total
    */
    double pi =
        4.0 * ((double)count / (double)total_samples);

    // Calcula o tempo de execução
    double elapsed =
        (double)(end - start) / CLOCKS_PER_SEC;

    // Exibe os resultados
    printf("Estimativa de PI : %.4f\n", pi);
    printf("Pontos dentro    : %llu\n", count);
    printf("Total de pontos  : %llu\n", total_samples);
    printf("Tempo total      : %.4f segundos\n", elapsed);

    return 0;
}
```

### Exercício: Busca de Nonce 


Otimize o código aplicando técnicas de otimização em GPU

```cpp 
// ============================================================
// Simulação simples de mineração de blockchain
//
// O programa:
//   1. Gera hashes a partir de um bloco + nonce
//   2. Procura hashes com zeros iniciais
//   3. Mede o tempo necessário para minerar
//
// Implementação sequencial em CPU.
// ============================================================

#include <iostream>   // Entrada e saída (cout)
#include <string>     // Manipulação de strings
#include <chrono>     // Medição de tempo
#include <functional> // std::hash
#include <sstream>    // stringstream
#include <iomanip>    // setw, setfill

/*
    Define a dificuldade da mineração.
*/
#define DIFICULDADE 7

/*
    Função responsável por gerar um hash hexadecimal.

    Recebe:
        uma string de entrada

    Retorna:
        uma string hexadecimal simulando um hash.
*/
std::string gerarHash(const std::string& entrada) {

    /*
        std::hash é uma função de hash da biblioteca padrão.

        NÃO é SHA-256 real, mas serve para simular
        mineração de blockchain.
    */
    std::hash<std::string> hash_fn;

    /*
        Calcula o hash numérico da entrada.
    */
    size_t valor = hash_fn(entrada);

    /*
        stringstream será usado para converter
        o valor numérico para hexadecimal.
    */
    std::stringstream ss;

    /*
        std::hex:
            converte para hexadecimal

        std::setw(16):
            largura mínima de 16 caracteres

        std::setfill('0'):
            completa com zeros à esquerda
    */
    ss << std::hex
       << std::setw(16)
       << std::setfill('0')
       << valor;

    /*
        Retorna o hash em formato string.
    */
    return ss.str();
}

/*
    Verifica se o hash possui
    determinada quantidade de zeros iniciais.
*/
bool hashValido(
    const std::string& hash,
    int dificuldade
) {

    /*
        Percorre os primeiros caracteres do hash.
    */
    for (int i = 0; i < dificuldade; i++) {

        /*
            Se algum caractere não for '0',
            o hash não é válido.
        */
        if (hash[i] != '0') {
            return false;
        }
    }

    return true;
}

int main() {

    /*
        Conteúdo do bloco.

        Em uma blockchain real poderia conter:
            - transações
            - timestamp
            - hash anterior
            - etc.
    */
    std::string bloco =
        "Transacao A -> B";

    /*
        Nonce inicial.

        O nonce será incrementado continuamente
        até encontrar um hash válido.
    */
    int nonce = 0;

    // Armazena o hash calculado
    std::string hash;

    /*
        Marca o instante inicial da mineração.
    */
    auto inicio =
        std::chrono::high_resolution_clock::now();

    /*
        Loop principal da mineração.

        A cada iteração:
            bloco + nonce -> hash
    */
    while (true) {

        /*
            Concatena o bloco com o nonce.

        */
        std::string tentativa =
            bloco + std::to_string(nonce);

        /*
            Calcula o hash da tentativa atual.
        */
        hash = gerarHash(tentativa);

        /*
            Verifica se o hash atende
            à dificuldade.
        */
        if (hashValido(hash, DIFICULDADE)) {

            /*
                Marca o instante final.
            */
            auto fim =
                std::chrono::high_resolution_clock::now();

            /*
                Calcula o tempo total
                de mineração em segundos.
            */
            double tempo =
                std::chrono::duration<double>(
                    fim - inicio
                ).count();

            // Exibe informações do bloco minerado
            std::cout << "Bloco minerado!\n";

            // Nonce encontrado
            std::cout
                << "Nonce: "
                << nonce
                << "\n";

            // Hash válido encontrado
            std::cout
                << "Hash : "
                << hash
                << "\n";

            // Tempo gasto na mineração
            std::cout
                << "Tempo: "
                << tempo
                << " segundos\n";

            /*
                Encerra a mineração.
            */
            break;
        }

        /*
            Testa o próximo nonce.
        */
        nonce++;
    }

    return 0;
}
```

