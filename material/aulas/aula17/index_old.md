
Nesta aula, o objetivo é compreender como estruturas de dados e estratégias de execução assíncrona influenciam diretamente o desempenho de um algoritmo na GPU. 

## Lidando com Matrizes esparsas

Uma matriz esparsa é uma matriz na qual a maioria dos elementos é zero. Para processar dados com essa característica, é preciso levar em conta que armazenar todos os zeros seria um grande desperdício de memória e de tempo de processamento.

Por exemplo, imagine uma matriz 1.000 × 1.000.
Se apenas 10.000 elementos são diferentes de zero, isso significa que **99% dos elementos são zeros**.
Armazenar todos os 1.000.000 de elementos seria extremamente ineficiente, além de custar mais memória, na programação teriamos que elaborar estratégias para lidar com esse dado e evitar por exemplo, erros de divisões por zero.


Para resolver isso, surgiram representações que guardam apenas os elementos não nulos e suas posições. Existem vários formatos, CSR, COO, ELL, CSC, cada um com vantagens específicas dependendo da arquitetura e do tipo de acesso.

O formato CSR (Compressed Sparse Row), é o mais popular em ambientes de HPC, principalmente para operações de multiplicação matriz-vetor (SpMV).
Ele organiza os dados de forma compacta e cache-friendly, o que reduz o volume de memória transferido e melhora a eficiência de acesso, especialmente na GPU, onde o gerenciamento de acesso a memória é fundamental.

### Estrutura CSR 

![https://www.john-ros.com/Rcourse/sparse.html](image.png)
Fonte: https://www.john-ros.com/Rcourse/sparse.html



Na matriz 3×4 temos cinco elementos não nulos: $$ a_{01}, a_{02}, a_{11}, a_{13},   a_{20}  $$

Em vez de armazenar os doze elementos (incluindo zeros), o formato CSR guarda apenas as informações essenciais. O  vetor, chamado **Non-zeros**, contém os valores não nulos da matriz, organizados linha por linha. 

Para que seja possível saber a posição de cada valor dentro da matriz original, é necessário também registrar as colunas correspondentes. Isso é feito no vetor **Column Indices**, que contém o índice da coluna de cada elemento armazenado em **Non-zeros**.

Por fim, o vetor mais importante é o **Row Pointers**, que indica onde começa e termina cada linha dentro do vetor de valores. Ele tem tamanho igual ao número de linhas mais um, pois o último elemento marca o fim da última linha. No exemplo, o vetor é ([0, 2, 4, 5]). O zero inicial indica que a primeira linha começa no índice 0 do vetor **Non-zeros**; o número 2 indica que a segunda linha começa no índice 2; o número 4 marca o início da terceira linha; e o número 5 representa o limite final do vetor. Assim, ao percorrer a linha (i), basta ler os elementos entre `row_ptr[i]` e `row_ptr[i+1]`. Isso elimina a necessidade de armazenar zeros, mantendo uma correspondência precisa entre os valores e suas posições.

De forma prática, este é um exemplo de implementação:

```cpp
#include <iostream>
#include <cuda_runtime.h>

// Kernel CUDA para multiplicação matriz-vetor no formato CSR
__global__ void multiplicar_csr_kernel(
    int num_linhas,
    const int *ponteiro_linha,
    const int *indice_coluna,
    const float *valores,
    const float *vetor_x,
    float *vetor_y)
{
    int linha = blockIdx.x * blockDim.x + threadIdx.x; // Cada thread calcula uma linha

    if (linha < num_linhas) {
        float soma = 0.0f;

        int inicio = ponteiro_linha[linha];       // Índice inicial da linha
        int fim    = ponteiro_linha[linha + 1];   // Índice final (não incluso)

        for (int i = inicio; i < fim; i++) {
            int coluna = indice_coluna[i];
            soma += valores[i] * vetor_x[coluna];  // Multiplica valor não nulo pelo x correspondente
        }

        vetor_y[linha] = soma; // Armazena o resultado final
    }
}

int main() {
    // Considerando a matriz do exemplo:
    // [0  a01 a02  0]
    // [0  a11  0  a13]
    // [a20  0  0   0]

    int num_linhas = 3;
    int num_colunas = 4;
    int num_nao_nulos = 5;

    // Estruturas CSR no host (CPU)
    int h_ponteiro_linha[] = {0, 2, 4, 5};         // Indica o início de cada linha
    int h_indice_coluna[]  = {1, 2, 1, 3, 0};      // Colunas correspondentes
    float h_valores[]      = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}; // Valores não nulos
    float h_vetor_x[]      = {1.0f, 1.0f, 1.0f, 1.0f};        // Vetor x
    float h_vetor_y[3]     = {0.0f, 0.0f, 0.0f};              // Resultado y

    // Alocação de memória na GPU
    int *d_ponteiro_linha, *d_indice_coluna;
    float *d_valores, *d_vetor_x, *d_vetor_y;

    cudaMalloc(&d_ponteiro_linha, (num_linhas + 1) * sizeof(int));
    cudaMalloc(&d_indice_coluna, num_nao_nulos * sizeof(int));
    cudaMalloc(&d_valores, num_nao_nulos * sizeof(float));
    cudaMalloc(&d_vetor_x, num_colunas * sizeof(float));
    cudaMalloc(&d_vetor_y, num_linhas * sizeof(float));

    // Cópia dos dados da CPU para a GPU
    cudaMemcpy(d_ponteiro_linha, h_ponteiro_linha, (num_linhas + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indice_coluna, h_indice_coluna, num_nao_nulos * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valores, h_valores, num_nao_nulos * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vetor_x, h_vetor_x, num_colunas * sizeof(float), cudaMemcpyHostToDevice);

    // Configuração de blocos e threads
    int threads_por_bloco = 128;
    int blocos = (num_linhas + threads_por_bloco - 1) / threads_por_bloco;

    // Execução do kernel
    multiplicar_csr_kernel<<<blocos, threads_por_bloco>>>(
        num_linhas, d_ponteiro_linha, d_indice_coluna, d_valores, d_vetor_x, d_vetor_y);

    // Cópia do resultado da GPU para a CPU
    cudaMemcpy(h_vetor_y, d_vetor_y, num_linhas * sizeof(float), cudaMemcpyDeviceToHost);

    // Exibição do resultado
    std::cout << "Resultado y = A * x:" << std::endl;
    for (int i = 0; i < num_linhas; i++) {
        std::cout << "y[" << i << "] = " << h_vetor_y[i] << std::endl;
    }

    // Liberação de memória da GPU
    cudaFree(d_ponteiro_linha);
    cudaFree(d_indice_coluna);
    cudaFree(d_valores);
    cudaFree(d_vetor_x);
    cudaFree(d_vetor_y);

    return 0;
}
```

Cada linha da matriz é independente das outras, então podemos ter uma thread por linha.

Cada thread recebe um número de `linha` e calcula o resultado daquela linha, salvando no vetor de saída `y`.

Para saber onde estão os elementos da `linha` dentro da estrutura CSR, a thread consulta o vetor `ponteiro_linha`. Os valores `ponteiro_linha[linha]` e `ponteiro_linha[linha + 1]` marcam início e fim, indicando quais posições do vetor `valores` pertencem aquela linha.

A thread então percorre esse trecho da memória. Em `indice_coluna[i]`, aponta a coluna onde o valor está localizado na matriz original.

Em `valores[i]`, está o número não nulo que deve ser usado no cálculo.

Com essas informações, a thread sabe que deve multiplicar o valor `valores[i]` pelo elemento correspondente do vetor `x`.

Cada produto parcial é somado no acumulador `soma`.

Quando a thread termina de percorrer seus elementos, ela grava o resultado acumulado em `vetor_y[linha]`.

No final, todas as threads juntas produzem o vetor `y`, que contém o resultado da multiplicação `A * x`.



## Computação assíncrona com streams 

Até agora, o que vimos foi computação síncrona, com um fluxo de trabalho desta forma:

1. **Host → transfere os dados para GPU**
2. **Device → Executa o kernel**
3. **Device → transfere os dados para o Host**

Durante cada etapa, o hardware que não está envolvido permanece ocioso, a GPU espera a CPU passar os dados para fazer a computação, a CPU espera a GPU fazer a computação dos dados para ter os dados trabalhados.

A computação assíncrona é outra estratégia. Ela permite enfileirar tarefas para que a CPU não precise esperar o término das operações na GPU, e para que a GPU possa executar diferentes funções.

Com isso temos uma "sobreposição de operações", melhor aproveitamento do hardware e redução do tempo total de execução da aplicação.

## O que é uma Stream

Uma stream é uma fila de comandos assíncronos na GPU.
Cada stream mantém sua própria sequência de tarefas, como:

```
Stream 0: Memcpy (H2D) → Kernel → Memcpy (D2H)
Stream 1: Memcpy (H2D) → Kernel → Memcpy (D2H)
Stream 2: ...
```

As operações dentro de uma mesma stream são executadas na ordem em que foram emitidas (ordem FIFO).
Mas operações em streams diferentes podem ser executadas de forma concorrente na GPU.

Assim, enquanto uma stream executa um kernel, outra pode estar transferindo dados, e outra pode executar outro kernel.

Sua missão é planejar o envio dessas tarefas de modo que as dependências sejam respeitadas, mas que o máximo possível de operações aconteçam em paralelo.

![alt text](C1060Timeline-1024x679-1.png)
Fonte: https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/

Na versão sequencial, todas as operações ocorrem de forma estritamente ordenada: primeiro os dados são transferidos da CPU para a GPU, depois o kernel é executado e, em seguida, os resultados são copiados de volta para a CPU. 

Na Versão assíncrona I, o código passa a utilizar streams CUDA, que permitem a execução de tarefas em filas independentes, como o C1060 possui apenas um motor de cópia. As operações de transferência são executadas de forma alternada dentro do mesmo Copy Engine, essa sequência impede a sobreposição entre cópias e execução de kernels, pois o motor de cópia permanece ocupado o tempo todo alternando entre transferências de entrada e saída. 

Na segunda versão assíncrona, a ordem de emissão das operações é modificada: todas as transferências do host para o device são lançadas primeiro, seguidas da execução dos kernels e, por fim, das transferências de volta. Essa mudança de ordem permite que, enquanto o Kernel processa um conjunto de dados, o Copy Engine possa iniciar a transferência dos dados seguintes. Assim, parte das cópias e das execuções ocorre simultaneamente, o que reduz o tempo total de execução. 



Na prática, para implementar isso em CUDA, fazemos algo assim:

```cpp
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
kernel<<<blocks, threads, 0, stream1>>>(d_A, d_B, d_C);
cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);
```

Repetindo isso para `stream2`, `stream3`, etc, a GPU executa várias dessas sequências em paralelo.

Aqui temos o exemplo da matriz esparsa, agora, de forma assíncrona:


```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define NSTREAMS 6  // número de streams (blocos em paralelo)
#define LARGURA 3840
#define ALTURA 2160

// Kernel: inverte a intensidade do pixel (simula processamento de imagem)
__global__ void inverter_pixels(unsigned char *imagem, int tamanho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tamanho) {
        imagem[idx] = 255 - imagem[idx];
    }
}

// Função auxiliar para medir tempo entre eventos
float medir_tempo(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    srand(time(nullptr));

    const int largura = LARGURA;
    const int altura = ALTURA;
    const int tamanho_total = largura * altura;
    const size_t bytes_total = tamanho_total * sizeof(unsigned char);

    std::cout << "Imagem sintética 4K gerada: " 
              << largura << "x" << altura << " (" 
              << (bytes_total / (1024.0 * 1024.0)) << " MB)\n";

    // Cria imagem de entrada com dados aleatórios
    unsigned char *h_imagem = (unsigned char*)malloc(bytes_total);
    unsigned char *h_saida  = (unsigned char*)malloc(bytes_total);
    for (int i = 0; i < tamanho_total; i++)
        h_imagem[i] = rand() % 256;

    // Divide a imagem em blocos verticais
    const int linhas_por_stream = altura / NSTREAMS;
    const int pixels_por_stream = linhas_por_stream * largura;
    const size_t bytes_por_stream = pixels_por_stream * sizeof(unsigned char);

    unsigned char *d_blocos[NSTREAMS];
    cudaStream_t streams[NSTREAMS];

    for (int i = 0; i < NSTREAMS; i++) {
        cudaMalloc(&d_blocos[i], bytes_por_stream);
        cudaStreamCreate(&streams[i]);
    }

    // =====================
    // EXECUÇÃO SÍNCRONA
    // =====================
    cudaEvent_t start_sync, stop_sync;
    cudaEventCreate(&start_sync);
    cudaEventCreate(&stop_sync);
    cudaEventRecord(start_sync);

    for (int i = 0; i < NSTREAMS; i++) {
        unsigned char *h_chunk_in  = h_imagem + i * pixels_por_stream;
        unsigned char *h_chunk_out = h_saida  + i * pixels_por_stream;

        cudaMemcpy(d_blocos[i], h_chunk_in, bytes_por_stream, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (pixels_por_stream + threads - 1) / threads;
        inverter_pixels<<<blocks, threads>>>(d_blocos[i], pixels_por_stream);

        cudaMemcpy(h_chunk_out, d_blocos[i], bytes_por_stream, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop_sync);
    cudaEventSynchronize(stop_sync);
    float tempo_sync = medir_tempo(start_sync, stop_sync);

    // =====================
    // EXECUÇÃO ASSÍNCRONA
    // =====================
    cudaEvent_t start_async, stop_async;
    cudaEventCreate(&start_async);
    cudaEventCreate(&stop_async);
    cudaEventRecord(start_async);

    for (int i = 0; i < NSTREAMS; i++) {
        unsigned char *h_chunk_in  = h_imagem + i * pixels_por_stream;
        unsigned char *h_chunk_out = h_saida  + i * pixels_por_stream;

        cudaMemcpyAsync(d_blocos[i], h_chunk_in, bytes_por_stream, cudaMemcpyHostToDevice, streams[i]);

        int threads = 256;
        int blocks = (pixels_por_stream + threads - 1) / threads;
        inverter_pixels<<<blocks, threads, 0, streams[i]>>>(d_blocos[i], pixels_por_stream);

        cudaMemcpyAsync(h_chunk_out, d_blocos[i], bytes_por_stream, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Sincroniza todas as streams
    for (int i = 0; i < NSTREAMS; i++)
        cudaStreamSynchronize(streams[i]);

    cudaEventRecord(stop_async);
    cudaEventSynchronize(stop_async);
    float tempo_async = medir_tempo(start_async, stop_async);

    // =====================
    // RESULTADOS
    // =====================
    std::cout << "--------------------------------------------\n";
    std::cout << "Tempo SÍNCRONO   : " << tempo_sync  << " ms\n";
    std::cout << "Tempo ASSÍNCRONO : " << tempo_async << " ms\n";
    std::cout << "Speedup          : " << tempo_sync / tempo_async << "x\n";
    std::cout << "--------------------------------------------\n";

    std::cout << "Amostra de pixels (antes e depois):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "Pixel[" << i << "] " 
                  << (int)h_imagem[i] << " → " 
                  << (int)h_saida[i] << "\n";
    }

    // Libera recursos
    for (int i = 0; i < NSTREAMS; i++) {
        cudaFree(d_blocos[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start_sync);
    cudaEventDestroy(stop_sync);
    cudaEventDestroy(start_async);
    cudaEventDestroy(stop_async);
    free(h_imagem);
    free(h_saida);

    return 0;
}
```