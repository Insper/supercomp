# Código Síncrono vs Assíncrono em CUDA

## Objetivo da Aula

Mostrar, de forma visual e mensurável, a diferença entre:

* **execução síncrona** (um kernel ou cópia de memória por vez)
* **execução assíncrona com *streams*** (múltiplas operações em paralelo)


## Preparação da conexão ssh com suporte ao X11


Não esqueça de acessar o Cluster Franky com suporte a interface gráfica:

```bash
ssh -X usuario@head-node
```

Verifique se o redirecionamento X11 está ativo:

```bash
xclock
```

Deve aparecer um reloginho na sua tela, como esse:

![alt text](relogio.png)

Se você está em um ambiente Windows e não quer o usar o WSL2. Acesse o cluster pelo MobaXterm, não sabe do que eu estou falando? [O aplicativo é esse aqui](https://mobaxterm.mobatek.net/download-home-edition.html)


## Profiling em GPU

O termo *profiling* é o processo de observar, registrar e analisar o comportamento de um programa enquanto ele é executado. Quando escrevemos aplicações CUDA, o código-fonte mostra o que queremos fazer, mas não revela como a GPU realmente executa as instruções, quanto tempo cada operação consome ou onde estão os gargalos. O profiling é justamente a ferramenta que nos permite “abrir a caixa preta” e enxergar o que está acontecendo dentro da GPU em tempo real.

Em um ambiente de HPC, medir e entender o desempenho é tão importante quanto escrever o código. Um kernel pode estar correto e ainda assim ser ineficiente, gastando mais tempo em cópias de memória do que em processamento, ou desperdiçando paralelismo por má organização das threads. 

A Nvidia tem duas ferramentas de profiling para CPU, O **Nsight Systems `nsys`** fornece uma visão global da aplicação: ele registra a linha do tempo da execução, mostrando todas as chamadas CUDA, transferências de dados entre CPU e GPU, sincronizações e kernels. Ele é útil para entender o fluxo entre CPU e GPU e visualizar quando as operações acontecem de forma sequencial ou paralela. 

Para começar, é preciso compilar o código:


```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define LARGURA 3840
#define ALTURA  2160

// Kernel simples: inverte a intensidade do pixel
__global__ void inverter_pixels(unsigned char *imagem, int tamanho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tamanho)
        imagem[idx] = 255 - imagem[idx];
}

// Função auxiliar para medir tempo
float medir_tempo(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    srand(time(nullptr));

    const int largura = LARGURA;
    const int altura  = ALTURA;
    const int tamanho_total = largura * altura;
    const size_t bytes_total = tamanho_total * sizeof(unsigned char);

    std::cout << "Imagem sintética 4K gerada: "
              << largura << "x" << altura
              << " (" << (bytes_total / (1024.0 * 1024.0)) << " MB)\n";

    // ===============================
    // Alocação de memória no host
    // ===============================
    unsigned char *h_imagem = (unsigned char*)malloc(bytes_total);
    unsigned char *h_saida  = (unsigned char*)malloc(bytes_total);

    for (int i = 0; i < tamanho_total; i++)
        h_imagem[i] = rand() % 256;

    // ===============================
    // Alocação de memória na GPU
    // ===============================
    unsigned char *d_imagem;
    cudaMalloc(&d_imagem, bytes_total);

    // ===============================
    // Medição com eventos CUDA
    // ===============================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ===============================
    // Cópia síncrona CPU → GPU
    // ===============================
    cudaMemcpy(d_imagem, h_imagem, bytes_total, cudaMemcpyHostToDevice);

    // ===============================
    // Execução do kernel
    //=============================
    int threads = 256;
    int blocks = (tamanho_total + threads - 1) / threads;

    inverter_pixels<<<blocks, threads>>>(d_imagem, tamanho_total);

    // Espera o kernel terminar
    cudaDeviceSynchronize();

    // ===============================
    //  Cópia síncrona GPU → CPU
    // ===============================
    cudaMemcpy(h_saida, d_imagem, bytes_total, cudaMemcpyDeviceToHost);

    // Marca o fim da região medida
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempo_ms = medir_tempo(start, stop);

    // ===============================
    // Resultados
    // ===============================
    std::cout << "--------------------------------------------\n";
    std::cout << "execução síncrona\n";
    std::cout << "Tempo total: " << tempo_ms << " ms\n";
    std::cout << "--------------------------------------------\n";

    for (int i = 0; i < 10; i++) {
        std::cout << "Pixel[" << i << "] "
                  << (int)h_imagem[i] << " → "
                  << (int)h_saida[i] << "\n";
    }

    // ===============================
    // Liberação de recursos
    // ===============================
    cudaFree(d_imagem);
    free(h_imagem);
    free(h_saida);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

```

Não se esqueça de carregar o módulo cuda
```bash
module load cuda/12.8.1
```

```bash
nvcc -O3 sincrono.cu -o sincrono
```

Para executar o binário no Cluster Franky, com suporte a GPU é assim:

```bash
srun --partition=merry_gpu --gres=gpu:1 ./sincrono
```

Para analisar o comportamento em detalhe, com suporte ao profiler, o comando fica assim:

```bash
srun --partition=merry_gpu --gres=gpu:1 nsys profile -o relatorio_sincrono --stats=true --trace=cuda,osrt ./sincrono
```

`--trace=cuda` Captura todas as chamadas CUDA: `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`, etc.

`--trace=osrt` Captura chamadas de sistema (CPU threads, sincronizações, alocações).                

`--stats=true`  Gera um relatório resumido no terminal                                               


A saída vai ser algo parecido com isso:

![alt text](output.png)

Se quiser ver na interface gráfica, o comando é esse:

```bash
nsys-ui relatorio_sincrono.nsys-rep
```

![alt text](image-1.png)

Este código não tem otimizações, é apenas um código que foi portado pra GPU de forma síncrona e sem cuidados com o gerenciamento dos dados, agora vamos aplicar algumas otimizações:


```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define NSTREAMS 2   // número de streams paralelas
#define LARGURA 3840
#define ALTURA  2160

// =======================================================
// Kernel simples: inverte a intensidade do pixel
// =======================================================
__global__ void inverter_pixels(unsigned char *imagem, int tamanho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tamanho)
        imagem[idx] = 255 - imagem[idx];
}

// =======================================================
// Mede tempo entre eventos CUDA
// =======================================================
float medir_tempo(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int largura = LARGURA;
    const int altura  = ALTURA;
    const int tamanho_total = largura * altura;
    const size_t bytes_total = static_cast<size_t>(tamanho_total) * sizeof(unsigned char);

    std::cout << "Imagem sintética 4K gerada: "
              << largura << "x" << altura << " ("
              << (bytes_total / (1024.0 * 1024.0)) << " MB)\n";

    // =======================================================
    // 1) Alocação de memória no host 
    // =======================================================
    unsigned char *h_imagem = static_cast<unsigned char*>(std::malloc(bytes_total));
    unsigned char *h_saida  = static_cast<unsigned char*>(std::malloc(bytes_total));

    for (int i = 0; i < tamanho_total; ++i)
        h_imagem[i] = static_cast<unsigned char>(std::rand() % 256);

    // =======================================================
    // 2) Divisão da imagem entre as streams
    // =======================================================
    const int linhas_por_stream   = altura / NSTREAMS;
    const int pixels_por_stream   = linhas_por_stream * largura;
    const size_t bytes_por_stream = static_cast<size_t>(pixels_por_stream) * sizeof(unsigned char);

    unsigned char *d_blocos[NSTREAMS];
    cudaStream_t  streams[NSTREAMS];

    for (int i = 0; i < NSTREAMS; ++i) {
        cudaMalloc(&d_blocos[i], bytes_por_stream);
        cudaStreamCreate(&streams[i]);
    }

    // =======================================================
    // 3) Criação de eventos para medir o tempo total e por stream
    // =======================================================
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total);

    cudaEvent_t start_stream[NSTREAMS], stop_stream[NSTREAMS];
    for (int i = 0; i < NSTREAMS; ++i) {
        cudaEventCreate(&start_stream[i]);
        cudaEventCreate(&stop_stream[i]);
    }

    // =======================================================
    // 4) Execução assíncrona
    // =======================================================
    for (int i = 0; i < NSTREAMS; ++i) {
        unsigned char *h_chunk_in  = h_imagem + i * pixels_por_stream;
        unsigned char *h_chunk_out = h_saida  + i * pixels_por_stream;

        cudaEventRecord(start_stream[i], streams[i]);  // início da stream

        // Cópia H2D assíncrona
        cudaMemcpyAsync(d_blocos[i], h_chunk_in, bytes_por_stream,
                        cudaMemcpyHostToDevice, streams[i]);

        // Kernel por stream
        int threads = 256;
        int blocks  = (pixels_por_stream + threads - 1) / threads;
        inverter_pixels<<<blocks, threads, 0, streams[i]>>>(d_blocos[i], pixels_por_stream);

        // Cópia D2H assíncrona
        cudaMemcpyAsync(h_chunk_out, d_blocos[i], bytes_por_stream,
                        cudaMemcpyDeviceToHost, streams[i]);

        cudaEventRecord(stop_stream[i], streams[i]);  // fim da stream
    }

    // =======================================================
    // 5) Sincronização geral
    // =======================================================
    for (int i = 0; i < NSTREAMS; ++i)
        cudaStreamSynchronize(streams[i]);

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    float tempo_total = medir_tempo(start_total, stop_total);

    // =======================================================
    // 6) Resultados
    // =======================================================
    std::cout << "--------------------------------------------\n";
    std::cout << "Tempo total (assíncrono, host pageable): " << tempo_total << " ms\n";
    std::cout << "--------------------------------------------\n";

    for (int i = 0; i < NSTREAMS; ++i) {
        float tempo_stream = medir_tempo(start_stream[i], stop_stream[i]);
        std::cout << "Stream " << i << " → tempo: " << tempo_stream << " ms\n";
    }

    std::cout << "--------------------------------------------\n";
    std::cout << "Amostra de pixels (antes e depois):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "Pixel[" << i << "] "
                  << static_cast<int>(h_imagem[i]) << " → "
                  << static_cast<int>(h_saida[i]) << "\n";
    }

    // =======================================================
    // 7) Liberação de recursos
    // =======================================================
    for (int i = 0; i < NSTREAMS; ++i) {
        cudaFree(d_blocos[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(start_stream[i]);
        cudaEventDestroy(stop_stream[i]);
    }

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    std::free(h_imagem);
    std::free(h_saida);

    return 0;
}

```


Vamos compilar
```bash
nvcc -O3 assincrono.cu -o assincrono
```

Vamos gerar o relatório:

```bash
srun --partition=merry_gpu --gres=gpu:1 nsys profile -o relatorio_assincrono --stats=true --trace=cuda,osrt ./assincrono
```


Se você abrir esse relatório no Nsight UI, verá exatamente isso visualmente:


```bash
nsys-ui relatorio_assincrono.nsys-rep 
```

O código não melhorou, é preciso ajustar um detalhe muito importante!


## Memória “pinned” (page-locked)

Em sistemas operacionais modernos, a memória RAM do host (CPU) é gerenciada virtualmente pelo sistema operacional.
Isso quer dizer que o sistema pode mover páginas de memória entre RAM e disco (swap), e pode realocar fisicamente os blocos conforme necessário.
Essa flexibilidade é ótima para multitarefa, mas traz um problema para a GPU: o DMA (Direct Memory Access) da placa precisa saber endereços físicos fixos para transferir dados diretamente da RAM para a memória da GPU.

Por padrão, quando você usa uma alocação comum em C++:

```cpp
unsigned char* h_imagem = (unsigned char*)malloc(bytes_total);
```

essa memória é “pageable”, ou seja, o sistema operacional pode movê-la a qualquer momento.
Quando a GPU precisa copiar dados dessa memória, o driver CUDA não consegue usar DMA direto, porque não tem garantia de que aqueles dados continuarão no mesmo lugar.

Então o que o driver faz é criar uma cópia intermediária temporária:

1. Ele aloca um buffer “pinned” interno, invisível ao programador;
2. Copia seus dados do buffer “pageable” para esse buffer “pinned”;
3. Só então o DMA da GPU transfere do buffer “pinned” para a VRAM.

Esse caminho duplica o número de cópias e adiciona latência, o que significa que `cudaMemcpy` e `cudaMemcpyAsync` ficam mais lentos e menos eficientes.


## O que muda com `cudaMallocHost`

Quando você usa:

```cpp
cudaMallocHost(&h_imagem, bytes_total);
```

ou:

```cpp
cudaHostAlloc(&h_imagem, bytes_total, cudaHostAllocDefault);
```

o driver CUDA solicita ao sistema operacional que fixe fisicamente essas páginas de memória na RAM.

Com isso, as transferências `cudaMemcpy` e `cudaMemcpyAsync` se tornam muito mais rápidas.
Mais importante ainda: as transferências assíncronas realmente funcionam como assíncronas.

Sem memória *pinned*, o comando:

```cpp
cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream);
```

não é verdadeiramente assíncrono, o driver precisa bloquear temporariamente a CPU para copiar os dados da memória pageable para o buffer pinned interno.
Na prática, o código se comporta quase como uma cópia síncrona.

Quando o *host buffer* é *pinned*, o `cudaMemcpyAsync` não precisa dessa etapa:
a GPU inicia a transferência imediatamente, em paralelo à execução dos kernels, utilizando o mecanismo dedicado.
É exatamente isso que permite sobreposição real de cópia e execução.


## Cuidados e boas práticas

1. **Não use pinned memory em excesso**:
   Cada alocação *pinned* reduz a quantidade de memória que o sistema operacional pode gerenciar dinamicamente.
   Se você fixar demais, pode degradar o desempenho global da máquina (especialmente se houver swap ativado).

2. **Ideal para buffers de I/O**:
   Use *pinned memory* apenas para regiões de dados que serão realmente transferidas para a GPU.
   Em outras palavras: é para buffers de entrada e saída, não para todo o estado do programa.

3. **Libere corretamente**:
   Use `cudaFreeHost(ptr)` (ou `cudaFree(ptr)` se foi alocado com `cudaMallocManaged`) para liberar a memória.
   Isso garante que o driver devolva o controle das páginas ao sistema operacional.



Vamos aplicar a pinned memory no código para ele ficar assíncrono de verdade:

```cpp
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>

constexpr int LARGURA = 3840;
constexpr int ALTURA = 2160;
constexpr int NSTREAMS = 2;
constexpr int TAMANHO_CHUNK = 1024 * 1024; // 1 MiB, um byte por pixel

__global__ void inverter_pixels(unsigned char* imagem, int tamanho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tamanho)
        imagem[idx] = 255 - imagem[idx];
}

// Cada stream reutiliza seu próprio trecho da memória de GPU.
// A ordem H2D -> kernel -> D2H na stream protege o reúso do buffer.
void processar_imagem(unsigned char* entrada, unsigned char* saida,
                     unsigned char* dispositivo, int tamanho,
                     cudaStream_t* streams) {
    for (int inicio = 0; inicio < tamanho;
         inicio += NSTREAMS * TAMANHO_CHUNK) {
        int ativos = std::min(NSTREAMS,
            (tamanho - inicio + TAMANHO_CHUNK - 1) / TAMANHO_CHUNK);

        // Envia os chunks do grupo antes de enfileirar seus retornos.
        for (int s = 0; s < ativos; ++s) {
            int offset = inicio + s * TAMANHO_CHUNK;
            int quantidade = std::min(TAMANHO_CHUNK, tamanho - offset);
            cudaMemcpyAsync(dispositivo + s * TAMANHO_CHUNK,
                            entrada + offset, quantidade,
                            cudaMemcpyHostToDevice, streams[s]);
        }

        for (int s = 0; s < ativos; ++s) {
            int offset = inicio + s * TAMANHO_CHUNK;
            int quantidade = std::min(TAMANHO_CHUNK, tamanho - offset);
            int blocos = (quantidade + 255) / 256;
            inverter_pixels<<<blocos, 256, 0, streams[s]>>>(
                dispositivo + s * TAMANHO_CHUNK, quantidade);
        }

        for (int s = 0; s < ativos; ++s) {
            int offset = inicio + s * TAMANHO_CHUNK;
            int quantidade = std::min(TAMANHO_CHUNK, tamanho - offset);
            cudaMemcpyAsync(saida + offset,
                            dispositivo + s * TAMANHO_CHUNK, quantidade,
                            cudaMemcpyDeviceToHost, streams[s]);
        }
    }
}

int main() {
    const int tamanho = LARGURA * ALTURA;
    const size_t bytes = tamanho * sizeof(unsigned char);

    // 1. Aloca os buffers uma única vez.
    unsigned char* h_imagem = nullptr;
    unsigned char* h_saida = nullptr;
    unsigned char* d_chunks = nullptr;

    cudaMallocHost(reinterpret_cast<void**>(&h_imagem), bytes);
    cudaMallocHost(reinterpret_cast<void**>(&h_saida), bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_chunks),
               NSTREAMS * TAMANHO_CHUNK * sizeof(unsigned char));

    std::srand(1337);
    for (int i = 0; i < tamanho; ++i)
        h_imagem[i] = std::rand() % 256;

    cudaStream_t streams[NSTREAMS];
    cudaEvent_t inicio_stream[NSTREAMS], fim_stream[NSTREAMS];

    for (int s = 0; s < NSTREAMS; ++s) {
        cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
        cudaEventCreate(&inicio_stream[s]);
        cudaEventCreate(&fim_stream[s]);
    }

    // 2. Aquece o kernel e as transferências antes da medição.
    // Os mesmos buffers serão usados na execução medida.
    processar_imagem(h_imagem, h_saida, d_chunks, tamanho, streams);
    for (int s = 0; s < NSTREAMS; ++s)
        cudaStreamSynchronize(streams[s]);

    // 3. Mede uma imagem, incluindo envio, processamento e retorno.
    const auto inicio = std::chrono::steady_clock::now();

    for (int s = 0; s < NSTREAMS; ++s)
        cudaEventRecord(inicio_stream[s], streams[s]);

    processar_imagem(h_imagem, h_saida, d_chunks, tamanho, streams);

    for (int s = 0; s < NSTREAMS; ++s)
        cudaEventRecord(fim_stream[s], streams[s]);

    // A CPU só lê os resultados depois que todas as streams terminam.
    for (int s = 0; s < NSTREAMS; ++s)
        cudaStreamSynchronize(streams[s]);

    const auto fim = std::chrono::steady_clock::now();
    const double tempo_total =
        std::chrono::duration<double, std::milli>(fim - inicio).count();

    // 4. Exibe os resultados no padrão dos exemplos do handout.
    std::cout << "Imagem sintética 4K gerada: "
              << LARGURA << "x" << ALTURA << " ("
              << bytes / (1024.0 * 1024.0) << " MB)\n"
              << "--------------------------------------------\n"
              << "Tempo total (assíncrono + pinned): "
              << tempo_total << " ms\n"
              << "--------------------------------------------\n";

    for (int s = 0; s < NSTREAMS; ++s) {
        float tempo = 0;
        cudaEventElapsedTime(&tempo, inicio_stream[s], fim_stream[s]);
        std::cout << "Stream " << s << " → tempo: " << tempo << " ms\n";
    }

    std::cout << "--------------------------------------------\n"
              << "Amostra de pixels (antes e depois):\n";
    for (int i = 0; i < 10; ++i)
        std::cout << "Pixel[" << i << "] " << static_cast<int>(h_imagem[i])
                  << " → " << static_cast<int>(h_saida[i]) << "\n";

    // 5. Libera os recursos após a conclusão de todas as operações.
    for (int s = 0; s < NSTREAMS; ++s) {
        cudaEventDestroy(inicio_stream[s]);
        cudaEventDestroy(fim_stream[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_chunks);
    cudaFreeHost(h_imagem);
    cudaFreeHost(h_saida);
    return 0;
}

```

Vamos compilar
```bash
nvcc -O3 otimizado.cu -o otimizado
```

Vamos gerar o relatório:

```bash
srun --partition=merry_gpu --gres=gpu:1 nsys profile -o relatorio_otimizado --stats=true --trace=cuda,osrt ./otimizado
```


Para abrir o relatório no Nsight UI:


```bash
nsys-ui relatorio_otimizado.nsys-rep 
```

A otimização focou no gerenciamento de memória e na organização das transferências entre CPU e GPU. A versão assíncrona utilizava `cudaMemcpyAsync` e duas streams, mas os buffers da CPU eram alocados com `malloc`, em memória pageable. Na versão otimizada, esses buffers passaram a usar `cudaMallocHost`, que aloca memória pinned e permite enfileirar as transferências sem a etapa intermediária de preparação dos dados pageable.

A imagem também passou a ser processada em chunks de até 1 MiB. Cada stream utiliza um buffer próprio na GPU e o reutiliza para os próximos chunks. A ordem de execução dentro da stream garante que cada envio aconteça depois do retorno do chunk anterior, protegendo a reutilização do buffer. Com duas streams, a alocação na GPU caiu de aproximadamente **7,91 MiB para 2 MiB**. Os buffers e as streams são criados uma única vez, e a CPU aguarda sua conclusão antes de acessar os resultados.

Os tempos observados mostram uma redução no intervalo de processamento da imagem.

| Versão | Tempo informado |
|---|---:|
| Assíncrona original, com memória pageable | 4,609 ms |
| Otimizada, com memória pinned e chunks | 0,972 ms |

Essa diferença corresponde a uma redução de **78,9%**, ou aceleração de aproximadamente **4,74×**. 


## SUA VEZ!!! 

O código a seguir simula um processamento de imagem em GPU, onde 100 imagens são processadaa por duas funções distintas, uma que inverte os pixels e outra que altera a saturação.

Apesar de funcionar corretamente, o programa apresenta baixo desempenho e uso ineficiente da GPU.

**Sua tarefa é otimizar o código para criar um pipeline assíncrono. O objetivo é sobrepor transferências e processamento de blocos diferentes, respeitando as dependências entre as operações.**

### Código base

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>
#include <cmath>

constexpr int LARGURA = 3840;
constexpr int ALTURA = 2160;
constexpr int NUM_IMAGENS = 100;
constexpr int CANAIS = 3; // RGB
constexpr float FATOR_SATURACAO = 1.5f;

__global__ void inverter_pixels(unsigned char* imagem, int tamanho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tamanho)
        imagem[idx] = 255 - imagem[idx];
}

// Cada thread ajusta um pixel RGB.
// Fator 0 gera cinza, 1 preserva as cores e valores maiores intensificam.
__global__ void alterar_saturacao(unsigned char* imagem, int pixels,
                                float fator) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixels) {
        int base = idx * CANAIS;
        float r = imagem[base];
        float g = imagem[base + 1];
        float b = imagem[base + 2];
        float cinza = 0.299f * r + 0.587f * g + 0.114f * b;

        r = cinza + fator * (r - cinza);
        g = cinza + fator * (g - cinza);
        b = cinza + fator * (b - cinza);

        imagem[base] = static_cast<unsigned char>(
            fminf(255.0f, fmaxf(0.0f, r)));
        imagem[base + 1] = static_cast<unsigned char>(
            fminf(255.0f, fmaxf(0.0f, g)));
        imagem[base + 2] = static_cast<unsigned char>(
            fminf(255.0f, fmaxf(0.0f, b)));
    }
}

int main() {
    std::srand(1337);

    const int pixels = LARGURA * ALTURA;
    const int tamanho = pixels * CANAIS;
    const size_t bytes = tamanho * sizeof(unsigned char);
    const int threads = 256;
    const int blocos_inversao = (tamanho + threads - 1) / threads;
    const int blocos_saturacao = (pixels + threads - 1) / threads;
    double tempo_total = 0.0;

    for (int imagem = 0; imagem < NUM_IMAGENS; ++imagem) {
        // Gera uma imagem fake  na CPU.
        unsigned char* h_imagem =
            static_cast<unsigned char*>(std::malloc(bytes));
        unsigned char* h_saida =
            static_cast<unsigned char*>(std::malloc(bytes));

        for (int i = 0; i < tamanho; ++i)
            h_imagem[i] = std::rand() % 256;

        // Mede também a alocação e a liberação da memória da GPU.
        const auto inicio = std::chrono::steady_clock::now();

        // 1. Aloca memória na GPU novamente para cada imagem.
        unsigned char* d_imagem = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d_imagem), bytes);

        // 2. A CPU aguarda a cópia da imagem para a GPU.
        cudaMemcpy(d_imagem, h_imagem, bytes, cudaMemcpyHostToDevice);

        // 3. Inverte os canais RGB e aguarda a conclusão.
        inverter_pixels<<<blocos_inversao, threads>>>(d_imagem, tamanho);
        cudaDeviceSynchronize();

        // 4. Ajusta a saturação da imagem invertida e aguarda novamente.
        alterar_saturacao<<<blocos_saturacao, threads>>>(
            d_imagem, pixels, FATOR_SATURACAO);
        cudaDeviceSynchronize();

        // 5. A CPU aguarda o retorno da imagem processada.
        cudaMemcpy(h_saida, d_imagem, bytes, cudaMemcpyDeviceToHost);

        // 6. Libera a memória da GPU antes da próxima imagem.
        cudaFree(d_imagem);

        const auto fim = std::chrono::steady_clock::now();
        tempo_total +=
            std::chrono::duration<double, std::milli>(fim - inicio).count();

        if (imagem == NUM_IMAGENS - 1) {
            std::cout << "Amostra de pixels da última imagem:\n";
            for (int i = 0; i < 10; ++i) {
                int base = i * CANAIS;
                std::cout << "Pixel[" << i << "] ("
                          << static_cast<int>(h_imagem[base]) << ", "
                          << static_cast<int>(h_imagem[base + 1]) << ", "
                          << static_cast<int>(h_imagem[base + 2]) << ") → ("
                          << static_cast<int>(h_saida[base]) << ", "
                          << static_cast<int>(h_saida[base + 1]) << ", "
                          << static_cast<int>(h_saida[base + 2]) << ")\n";
            }
        }

        std::free(h_imagem);
        std::free(h_saida);
    }
    std::cout << "--------------------------------------------\n"
            << NUM_IMAGENS << " imagens sintéticas processadas: "
            << LARGURA << "x" << ALTURA << " (RGB)\n"
            << "Operações: inversão e saturação (fator "
            << FATOR_SATURACAO << ")\n"
            << "Execução ingênua e síncrona em GPU\n"
            << "Tempo total: " << tempo_total / 1000.0 << " s\n"
            << "Tempo por imagem: "
            << tempo_total / (1000.0 * NUM_IMAGENS) << " s\n"
            << "--------------------------------------------\n";
    return 0;
}

```


