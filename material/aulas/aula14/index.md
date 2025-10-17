# Data Race, Atomics e Throughput em GPU

Quando começamos a ver sobre paralelismo em CPU com **OpenMP**, aprendemos que certas operações compartilhadas entre threads — como somas globais ou atualizações em vetores — exigem cuidados.
Usar um `#pragma omp critical` ou um `#pragma omp atomic` garante correção, mas também gera um gargalo, pois apenas uma thread pode acessar aquele trecho de código por vez.
Em muitos casos, podemos substituir o uso de `critical` por **estratégias mais inteligentes**, como reduções (`reduction`) ou vetores locais por thread, justamente para **evitar o custo da sincronização**.

Em CUDA o raciocínio é o mesmo, mas em uma escala muito maior.
Uma GPU não executa 8 ou 16 threads, e sim **milhares** às vezes **dezenas de milhares** de forma simultânea.
Isso significa que qualquer ponto de contenção, como uma operação `atomicAdd()` sobre um mesmo endereço de memória, pode eliminar completamente o paralelismo e fazer com que o desempenho da GPU drasticamente.

Por isso, **é importante evitar ao máximo o uso de operações atômicas**


## Por que o atomic é tão custoso na GPU

Uma operação atômica é uma forma de **bloquear temporariamente** um endereço de memória enquanto uma thread o atualiza, impedindo que outra thread interfira.
Em CPU, o impacto é pequeno, porque há poucas threads competindo.
Mas na GPU, a atomic vira um verdadeiro funil: centenas ou milhares de threads tentam acessar o mesmo dado ao mesmo tempo, e o hardware é obrigado a **serializar os acessos**, uma thread por vez.

Em OpenMP, se você usar `#pragma omp atomic`, oito threads se alternam para atualizar uma variável o atraso é perceptível, mas suportável.
Em CUDA, `atomicAdd()` pode ser disputado por **10.000 threads ao mesmo tempo**, e o tempo de espera se torna centenas de vezes maior.
Na prática, o throughput (quantidade de operações concluídas por segundo) despenca.
O programa perde completamente o sentido de ser paralelo.


## Uma solução: reduzir a competição por memória

A melhor forma de evitar atomics não é torcer para que elas fiquem baratas, mas **reorganizar o algoritmo** para que cada thread ou bloco trabalhe **em regiões de memória diferentes**.
No exemplo do histograma, em vez de todas as threads atualizarem o mesmo vetor global, cada bloco de threads constrói **seu próprio histograma local**, em *shared memory* (memória compartilhada do bloco).
Essa memória é muito mais rápida, e como é exclusiva daquele bloco, **não há conflito entre blocos** portanto, **nenhum atomic é necessário**.

Cada thread do bloco incrementa contadores no seu histograma local de forma direta (sem precisar de `atomicAdd()` global), e no final o bloco escreve o resultado em uma região separada da memória global.
Depois, uma etapa de fusão combina os histogramas locais para gerar o resultado final.
Essa fusão pode ser feita no host (CPU) ou em um segundo kernel da GPU.
Mesmo que use algumas atomics na fusão, o custo é muito menor, porque agora temos poucos blocos competindo, e não milhares de threads.


## Exemplo: três abordagens de histograma

Neste exemplo temos três implementações do mesmo histograma em CUDA:

1. **Versão ingênua:**
   As threads incrementam diretamente o vetor `hist[bin]++`.
   É a mais rápida, mas incorreta pois acontece *data race*.

2. **Versão com `atomicAdd`:**
   Corrige o problema, mas força o hardware a serializar as operações.
   Funciona, mas é como colocar `#pragma omp critical` dentro do loop cada incremento é seguro, porém caro.

3. **Versão com memória compartilhada:**
   Cada bloco calcula seu histograma local em *shared memory* e depois os resultados são somados.
   Essa abordagem evita o conflito global e preserva o paralelismo.
   É o equivalente GPU da técnica de *reduction* do OpenMP — divide o trabalho, reduz localmente, combina no final.

```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

//
// =====================================================
// Kernel ingênuo — demonstra condição de corrida (data race)
// =====================================================
//
// Cada thread lê um valor do vetor `dados` e incrementa o contador
// do "chunk" correspondente no vetor global `histograma`.
//
// Problema: várias threads podem tentar incrementar o mesmo índice
// ao mesmo tempo. Como a operação (ler → somar → escrever) não é atômica,
// o resultado final se corrompe.
//
__global__ void histograma_ingenuo(const int *dados, int *histograma, int N, int numChunks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int chunk = dados[i];
        histograma[chunk]++;  // condição de corrida (data race)
    }
}

//
// =====================================================
// Kernel com atomicAdd — correto, mas reduz throughput
// =====================================================
//
// A função `atomicAdd()` garante exclusividade de acesso a um endereço.
// Assim, o incremento é seguro, mas o paralelismo efetivo diminui,
// pois várias threads competem para acessar o mesmo chunk.
//
__global__ void histograma_atomico(const int *dados, int *histograma, int N, int numChunks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int chunk = dados[i];
        atomicAdd(&histograma[chunk], 1);  // funciona, porém destroi o paralelismo
    }
}

//
// =====================================================
// Kernel otimizado — histograma local em memória compartilhada
// =====================================================
//
// Cada bloco cria um histograma local na memória compartilhada (`shared memory`),
// que é muito mais rápida e exclusiva de cada bloco.
// Assim, evitamos o uso de operações atômicas globais.
// 
// Após o cálculo local, cada bloco copia seu histograma parcial
// para a memória global, e a fusão final é feita na CPU.
//
__global__ void histograma_compartilhado(const int *dados, int *histogramas_blocos, int N, int numChunks) {
    extern __shared__ int hist_local[];  // memória compartilhada dinâmica
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Etapa 1: Inicializa o histograma local com zeros ---
    for (int i = threadIdx.x; i < numChunks; i += blockDim.x)
        hist_local[i] = 0;
    __syncthreads();

    // --- Etapa 2: Atualiza o histograma local ---
    if (tid_global < N) {
        int chunk = dados[tid_global];
        hist_local[chunk]++;
    }
    __syncthreads();

    // --- Etapa 3: Copia o histograma local para a memória global ---
    for (int i = threadIdx.x; i < numChunks; i += blockDim.x)
        histogramas_blocos[blockIdx.x * numChunks + i] = hist_local[i];
}

//
// =====================================================
// Fusão dos histogramas locais na CPU
// =====================================================
//
// Após cada bloco gerar seu histograma local na GPU,
// esta função soma todos os histogramas parciais
// em um histograma final consolidado.
//
void fundir_histogramas_CPU(const std::vector<int> &histogramas_blocos,
                            std::vector<int> &histograma_final,
                            int numBlocos, int numChunks) {
    for (int b = 0; b < numBlocos; b++)
        for (int c = 0; c < numChunks; c++)
            histograma_final[c] += histogramas_blocos[b * numChunks + c];
}

//
// =====================================================
// Função principal
// =====================================================
//
// Mede o tempo de execução de cada abordagem (ingênua, atômica, compartilhada)
// e compara os resultados.
//
int main() {
    const int N = 1 << 20;        // 1 milhão de elementos
    const int numChunks = 256;    // quantidade de "caixas" do histograma
    const int tamBloco = 256;     // threads por bloco
    const int numBlocos = (N + tamBloco - 1) / tamBloco;

    std::cout << "=== HISTOGRAMA EM GPU ===\n";
    std::cout << "Elementos: " << N
              << " | Chunks: " << numChunks
              << " | " << numBlocos << " blocos x "
              << tamBloco << " threads\n\n";

    // -----------------------------
    // Alocação e inicialização no host
    // -----------------------------
    std::vector<int> h_dados(N);
    for (auto &v : h_dados) v = rand() % numChunks;

    std::vector<int> h_hist_ingenuo(numChunks, 0);
    std::vector<int> h_hist_atomico(numChunks, 0);
    std::vector<int> h_hist_compart(numChunks, 0);

    // -----------------------------
    // Alocação na GPU
    // -----------------------------
    int *d_dados = nullptr;
    int *d_hist = nullptr;
    int *d_hist_blocos = nullptr;
    cudaMalloc(&d_dados, N * sizeof(int));
    cudaMalloc(&d_hist, numChunks * sizeof(int));
    cudaMalloc(&d_hist_blocos, numBlocos * numChunks * sizeof(int));

    cudaMemcpy(d_dados, h_dados.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    size_t tamMemCompart = numChunks * sizeof(int);

    // Variáveis para medir tempo
    cudaEvent_t inicio, fim;
    cudaEventCreate(&inicio);
    cudaEventCreate(&fim);
    float tempo_ingenuo = 0.0f, tempo_atomico = 0.0f, tempo_compart = 0.0f;

    // =====================================================
    // Versão ingênua
    // =====================================================
    cudaMemset(d_hist, 0, numChunks * sizeof(int));
    cudaEventRecord(inicio);
    histograma_ingenuo<<<numBlocos, tamBloco>>>(d_dados, d_hist, N, numChunks);
    cudaEventRecord(fim);
    cudaEventSynchronize(fim);
    cudaEventElapsedTime(&tempo_ingenuo, inicio, fim);
    cudaMemcpy(h_hist_ingenuo.data(), d_hist, numChunks * sizeof(int), cudaMemcpyDeviceToHost);

    // =====================================================
    // Versão atômica
    // =====================================================
    cudaMemset(d_hist, 0, numChunks * sizeof(int));
    cudaEventRecord(inicio);
    histograma_atomico<<<numBlocos, tamBloco>>>(d_dados, d_hist, N, numChunks);
    cudaEventRecord(fim);
    cudaEventSynchronize(fim);
    cudaEventElapsedTime(&tempo_atomico, inicio, fim);
    cudaMemcpy(h_hist_atomico.data(), d_hist, numChunks * sizeof(int), cudaMemcpyDeviceToHost);

    // =====================================================
    // Versão otimizada (memória compartilhada)
    // =====================================================
    cudaEventRecord(inicio);
    histograma_compartilhado<<<numBlocos, tamBloco, tamMemCompart>>>(d_dados, d_hist_blocos, N, numChunks);
    cudaEventRecord(fim);
    cudaEventSynchronize(fim);
    cudaEventElapsedTime(&tempo_compart, inicio, fim);

    std::vector<int> h_hist_blocos(numBlocos * numChunks);
    cudaMemcpy(h_hist_blocos.data(), d_hist_blocos, numBlocos * numChunks * sizeof(int), cudaMemcpyDeviceToHost);
    fundir_histogramas_CPU(h_hist_blocos, h_hist_compart, numBlocos, numChunks);

    // =====================================================
    // Cálculo do throughput (M ops/s)
    // =====================================================
    auto throughput = [&](float ms) {
        return static_cast<double>(N) / (ms / 1000.0) / 1e6;
    };

    double thr_ingenuo  = throughput(tempo_ingenuo);
    double thr_atomico  = throughput(tempo_atomico);
    double thr_compart  = throughput(tempo_compart);

    std::cout << "──────────────────────────────────────────────────────────────\n";
    std::cout << "Versão         | Tempo (ms)      | Throughput (M ops/s)\n";
    std::cout << "──────────────────────────────────────────────────────────────\n";
    std::cout << "Ingênua        | " << tempo_ingenuo  << "         | " << thr_ingenuo  << "\n";
    std::cout << "Atômica        | " << tempo_atomico << "        | " << thr_atomico << "\n";
    std::cout << "Otimizada      | " << tempo_compart << "        | " << thr_compart << "\n";
    std::cout << "──────────────────────────────────────────────────────────────\n\n";

    // =====================================================
    // Liberação de recursos
    // =====================================================
    cudaFree(d_dados);
    cudaFree(d_hist);
    cudaFree(d_hist_blocos);
    cudaEventDestroy(inicio);
    cudaEventDestroy(fim);

    return 0;
}

```

Lembre-se de carregar o modulo cuda disponível, depois compile com: 

```
nvcc -Ofast hist.cu -o hist
```

Execute usando o srun:

```
srun --partition=gpu --gres=gpu:1 ./hist
```


Em programação paralela **sincronizar sempre tem custo**. Mas, na GPU, esse custo é multiplicado por milhares de threads, e o impacto pode ser catastrófico.
Por isso, **usar funções atômicas deve ser o último recurso**, reservado apenas para casos em que não há outra forma de evitar uma condição de corrida.

A verdadeira otimização em CUDA não está em “usar mais threads”, e sim em **organizar o trabalho de modo que cada thread e cada bloco acessem dados diferentes**.
Sempre que o acesso for independente, a GPU mostra toda sua força; quando há disputa, ela se comporta de forma lenta.

Perfeito 🌟 — vamos montar uma explicação **didática e direta** de **throughput**, adaptada para o contexto desse código CUDA (histograma com “chunks”), com analogias que dialogam com o que seus alunos já viram em OpenMP e CPU paralela.

---

### O que é **Throughput**

O termo **throughput** mede **a quantidade de trabalho que um sistema realiza por unidade de tempo**.
No nosso caso, ele indica **quantos incrementos (operações)** a GPU consegue fazer **por segundo**.

Em outras palavras:

> **Throughput = quantas operações o programa consegue realizar por segundo.**

Cada thread da GPU processa um elemento do vetor de entrada `dados[]` e incrementa o contador do “chunk” correspondente no vetor `histograma[]`.
Logo, temos **N operações** (uma para cada elemento).

O tempo total de execução (`tempo_ms`) é medido com os eventos do CUDA (`cudaEventRecord`).


## Fórmula utilizada no código

O throughput é calculado como:

[
\text{Throughput (M ops/s)} = \frac{N}{t_s} \div 10^6
]

onde:

| Símbolo  | Significado                                                  |
| :------- | :----------------------------------------------------------- |
| ( N )    | Número total de operações executadas (elementos processados) |
| ( t_s )  | Tempo total do kernel em segundos                            |
| ( 10^6 ) | Conversão para “milhões de operações por segundo”            |

Como `cudaEventElapsedTime()` retorna o tempo em **milissegundos**, fazemos:

[
t_s = \frac{t_{ms}}{1000}
]

e portanto, no código:

```cpp
auto throughput = [&](float ms) {
    return static_cast<double>(N) / (ms / 1000.0) / 1e6;
};
```

