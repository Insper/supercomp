# Simulando a  Prova Final


## Exercício 1

Implemente um programa que recebe um vetor de floats e realiza **duas etapas de processamento em GPU**:

1. Calcule o prefix sum (soma cumulativa) de todos os elementos.
   Exemplo: `[2, 1, 3, 4] → [2, 3, 6, 10]`.

2. Use o último elemento (a soma total) para normalizar todos os valores:
   `[2, 3, 6, 10] → [0.2, 0.3, 0.6, 1.0]`.

| Critério                                   | Descrição                                                                                                                                       | Peso     |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| Compilação sem erros   | O código compila corretamente com `nvcc`.  | **0.2**  |
| Implementação em GPU Síncrona         | Programação paralela em GPU de forma síncrona. | **+0.6** |
| Implementação em GPU Assíncrona        | Utiliza streams, otimiza o uso de CPU e GPU   | **+0.9** |
| Uso correto do Slurm no Cluster Franky |Configurou corretamente o ambiente HPC (via `srun` ou `sbatch`), com parâmetros adequados de GPU. | **+0.3** |
| **Total**                                  |                                                                                                                                                 | **2.0**  |


```cpp
#include <iostream>
#include <vector>
#include <chrono>

// SCAN + NORMALIZAÇÃO 

int main() {
    const size_t N = 87654;
    std::vector<float> v(N);
    std::vector<float> prefix(N);

    auto inicio = std::chrono::high_resolution_clock::now();

    // Gera o vetor
    for (size_t i = 0; i < N; ++i)
        v[i] = static_cast<float>((i + 1) * 2);

    // prefix sum
    prefix[0] = v[0];
    for (size_t i = 1; i < N; ++i)
        prefix[i] = prefix[i - 1] + v[i];

    // Normalização
    float total = prefix.back();  // soma total
    for (size_t i = 0; i < N; ++i)
        prefix[i] /= total;

    auto fim = std::chrono::high_resolution_clock::now();
    double tempo = std::chrono::duration<double, std::milli>(fim - inicio).count();

    // RESULTADOS
    std::cout << "\nÚltimos 10 valores do prefix sum:\n[ ";
    for (size_t i = N - 10; i < N; ++i)
        std::cout << prefix[i] * total << " ";
    std::cout << "]\n";

    std::cout << "Últimos 10 valores normalizados:\n[ ";
    for (size_t i = N - 10; i < N; ++i)
        std::cout << prefix[i] << " ";
    std::cout << "]\n";

    std::cout << "\nTempo CPU: " << tempo << " ms\n";
}

```
??? Note "Implementação Síncrona"
    ```cpp
    #include <iostream>
    #include <vector>
    #include <chrono>
    #include <cuda_runtime.h>
    #define THREADS 1024

    __global__ void scan(float *dados, float *somas_blocos, int N) {
        __shared__ float temp[THREADS];
        int tid = threadIdx.x;
        int gid = blockIdx.x * blockDim.x + tid;

        if (gid < N)
            temp[tid] = dados[gid];
        else
            temp[tid] = 0.0f;
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            int idx = (tid + 1) * offset * 2 - 1;
            if (idx < blockDim.x)
                temp[idx] += temp[idx - offset];
            __syncthreads();
        }

        if (tid == blockDim.x - 1)
            somas_blocos[blockIdx.x] = temp[tid];

        if (tid == blockDim.x - 1)
            temp[tid] = 0.0f;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            int idx = (tid + 1) * offset * 2 - 1;
            if (idx < blockDim.x) {
                float t = temp[idx - offset];
                temp[idx - offset] = temp[idx];
                temp[idx] += t;
            }
            __syncthreads();
        }

        if (gid < N)
            dados[gid] = temp[tid];
    }

    __global__ void adicionar_offsets(float *dados, const float *somas_blocos, int N) {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        int bloco = blockIdx.x;
        if (gid < N && bloco > 0) {
            float offset = somas_blocos[bloco - 1];
            dados[gid] += offset;
        }
    }

    __global__ void normalizar(float *dados, float total, int N) {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < N && total != 0.0f)
            dados[gid] /= total;
    }

    int main() {
        const int N = 87654;
        const int threads = 1024;
        const int blocos = (N + threads - 1) / threads;

        std::vector<float> h_v(N);
        for (int i = 0; i < N; ++i)
            h_v[i] = static_cast<float>((i + 1) * 2);

        float *d_v, *d_somas_blocos;
        cudaMalloc(&d_v, N * sizeof(float));
        cudaMalloc(&d_somas_blocos, blocos * sizeof(float));

        cudaMemcpy(d_v, h_v.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        auto inicio = std::chrono::high_resolution_clock::now();

        scan<<<blocos, threads, threads * sizeof(float)>>>(d_v, d_somas_blocos, N);
        cudaDeviceSynchronize();

        if (blocos > 1) {
            scan<<<1, threads, threads * sizeof(float)>>>(d_somas_blocos, d_somas_blocos, blocos);
            cudaDeviceSynchronize();
            adicionar_offsets<<<blocos, threads>>>(d_v, d_somas_blocos, N);
            cudaDeviceSynchronize();
        }

        float total = 0.0f;
        cudaMemcpy(&total, &d_v[N - 1], sizeof(float), cudaMemcpyDeviceToHost);

        normalizar<<<blocos, threads>>>(d_v, total, N);
        cudaDeviceSynchronize();

        std::vector<float> h_prefix(N);
        cudaMemcpy(h_prefix.data(), d_v, N * sizeof(float), cudaMemcpyDeviceToHost);

        auto fim = std::chrono::high_resolution_clock::now();
        double tempo = std::chrono::duration<double, std::milli>(fim - inicio).count();

        std::cout << "GPU Síncrono \n";
        std::cout << "N = " << N << "\n\n";

        std::cout << "Últimos 10 valores do prefix sum:\n[ ";
        for (int i = N - 10; i < N; ++i)
            std::cout << h_prefix[i] * total << " ";
        std::cout << "]\n";

        std::cout << "Últimos 10 valores normalizados:\n[ ";
        for (int i = N - 10; i < N; ++i)
            std::cout << h_prefix[i] << " ";
        std::cout << "]\n";

        std::cout << "\nSoma total: " << total << "\n";
        std::cout << "Tempo total GPU síncrono: " << tempo << " ms\n";

        cudaFree(d_v);
        cudaFree(d_somas_blocos);
    }
    ```


??? Note "Implementação Assíncrona"
    ```cpp
    #include <iostream>
    #include <vector>
    #include <cuda_runtime.h>

    #define NSTREAMS 2
    #define THREADS  1024

    __global__ void scan(float *dados, float *somas_blocos, int N) {
        __shared__ float temp[THREADS];
        int tid = threadIdx.x;
        int gid = blockIdx.x * blockDim.x + tid;

        if (gid < N) temp[tid] = dados[gid];
        else         temp[tid] = 0.0f;
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            int idx = (tid + 1) * (offset << 1) - 1;
            if (idx < blockDim.x) temp[idx] += temp[idx - offset];
            __syncthreads();
        }

        if (tid == blockDim.x - 1) somas_blocos[blockIdx.x] = temp[tid];
        if (tid == blockDim.x - 1) temp[tid] = 0.0f;
        __syncthreads();

        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            int idx = (tid + 1) * (offset << 1) - 1;
            if (idx < blockDim.x) {
                float t = temp[idx - offset];
                temp[idx - offset] = temp[idx];
                temp[idx] += t;
            }
            __syncthreads();
        }

        if (gid < N) dados[gid] = temp[tid];
    }

    __global__ void adicionar_offsets(float *dados, const float *somas_blocos, int N) {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        int bloco = blockIdx.x;
        if (gid < N && bloco > 0) {
            float offset = somas_blocos[bloco - 1];
            dados[gid] += offset;
        }
    }

    __global__ void add_offset_and_normalize(float *dados, float offset, float total, int N) {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < N && total != 0.0f) {
            dados[gid] = (dados[gid] + offset) / total;
        }
    }

    static inline float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
        float ms=0.0f; cudaEventElapsedTime(&ms, a, b); return ms;
    }

    int main() {
        const int N = 987654321;
        const int chunk_elems = (N + NSTREAMS - 1) / NSTREAMS;

        // Host pinned
        float *h_in  = nullptr;
        float *h_out = nullptr;
        cudaMallocHost(&h_in,  N * sizeof(float));
        cudaMallocHost(&h_out, N * sizeof(float));

        for (int i = 0; i < N; ++i) h_in[i] = float((i + 1) * 2); // pares

        // Streams + eventos
        cudaStream_t streams[NSTREAMS];
        for (int i = 0; i < NSTREAMS; ++i) cudaStreamCreate(&streams[i]);
        cudaEvent_t start_total, stop_total;
        cudaEventCreate(&start_total); cudaEventCreate(&stop_total);

        // Buffers por stream
        std::vector<float*> d_chunk(NSTREAMS, nullptr);
        std::vector<float*> d_blockSums(NSTREAMS, nullptr);
        std::vector<int>    len(NSTREAMS, 0);
        std::vector<int>    blocks(NSTREAMS, 0);
        std::vector<float>  chunk_totals(NSTREAMS, 0.0f);

        for (int i = 0; i < NSTREAMS; ++i) {
            int base = i * chunk_elems;
            int L    = std::min(chunk_elems, std::max(0, N - base));
            len[i]   = L;
            blocks[i]= (L + THREADS - 1) / THREADS;
            if (L > 0) {
                cudaMalloc(&d_chunk[i],     L * sizeof(float));
                cudaMalloc(&d_blockSums[i], std::max(1, blocks[i]) * sizeof(float));
            }
        }

        cudaEventRecord(start_total);

        for (int i = 0; i < NSTREAMS; ++i) {
            if (len[i] == 0) continue;
            int base = i * chunk_elems;
            cudaMemcpyAsync(d_chunk[i], h_in + base, len[i]*sizeof(float),
                            cudaMemcpyHostToDevice, streams[i]);

            scan<<<blocks[i], THREADS, 0, streams[i]>>>(d_chunk[i], d_blockSums[i], len[i]);
            if (blocks[i] > 1) {
                scan<<<1, THREADS, 0, streams[i]>>>(d_blockSums[i], d_blockSums[i], blocks[i]);
                adicionar_offsets<<<blocks[i], THREADS, 0, streams[i]>>>(d_chunk[i], d_blockSums[i], len[i]);
            }

            cudaMemcpyAsync(&chunk_totals[i], d_chunk[i] + (len[i]-1), sizeof(float),
                            cudaMemcpyDeviceToHost, streams[i]);
        }

        for (int i = 0; i < NSTREAMS; ++i) cudaStreamSynchronize(streams[i]);

        std::vector<float> chunk_offsets(NSTREAMS, 0.0f);
        float global_total = 0.0f;
        for (int i = 0; i < NSTREAMS; ++i) {
            chunk_offsets[i] = global_total;
            global_total    += chunk_totals[i];
        }

        for (int i = 0; i < NSTREAMS; ++i) {
            if (len[i] == 0) continue;
            add_offset_and_normalize<<<blocks[i], THREADS, 0, streams[i]>>>(
                d_chunk[i], chunk_offsets[i], global_total, len[i]
            );
            int base = i * chunk_elems;
            cudaMemcpyAsync(h_out + base, d_chunk[i], len[i]*sizeof(float),
                            cudaMemcpyDeviceToHost, streams[i]);
        }

        for (int i = 0; i < NSTREAMS; ++i) cudaStreamSynchronize(streams[i]);

        cudaEventRecord(stop_total);
        cudaEventSynchronize(stop_total);
        float tempo_total = elapsed_ms(start_total, stop_total);

        // Resultados (últimos 10)
        std::cout << "GPU Assíncrono (NSTREAMS=" << NSTREAMS << ")\n";
        std::cout << "N = " << N << "\n\n";

        std::cout << "Últimos 10 valores do prefix sum:\n[ ";
        // reconstruir últimos 10 do prefix sum original: h_out está normalizado, então * global_total
        for (int i = N - 10; i < N; ++i) std::cout << h_out[i] * global_total << " ";
        std::cout << "]\n";

        std::cout << "Últimos 10 valores normalizados:\n[ ";
        for (int i = N - 10; i < N; ++i) std::cout << h_out[i] << " ";
        std::cout << "]\n";

        std::cout << "\nSoma total: " << global_total << "\n";
        std::cout << "Tempo total GPU assíncrono: " << tempo_total << " ms\n";

        // Libera
        for (int i = 0; i < NSTREAMS; ++i) {
            if (d_chunk[i])     cudaFree(d_chunk[i]);
            if (d_blockSums[i]) cudaFree(d_blockSums[i]);
            cudaStreamDestroy(streams[i]);
        }
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
        return 0;
    }

    ```


## Exercício 

Paralelize esse código que calcula a média harmônica dos elementos de um vetor em GPU:

### **Rubrica**
| Critério                                                | Descrição                                                                                                                                                     | Peso     |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | 
| **Compilação sem erros** | O código compila corretamente usando `nvcc`, sem erros | **0.2**  |
| **Implementação em GPU Síncrona**  | O código é paralelizado corretamente em GPU de forma assíncrona. | **+0.6** |
| **Implementação em GPU Assíncrona** | O código utiliza *streams*, com sobreposição de operações entre GPU e CPU. | **+0.9** | 
|**Uso correto do SLURM no Cluster Franky**  | Configurou corretamente o ambiente HPC (via `srun` ou `sbatch`), com parâmetros adequados de GPU.| **+0.3** | 
| **Total**  |                    | **2.0**  | 



```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>


struct Resultado {
    std::vector<double> valores;
    long long soma;
    double tempo_ms;
};

Resultado gerar_valores_e_somar(int N) {
    Resultado r;
    r.soma = 0;
    r.valores.resize(N);

    auto inicio = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        r.valores[i] = i + 1;
        r.soma += i + 1;
    }

    auto fim = std::chrono::high_resolution_clock::now();
    r.tempo_ms = std::chrono::duration<double, std::milli>(fim - inicio).count();

    return r;
}


int main() {
    const int N = 89878;
    double soma_inversos = 0.0;

    std::cout << "Calculando resultados para N = " << N << "...\n\n";

    // Gera vetor e calcula somatória (tudo em uma função)
    Resultado dados = gerar_valores_e_somar(N);

    // MÉDIA HARMÔNICA
    auto inicio_h = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < dados.valores.size(); ++i) {
        double x = dados.valores[i];
        if (x != 0.0)
            soma_inversos += 1.0 / x;
        else
            soma_inversos += 0.0;
    }

    double H = 0.0;
    if (soma_inversos != 0.0)
        H = dados.valores.size() / soma_inversos;

    auto fim_h = std::chrono::high_resolution_clock::now();
    double tempo_h = std::chrono::duration<double, std::milli>(fim_h - inicio_h).count();

    // =========================================================
    // RESULTADOS
    // =========================================================
    std::cout << "Somatória de 1 até " << N << ": " << dados.soma << "\n";
    std::cout << "Média harmônica de 1 até " << N << ": " << H << "\n\n";
    std::cout << "Tempo da geração e somatória: " << dados.tempo_ms << " ms\n";
    std::cout << "Tempo da média harmônica: " << tempo_h << " ms\n";

    return 0;
}

```
??? Note "Implementação Síncrona"

    ```cpp
    #include <iostream>
    #include <cuda_runtime.h>


    // gera vetor [1, 2, 3, ..., N]
    __global__ void gerar_valores(float *valores, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            valores[idx] = static_cast<float>(idx + 1);
        }
    }


    // redução 
    __global__ void reduzir_soma(const float *entrada, float *saida, int N) {
        
        __shared__ float cache[1024];  // tamanho fixo igual a blockDim.x

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N)
            cache[tid] = entrada[idx];
        else
            cache[tid] = 0.0;

        __syncthreads();

        int passo = blockDim.x / 2;
        while (passo > 0) {
            if (tid < passo)
                cache[tid] += cache[tid + passo];
            __syncthreads();
            passo /= 2;
        }

        if (tid == 0)
            saida[blockIdx.x] = cache[0];
    }


    //calcula inversos 
 
     __global__ void calcular_inversos(const float *entrada, float *inversos, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N) {
            float x = entrada[idx];
            if (x != 0.0)
                inversos[idx] = 1.0 / x;
            else
                inversos[idx] = 0.0;
        }
    }


    // calcula média harmônica 
    __global__ void calcular_media_harmonica(float *H, float soma_inversos, int N) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            if (soma_inversos != 0.0)
                *H = static_cast<float>(N) / soma_inversos;
            else
                *H = 0.0;
        }
    }


    // Função recursiva 
    void reduzir_total(float *d_dados, int N, float *resultado) {
        int threads = 128;
        int blocos = (N + threads - 1) / threads;
        float *d_temp = nullptr;

        while (true) {
            cudaMalloc(&d_temp, blocos * sizeof(float));
            reduzir_soma<<<blocos, threads, threads * sizeof(float)>>>(d_dados, d_temp, N);
            cudaDeviceSynchronize();

            // se sobrou só 1 bloco, o resultado final está em d_temp[0]
            if (blocos == 1) {
                cudaMemcpy(resultado, d_temp, sizeof(float), cudaMemcpyDeviceToDevice);
                cudaFree(d_temp);
                break;
            }

            cudaFree(d_dados);
            d_dados = d_temp;
            N = blocos;
            blocos = (N + threads - 1) / threads;
        }
    }


    int main() {
        const int N = 89878;
        const int threads = 128;
        const int blocos = (N + threads - 1) / threads;

        std::cout << "GPU síncrono \n";
        std::cout << "N = " << N << "\n\n";

        // Alocação de memória na GPU
        float *d_valores, *d_inversos, *d_soma_total, *d_soma_inv, *d_H;
        cudaMalloc(&d_valores, N * sizeof(float));
        cudaMalloc(&d_inversos, N * sizeof(float));
        cudaMalloc(&d_soma_total, sizeof(float));
        cudaMalloc(&d_soma_inv, sizeof(float));
        cudaMalloc(&d_H, sizeof(float));


        // Medição de tempo
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Gera vetor [1..N]
        gerar_valores<<<blocos, threads>>>(d_valores, N);
        cudaDeviceSynchronize();

        // Soma total dos valores
        reduzir_total(d_valores, N, d_soma_total);
        cudaDeviceSynchronize();

        // Calcula inversos
        calcular_inversos<<<blocos, threads>>>(d_valores, d_inversos, N);
        cudaDeviceSynchronize();

        // Soma total dos inversos
        reduzir_total(d_inversos, N, d_soma_inv);
        cudaDeviceSynchronize();

        // Copia soma_inversos para CPU para cálculo final
        float soma_inversos = 0.0;
        cudaMemcpy(&soma_inversos, d_soma_inv, sizeof(float), cudaMemcpyDeviceToHost);

        // Calcula média harmônica na GPU
        calcular_media_harmonica<<<1, 1>>>(d_H, soma_inversos, N);
        cudaDeviceSynchronize();

        // Copia resultado final
        float H = 0.0;
        cudaMemcpy(&H, d_H, sizeof(float), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float tempo_ms;
        cudaEventElapsedTime(&tempo_ms, start, stop);

        // -------------------------------
        // Exibe resultados
        // -------------------------------
        float soma_total = 0.0;
        cudaMemcpy(&soma_total, d_soma_total, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Somatória de 1 até " << N << ": " << soma_total << "\n";
        std::cout << "Média harmônica Síncrono: " << H << "\n";
        std::cout << "Tempo total GPU Síncrono: " << tempo_ms << " ms\n";

        // -------------------------------
        // Liberação de memória
        // -------------------------------
        cudaFree(d_valores);
        cudaFree(d_inversos);
        cudaFree(d_soma_total);
        cudaFree(d_soma_inv);
        cudaFree(d_H);

        return 0;
    }

    ```


??? Note "Implementação Assíncrona"
    ```cpp
    #include <iostream>
    #include <cuda_runtime.h>

    // Pipeline:
    //   Stream A → gera vetor e calcula somatória total
    //   Stream B → calcula inverso e soma total dos inversos

    // gera vetor [1, 2, 3, ..., N]
    __global__ void gerar_valores(float *valores, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            valores[idx] = static_cast<float>(idx + 1);
        }
    }


    // redução (soma dentro do bloco)
    __global__ void reduzir_soma(const float *entrada, float *saida, int N) {
        extern __shared__ float cache[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N)
            cache[tid] = entrada[idx];
        else
            cache[tid] = 0.0;

        __syncthreads();

        int passo = blockDim.x / 2;
        while (passo > 0) {
            if (tid < passo)
                cache[tid] += cache[tid + passo];
            __syncthreads();
            passo /= 2;
        }

        if (tid == 0)
            saida[blockIdx.x] = cache[0];
    }


    // calcula inversos 
    __global__ void calcular_inversos(const float *entrada, float *inversos, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N) {
            float x = entrada[idx];
            if (x != 0.0)
                inversos[idx] = 1.0 / x;
            else
                inversos[idx] = 0.0;
        }
    }


    // calcula média harmônica 
    __global__ void calcular_media_harmonica(float *H, float soma_inversos, int N) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            if (soma_inversos != 0.0)
                *H = static_cast<float>(N) / soma_inversos;
            else
                *H = 0.0;
        }
    }


    // Redução recursiva 
    void reduzir_total_async(float *d_dados, int N, float *resultado, cudaStream_t stream) {
        int threads = 128;
        int blocos = (N + threads - 1) / threads;
        float *d_temp = nullptr;

        while (true) {
            cudaMallocAsync(&d_temp, blocos * sizeof(float), stream);
            reduzir_soma<<<blocos, threads, threads * sizeof(float), stream>>>(d_dados, d_temp, N);
            cudaStreamSynchronize(stream);

            if (blocos == 1) {
                cudaMemcpyAsync(resultado, d_temp, sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
                cudaFreeAsync(d_temp, stream);
                break;
            }

            cudaFreeAsync(d_dados, stream);
            d_dados = d_temp;
            N = blocos;
            blocos = (N + threads - 1) / threads;
        }
    }


    // ------------------------------------------------------------
    // Função principal
    // ------------------------------------------------------------
    int main() {
        const int N = 89878;
        const int threads = 128;
        const int blocos = (N + threads - 1) / threads;

        std::cout << "GPU assíncrono\n";
        std::cout << "N = " << N << "\n\n";

        // =========================================================
        // CRIAÇÃO DOS STREAMS
        // =========================================================
        cudaStream_t streamA, streamB;
        cudaStreamCreate(&streamA);  // somatória
        cudaStreamCreate(&streamB);  // inversos

        // Eventos para sincronização entre streams
        cudaEvent_t evt_soma_pronta, evt_inv_pronto;
        cudaEventCreate(&evt_soma_pronta);
        cudaEventCreate(&evt_inv_pronto);

        // =========================================================
        // ALOCAÇÃO DE MEMÓRIA
        // =========================================================
        float *d_valores, *d_inversos;
        float *d_soma_total, *d_soma_inv, *d_H;

        cudaMalloc(&d_valores, N * sizeof(float));
        cudaMalloc(&d_inversos, N * sizeof(float));
        cudaMalloc(&d_soma_total, sizeof(float));
        cudaMalloc(&d_soma_inv, sizeof(float));
        cudaMalloc(&d_H, sizeof(float));

        // =========================================================
        // MEDIÇÃO DE TEMPO GLOBAL
        // =========================================================
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Stream A → gera vetor e soma
        gerar_valores<<<blocos, threads, 0, streamA>>>(d_valores, N);
        reduzir_total_async(d_valores, N, d_soma_total, streamA);
        cudaEventRecord(evt_soma_pronta, streamA); // sinaliza fim da somatória

        // Stream B → calcula inversos e soma inversos
        // depende do vetor gerado, então espera evt_soma_pronta
        cudaStreamWaitEvent(streamB, evt_soma_pronta, 0);
        calcular_inversos<<<blocos, threads, 0, streamB>>>(d_valores, d_inversos, N);
        reduzir_total_async(d_inversos, N, d_soma_inv, streamB);
        cudaEventRecord(evt_inv_pronto, streamB);

        // Calculo final
        cudaStreamWaitEvent(streamA, evt_inv_pronto, 0);

        float soma_inversos = 0.0;
        cudaMemcpyAsync(&soma_inversos, d_soma_inv, sizeof(float),
                        cudaMemcpyDeviceToHost, streamA);

        cudaStreamSynchronize(streamA); // garante que soma_inversos chegou

        calcular_media_harmonica<<<1, 1, 0, streamA>>>(d_H, soma_inversos, N);

        // =========================================================
        // FINALIZAÇÃO
        // =========================================================
        float H = 0.0;
        float soma_total = 0.0;
        cudaMemcpyAsync(&H, d_H, sizeof(float), cudaMemcpyDeviceToHost, streamA);
        cudaMemcpyAsync(&soma_total, d_soma_total, sizeof(float), cudaMemcpyDeviceToHost, streamA);

        cudaEventRecord(stop, streamA);
        cudaEventSynchronize(stop);

        float tempo_ms;
        cudaEventElapsedTime(&tempo_ms, start, stop);

        // =========================================================
        // RESULTADOS
        // =========================================================
        std::cout << "Somatória de 1 até " << N << ": " << soma_total << "\n";
        std::cout << "Média harmônica GPU assincrona: " << H << "\n";
        std::cout << "Tempo total GPU assincrona: " << tempo_ms << " ms\n";

        // =========================================================
        // LIMPEZA
        // =========================================================
        cudaFree(d_valores);
        cudaFree(d_inversos);
        cudaFree(d_soma_total);
        cudaFree(d_soma_inv);
        cudaFree(d_H);

        cudaStreamDestroy(streamA);
        cudaStreamDestroy(streamB);
        cudaEventDestroy(evt_soma_pronta);
        cudaEventDestroy(evt_inv_pronto);

        return 0;
    }


    ```
