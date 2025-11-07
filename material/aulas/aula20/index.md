# Exercícios - Convolução , CSR e Otimizações em GPU

### Exercício 1

Considere uma imagem 4K (3840×2160 pixels) em grayscale representada como matriz `I[x][y]`.
Quando aplicamos um filtro de convolução, como uma máscara Laplaciana, esse filtro é representado por uma matriz de pesos, como:

### filtro Laplaciano 3×3:

```
M =  | 0   -1   0 |
     | -1   4  -1 |
     | 0   -1   0 |

```

Ao aplicar o filtro sobre a imagem, é realizado uma operação de convolução que destaca as bordas da imagem realçando as figuras da imagem.


Otimize o código abaixo aplicando a a [técnica CSR](../aula17/index.md) e [paralelizando a convolução em GPU](../aula15/index.md):


### **Rubrica**
| Critério                                                | Descrição                                                                                                                                                     | Peso     |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | 
| **Compilação sem erros** | O código compila corretamente usando `nvcc`, sem erros | **0.2**  |
| **Implementação em GPU**  | O código aplica corretamente a técnica de stencil para paralelizar a operação de convolução em GPU. | **+1.5** |
| **Implementação da técnica CSR** | O código aplica corretamente a técnica CSR para otimizar o gerenciamento dos dados não nulos da matriz. | **+1.5** | 
|**Uso correto do SLURM no Cluster Franky**  | Configurou corretamente o ambiente HPC (via `srun` ou `sbatch`), com parâmetros adequados de GPU.| **+0.3** | 
| **Total**  |                    | **4.0**  | 




```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// Gera uma imagem binária com 4 quadrantes em padrão quad (0 e 255)
void gerarImagemQuad(std::vector<unsigned char>& imagem, int largura, int altura) {
    for (int y = 0; y < altura; y++) {
        for (int x = 0; x < largura; x++) {
            bool direita = x >= largura / 2;
            bool abaixo = y >= altura / 2;
            if ((direita && !abaixo) || (!direita && abaixo))
                imagem[y * largura + x] = 255;
            else
                imagem[y * largura + x] = 0;
        }
    }
}

// Mostra uma linha da imagem para visualização
void mostrarLinhaCentral(const std::vector<unsigned char>& imagem, int largura, int altura) {
    int y_centro = altura / 2;
    int inicio = y_centro * largura + (largura / 2) - 10;

    std::cout << " (60 pixels centrais):\n[ ";
    for (int i = 0; i < 60; i++) {
        std::cout << (int)imagem[inicio + i] << " ";
    }
    std::cout << "]\n\n";
}

// Aplica o filtro Laplaciano 3x3 na imagem
void aplicarFiltroLaplaciano(const std::vector<unsigned char>& imagem, std::vector<unsigned char>& saida, int largura, int altura, double& tempo_ms) {
    int N = largura * altura;

    int kernel[3][3] = {
        {  0, -1,  0 },
        { -1,  4, -1 },
        {  0, -1,  0 }
    };

    std::vector<int> saida_i(N, 0);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Convolução (sem processar as bordas)
    for (int y = 1; y < altura - 1; y++) {
        for (int x = 1; x < largura - 1; x++) {
            int acc = 0;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int peso = kernel[j + 1][i + 1];
                    int val  = (int)imagem[(y + j) * largura + (x + i)];
                    acc += peso * val;
                }
            }
            saida_i[y * largura + x] = acc;
        }
    }

    // Normalização (binária: borda vira 255, o resto vira 0)
    for (int idx = 0; idx < N; idx++) {
        int v = saida_i[idx];
        saida[idx] = (v > 0) ? 255 : 0;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    tempo_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Exibe a máscara Laplaciana usada
void mostrarMascara(int kernel[3][3]) {
    std::cout << "\nMáscara utilizada:\n";
    for (int j = 0; j < 3; j++) {
        std::cout << "| ";
        for (int i = 0; i < 3; i++) {
            std::cout << std::setw(3) << kernel[j][i] << " ";
        }
        std::cout << "|\n";
    }
}

int main() {
    int largura = 3840;
    int altura  = 2160;
    int N = largura * altura;

    std::vector<unsigned char> imagem(N, 0);
    std::vector<unsigned char> saida(N, 0);

    std::cout << "=== IMAGEM ORIGINAL ===\n";
    std::cout << "Resolução: " << largura << "x" << altura << " (" << N << " pixels)\n\n";

    gerarImagemQuad(imagem, largura, altura);
    mostrarLinhaCentral(imagem, largura, altura);

    double tempo_ms = 0.0;
    aplicarFiltroLaplaciano(imagem, saida, largura, altura, tempo_ms);

    std::cout << "=== FILTRO LAPLACIANO 3x3 ===\n";
    std::cout << "Tempo CPU: " << tempo_ms << " ms\n";

    int kernel[3][3] = {
        {  0, -1,  0 },
        { -1,  4, -1 },
        {  0, -1,  0 }
    };
    mostrarMascara(kernel);

    // Exibe amostra da imagem filtrada
    int base = (altura / 2) * largura + (largura / 2);
    std::cout << "\n Imagem filtrada, 60 pixels centrais:\n[ ";
    for (int i = 0; i < 60; ++i)
        std::cout << (int)saida[base + i] << " ";
    std::cout << "]\n";

    return 0;
}


```

??? Implementação em GPU
    ```cpp
    #include <iostream>
    #include <cuda_runtime.h>

    #define WIDTH  3840
    #define HEIGHT 2160
    #define N      (WIDTH * HEIGHT)

    // Máscara Laplaciana 3x3 em formato CSR
    #define MASK_SIZE 9
    #define NONZEROS 5

    __constant__ int csr_values[NONZEROS]   = { -1, -1,  4, -1, -1 };
    __constant__ int csr_col_idx[NONZEROS]  = {  1, 3, 4, 5, 7 };
    __constant__ int csr_row_ptr[4]         = { 0, 2, 3, 5 };

    // Kernel CUDA: aplica filtro Laplaciano com máscara CSR
    __global__ void filtro_laplaciano_csr(unsigned char *img_in, unsigned char *img_out, int largura, int altura) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= 1 && x < largura - 1 && y >= 1 && y < altura - 1) {
            int acc = 0;
            for (int linha = 0; linha < 3; linha++) {
                for (int i = csr_row_ptr[linha]; i < csr_row_ptr[linha + 1]; i++) {
                    int col = csr_col_idx[i];
                    int peso = csr_values[i];

                    int dx = (col % 3) - 1;
                    int dy = linha - 1;

                    int px = x + dx;
                    int py = y + dy;

                    int valor = img_in[py * largura + px];
                    acc += peso * valor;
                }
            }
            img_out[y * largura + x] = (acc > 0) ? 255 : 0;
        }
    }

    // Função: Gera imagem 
    void gerar_imagem_quad(unsigned char *imagem) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                bool direita = x >= WIDTH / 2;
                bool abaixo  = y >= HEIGHT / 2;
                imagem[y * WIDTH + x] = (direita != abaixo) ? 255 : 0;
            }
        }
    }

    // Mostra 20 pixels centrais da imagem
    void mostrar_pixels_centrais(unsigned char *saida) {
        int base = (HEIGHT / 2) * WIDTH + (WIDTH / 2);
        std::cout << "Pixels centrais:\n[ ";
        for (int i = 0; i < 60; ++i)
            std::cout << (int)saida[base + i] << " ";
        std::cout << "]\n";
    }

    // Lança kernel e mede tempo de execução
    float aplicar_filtro_gpu(unsigned char *imagem, unsigned char *saida) {
        dim3 threads(16, 16);
        dim3 blocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        filtro_laplaciano_csr<<<blocks, threads>>>(imagem, saida, WIDTH, HEIGHT);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    int main() {
        unsigned char *imagem, *saida;
        cudaMallocManaged(&imagem, N);
        cudaMallocManaged(&saida,  N);

        gerar_imagem_quad(imagem);

        float tempo = aplicar_filtro_gpu(imagem, saida);
        std::cout << "Tempo GPU: " << tempo << " ms\n";

        mostrar_pixels_centrais(saida);

        cudaFree(imagem);
        cudaFree(saida);
        return 0;
    }

    ```

??? Arquivo Slurm 
    Submetendo com `srun`:
    ```bash
    module load cuda/12.8.1
    srun --partition=gpu --gres=gpu:1 ./binario_gpu
    ```

    Submetendo com `sbatch`:
    ```bash
    #!/bin/bash
    #SBATCH --job-name=Ex01
    #SBATCH --output=saida.out
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --time=00:05:00
    #SBATCH --mem=1G

    module load cuda/12.8.1
    ./binario_gpu
    ```


## Exercício 2

A computação do calculo de matrizes esparças podem apresentar gargalos de desempenho quando o número de elementos não nulos por linha é irregular.

Seu objetivo é aplicar técnicas de otimização para garantir desempenho na implementação abaixo: 

** Missões:**
* **Faça uso eficiente de memória, garantindo boa localidade espacial**;
* **Garanta uma configuração adequada de blocos e threads**;
* **Utilize técnicas de profiling para medir adequadamente as melhorias do código otimizado**


 

### **Rúbrica**

| Critério                                     | Descrição                                                       | Peso    |
| -------------------------------------------- | --------------------------------------------------------------- | ------- |
| **Compilação e execução sem erros**          | O código compila com `nvcc` e executa corretamente.             | **0.2** |
| **Implementação do kernel otimizado**        | Uso de memória compartilhada e boa localidade espacial.               | **0.4** |
| **Configuração eficiente de blocos/threads** | Escolha adequada para maximizar desempenho.                     | **0.3** |
| **Análise de desempenho**                    | Análise de desempenho com base no profiling do código | **0.6** |
| **Total**                                    |                                                                 | **1.5** |

```cpp

    #include <stdio.h>
    #include <cuda_runtime.h>

    __global__ void spmv_csr_ruim(const int *row_ptr, const int *col_idx, const float *val, const float *x, float *y, int N) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        // Cada thread processa apenas UMA linha, mesmo com baixa ocupação
        if (gid < N) {
            float acc = 0.0f;
            for (int i = row_ptr[gid]; i < row_ptr[gid + 1]; i++) {
                int col = col_idx[i];
                acc += val[i] * x[col];
            }
            for (int j = 0; j < 10000; ++j) {
                acc += 0.0f;
            }
            y[gid] = acc;
        }
    }

    int main() {
        const int N = 5; // 5x5 matriz esparsa
        const int nnz = 10; // número de elementos não-nulos

        int h_row_ptr[6] = {0, 2, 4, 6, 8, 10};
        int h_col_idx[10] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 0};
        float h_val[10]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1};
        float h_x[5]      = {1, 1, 1, 1, 1};
        float h_y[5]      = {0};

        int *d_row_ptr, *d_col_idx;
        float *d_val, *d_x, *d_y;

        cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, nnz * sizeof(int));
        cudaMalloc(&d_val, nnz * sizeof(float));
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));

        cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, h_val, nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(32);
        dim3 grid((N + block.x - 1) / block.x);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        spmv_csr_ruim<<<grid, block>>>(d_row_ptr, d_col_idx, d_val, d_x, d_y, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Tempo kernel base (ruim): %f ms\n", ms);

        cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        printf("y[0..4]: [ ");
        for (int i = 0; i < N; i++) printf("%.0f ", h_y[i]);
        printf("]\n");

        cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_val);
        cudaFree(d_x); cudaFree(d_y);

        return 0;
    }

```

??? Gabarito
    ```cpp
    #include <stdio.h>
    #include <cuda_runtime.h>

    __global__ void spmv_csr_otimizado(const int *row_ptr, const int *col_idx, const float *val, const float *x, float *y, int N) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid < N) {
            float acc = 0.0f;
            int row_start = row_ptr[gid];
            int row_end = row_ptr[gid + 1];
            for (int i = row_start; i < row_end; i++) {
                acc += val[i] * x[col_idx[i]];
            }
            y[gid] = acc;
        }
    }

    int main() {
        const int N = 5; // 5x5 matriz esparsa
        const int nnz = 10; // número de elementos não-nulos

        int h_row_ptr[6] = {0, 2, 4, 6, 8, 10};
        int h_col_idx[10] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 0};
        float h_val[10]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1};
        float h_x[5]      = {1, 1, 1, 1, 1};
        float h_y[5]      = {0};

        int *d_row_ptr, *d_col_idx;
        float *d_val, *d_x, *d_y;

        cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, nnz * sizeof(int));
        cudaMalloc(&d_val, nnz * sizeof(float));
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));

        cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, h_val, nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(128);
        dim3 grid((N + block.x - 1) / block.x);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        spmv_csr_otimizado<<<grid, block>>>(d_row_ptr, d_col_idx, d_val, d_x, d_y, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Tempo otimizado: %f ms\n", ms);

        cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        printf("y[0..4]: [ ");
        for (int i = 0; i < N; i++) printf("%.0f ", h_y[i]);
        printf("]\n");

        cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_val);
        cudaFree(d_x); cudaFree(d_y);

        return 0;
    }

    ```




??? Profiling com Slurm
    É importante sempre garantir que o modulo cuda foi carregado no ambiente:
    ```bash
    module load cuda/12.8.1
    ```
    Comando para realizar o profiling do código 
    ```bash
    srun --partition=gpu --gres=gpu:1 nsys profile -o relatorio --stats=true --trace=cuda,osrt ./binario
    ```

    Lembre-se de executar mais de uma vez, a primeira execução sempre vai ser mais lenta!