#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define LARGURA 3840
#define ALTURA  2160

// =======================================================
// Kernel: inverte a intensidade do pixel
// =======================================================
__global__ void inverter_pixels(unsigned char *imagem, int tamanho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tamanho)
        imagem[idx] = 255 - imagem[idx];
}

// =======================================================
// Função auxiliar para medir tempo entre eventos
// =======================================================
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

    // =======================================================
    // 1) Alocação de memória no host
    // =======================================================
    unsigned char *h_imagem = (unsigned char*)malloc(bytes_total);
    unsigned char *h_saida  = (unsigned char*)malloc(bytes_total);

    for (int i = 0; i < tamanho_total; i++)
        h_imagem[i] = rand() % 256;

    // =======================================================
    // 2) Alocação de memória na GPU
    // =======================================================
    unsigned char *d_imagem;
    cudaMalloc(&d_imagem, bytes_total);

    // =======================================================
    // 3) Medição com eventos CUDA
    // =======================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // =======================================================
    // 4) Cópia síncrona CPU → GPU
    // =======================================================
    cudaMemcpy(d_imagem, h_imagem, bytes_total, cudaMemcpyHostToDevice);

    // =======================================================
    // 5) Execução do kernel
    // =======================================================
    int threads = 256;
    int blocks = (tamanho_total + threads - 1) / threads;

    inverter_pixels<<<blocks, threads>>>(d_imagem, tamanho_total);
    cudaDeviceSynchronize();  // força a GPU a esperar o kernel

    // =======================================================
    // 6) Cópia síncrona GPU → CPU
    // =======================================================
    cudaMemcpy(h_saida, d_imagem, bytes_total, cudaMemcpyDeviceToHost);

    // =======================================================
    // 7) Finalização da medição
    // =======================================================
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempo_total = medir_tempo(start, stop);

    // =======================================================
    // 8) Resultados
    // =======================================================
    std::cout << "--------------------------------------------\n";
    std::cout << "Processamento concluído - execução síncrona em GPU\n";
    std::cout << "Tempo total: " << tempo_total << " ms\n";
    std::cout << "--------------------------------------------\n";

    for (int i = 0; i < 10; i++) {
        std::cout << "Pixel[" << i << "] "
                  << (int)h_imagem[i] << " → "
                  << (int)h_saida[i] << "\n";
    }

    // =======================================================
    // 9) Liberação de recursos
    // =======================================================
    cudaFree(d_imagem);
    free(h_imagem);
    free(h_saida);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
