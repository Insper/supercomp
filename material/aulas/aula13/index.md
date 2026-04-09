
# Uma Introdução a CUDA

O **CUDA (Compute Unified Device Architecture)** é o modelo de programação paralela desenvolvido pela **NVIDIA**.
Ele permite que desenvolvedores usem o poder das **GPUs** para acelerar aplicações de alto desempenho, executando milhares de threads simultaneamente.

Em um **sistema HPC com SLURM**, como o Franky ou o SDumont, a execução de programas CUDA segue três etapas principais:

1. **Carregar o módulo CUDA** disponível no cluster.
2. **Compilar o código** com o compilador `nvcc`.
3. **Executar o binário** com o `srun` ou `sbatch`(solicitando uma GPU).

---

## Etapa 1 — Preparando o ambiente


Liste os módulos disponíveis e carregue o CUDA:

```bash
module avail cuda
module load cuda/12.8.1
```

Verifique se o compilador CUDA está ativo:

```bash
nvcc --version
```

Deve aparecer algo como

![alt text](image.png)


## Etapa 2 — Código Base (CPU)

Vamos começar com um programa simples em **C++** que soma os elementos de dois vetores na **CPU**.

Crie o arquivo:

```bash
nano exemplo-cpu.cpp
```

Cole o código abaixo:

```cpp
#include <iostream>
#include <cmath>
#include <chrono> 

// Função que soma os elementos de dois vetores
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 100'000'000; 
  float *x = new float[N];
  float *y = new float[N];

  // Inicializa os vetores
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Início da medição de tempo
  auto start = std::chrono::high_resolution_clock::now();

  // Executa a soma
  add(N, x, y);

  // Fim da medição
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  // Calcula erro e soma total
  float maxError = 0.0f;
  double somaTotal = 0.0;

  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
    somaTotal += y[i];
  }

  std::cout << "Erro máximo: " << maxError << std::endl;
  std::cout << "Soma total: " << somaTotal << std::endl;
  std::cout << "Tempo de execução: " << elapsed.count() << " segundos" << std::endl;

  delete [] x;
  delete [] y;

  return 0;
}

```

Compile e execute localmente (sem GPU):

```bash
 srun --partition=gpu  --pty ./ex-cpu
```

Saída esperada:

```
Erro máximo: 0
Soma total: 3e+06
Tempo de execução: 0.00013891 segundos
```


## Etapa 3 — Migrando para GPU (CUDA)

Agora, vamos reescrever o mesmo código para rodar **na GPU**.

Crie o arquivo:

```bash
nano exemplo-gpu.cu
```

Analise o código abaixo:

```cpp
#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// Kernel CUDA (função que será executada na GPU)
__global__
void add(int n, float *x, float *y)
{
  // Calcula o índice global da thread
   // cada thread obtém um índice único 'i'
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Garante que a thread só acesse posições válidas do vetor
  if (i < n)

    // Cada thread realiza a soma de UM elemento:
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 100'000'000;
  size_t size = N * sizeof(float);

  // Cria os vetores na CPU
  float *x = new float[N];
  float *y = new float[N];

  // Inicializa na CPU
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Cria os vetores para a GPU
  float *d_x, *d_y;
  // reserva o espaço para os vetores na GPU
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);

  // início do tempo (incluindo transferência)
  auto start = std::chrono::high_resolution_clock::now();

  // Transferencia dos dados CPU → GPU
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

  // Configurando o kernel
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Computação na GPU
  add<<<numBlocks, blockSize>>>(N, d_x, d_y);

  // Semáforo para aguardar o término da computação na GPU 
  cudaDeviceSynchronize();

  // Transferencia de dados GPU → CPU
  cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

  // fim do tempo
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  // Criação dos das variaveis para exibir o output
  float maxError = 0.0f;
  double somaTotal = 0.0;

  // Preparação dos dados para exibir
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
    somaTotal += y[i];
  }

  std::cout << cudaGetLastError() << std::endl;
  std::cout << "Erro máximo: " << maxError << std::endl;
  std::cout << "Soma total: " << somaTotal << std::endl;
  std::cout << "Tempo total (CPU->GPU->CPU): "
            << elapsed.count() << " segundos" << std::endl;

  // Libera memória
  cudaFree(d_x);
  cudaFree(d_y);
  delete[] x;
  delete[] y;

  return 0;
}

```

Compile com o `nvcc`:

```bash
nvcc exemplo-gpu.cu -o ex-gpu
```

## Etapa 4 — Executando no nó GPU com SLURM

Agora, vamos executar o binário **pedindo uma GPU** via `srun`:

```bash
srun --partition=gpu --gres=gpu:1 ./ex-gpu
```

💡 **Explicação dos parâmetros:**

| Opção                         | Significado                  |
| ----------------------------- | ---------------------------- |
| `--partition=sequana_gpu_dev` | Escolhe a fila de GPU        |
| `--gres=gpu:1`                | Solicita 1 GPU               |
| `./ex1`                       | Executa o programa compilado |

Saída esperada:

```
Erro máximo: 0
Soma total: 3e+06
```

---

##  Explicação do Kernel CUDA

Até agora, vimos como executar um **kernel CUDA** com múltiplas *threads* dentro de **um único bloco**.
Mas as **GPUs modernas** são compostas por **múltiplos processadores paralelos**, chamados **Streaming Multiprocessors (SMs)**.
Cada **SM** pode executar **vários blocos de threads simultaneamente**.

Por exemplo:

* Uma **GPU Tesla P100 (arquitetura Pascal)** possui **56 SMs**.
* Cada SM pode manter até **2048 threads ativas**.
* Isso totaliza mais de **100 mil threads em execução paralela real**!

Para aproveitar todo esse paralelismo, precisamos **lançar o kernel com múltiplos blocos**, e não apenas um.


### O que é uma *Grid* e o que é um *Block*?

Em CUDA:

* Cada **bloco** (*block*) é um grupo de threads que trabalham juntas e compartilham memória local.
* O conjunto de todos os blocos forma uma **grade** (*grid*).

Portanto:

> Uma **grade** é composta por **vários blocos**, e cada **bloco** contém várias **threads**.

A GPU distribui esses blocos entre seus SMs, executando eles conforme há recursos disponíveis.


### Configurando a Execução com Múltiplos Blocos

Se temos `N` elementos (ex: 1 milhão) e queremos 256 threads por bloco,
precisamos calcular **quantos blocos** são necessários para cobrir todos os elementos.

O cálculo é simples:

```cpp
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
```

* `blockSize` → número de threads por bloco (normalmente múltiplo de 32).
* `numBlocks` → número de blocos necessários (arredondando para cima).
* `<<<numBlocks, blockSize>>>` → diz ao CUDA quantos blocos e threads criar.

> Dessa forma, garantimos pelo menos **N threads** para processar todos os elementos.


### Calculando o Índice Global de Cada Thread

Agora, dentro do *kernel*, precisamos adaptar o código para que **cada thread saiba qual parte do vetor processar**.

CUDA fornece variáveis internas que ajudam nisso:

* `threadIdx.x` → índice da thread dentro do bloco
* `blockIdx.x` → índice do bloco dentro da grade
* `blockDim.x` → número de threads por bloco
* `gridDim.x` → número total de blocos

O índice global é calculado assim:

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
```

Esse cálculo é **padrão em CUDA**, porque é o permite mapear cada thread a uma posição única no vetor.



## Exercícios:

1. Teste outros tamanhos de vetor.
2. Modifique o número de blocos e threads para observar o impacto.
3. Teste kernels com mais dimensões.


## Se quiser aprender mais

1. Explore a [documentação do CUDA Toolkit](https://docs.nvidia.com/cuda/index.html).
   Veja o [Guia Rápido de Instalação](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) e o [Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
2. Teste o uso de `printf()` dentro do kernel para imprimir `threadIdx.x` e `blockIdx.x`.
3. Experimente `threadIdx.y`, `threadIdx.z` e `blockIdx.y`.
   Descubra como definir grids e blocos em múltiplas dimensões.
4. Leia sobre a [Unified Memory no CUDA 8](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/) e o mecanismo de migração de páginas da arquitetura Pascal.



obs: Material adaptado do Deep Learning Institute NVIDIA e do NVIDIA Teaching kit - Accelerated Computing

```
