
# Uma Introdução a CUDA

O **CUDA (Compute Unified Device Architecture)** é o modelo de programação paralela desenvolvido pela **NVIDIA**.
Ele permite que desenvolvedores usem o poder das **GPUs** para acelerar aplicações de alto desempenho, executando milhares de threads simultaneamente.

Em um **sistema HPC com SLURM**, como o Franky ou o SDumont, a execução de programas CUDA segue três etapas principais:

1. **Carregar o módulo CUDA** disponível no cluster.
2. **Compilar o código** com o compilador `nvcc`.
3. **Executar o binário** com o `srun` ou `sbatch`(solicitando uma GPU).

---

## Etapa 1 — Preparando o ambiente

Para saber informações sobre as filas que você tem acesso

```bash
sacctmgr list user $USER -s format=partition%20,MaxJobs,MaxSubmit,MaxNodes,MaxCPUs,MaxWall
```

Dentro do seu diretório de trabalho `/scratch/insperhpc/seu_login/GPU/`:

```bash
mkdir /scratch/insperhpc/seu_login/GPU
cd /scratch/insperhpc/seu_login/GPU
```

Liste os módulos disponíveis e carregue o CUDA:

```bash
module avail cuda
module load cuda/12.6_sequana
```

Verifique se o compilador CUDA está ativo:

```bash
nvcc --version
```

---

## Etapa 2 — Código Base (CPU)

Vamos começar com um programa simples em **C++** que soma os elementos de dois vetores na **CPU**.

Crie o arquivo:

```bash
nano exemplo_cpu.cpp
```

Cole o código abaixo:

```cpp
#include <iostream>
#include <math.h>

// Função que soma os elementos de dois vetores
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1 milhão de elementos
  float *x = new float[N];
  float *y = new float[N];

  // Inicializa os vetores
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Executa a soma na CPU
  add(N, x, y);

  // Verifica erro
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));

  std::cout << "Erro máximo: " << maxError << std::endl;

  delete [] x;
  delete [] y;

  return 0;
}
```

Compile e execute localmente (sem GPU):

```bash
g++ exemplo_cpu.cpp -o ex_cpu
```

```bash
srun --partition=sequana_cpu_dev ./ex_cpu
```

Saída esperada:

```
srun: job 123456 queued and waiting for resources
srun: job 123456 has been allocated resources
Erro máximo: 0
```

---

## Etapa 3 — Migrando para GPU (CUDA)

Agora, vamos reescrever o mesmo código para rodar **na GPU**.

Crie o arquivo:

```bash
nano exemplo1.cu
```

Cole o código abaixo:

```cpp
#include <iostream>
#include <math.h>

// Kernel CUDA: soma os elementos de dois vetores
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Aloca Memória Unificada – acessível pela CPU e pela GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // Inicializa vetores na CPU
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Configuração da grade e dos blocos
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Lança o kernel na GPU
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Aguarda a GPU terminar
  cudaDeviceSynchronize();

  // Verifica o erro máximo
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));

  std::cout << "Erro máximo: " << maxError << std::endl;

  // Libera memória
  cudaFree(x);
  cudaFree(y);

  return 0;
}
```

Compile com o `nvcc`:

```bash
nvcc exemplo1.cu -o ex1
```

---

## Etapa 4 — Executando no nó GPU com SLURM

Agora, vamos executar o binário **pedindo uma GPU** via `srun`:

```bash
srun --partition=sequana_gpu_dev --gres=gpu:1 ./ex1
```

💡 **Explicação dos parâmetros:**

| Opção                         | Significado                  |
| ----------------------------- | ---------------------------- |
| `--partition=sequana_gpu_dev` | Escolhe a fila de GPU        |
| `--gres=gpu:1`                | Solicita 1 GPU               |
| `./ex1`                       | Executa o programa compilado |

Saída esperada:

```
srun: job 11402255 queued and waiting for resources
srun: job 11402255 has been allocated resources
Erro máximo: 0
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


### Percorrendo Grandes Vetores com *Grid-Stride Loop*

Mesmo com muitos blocos, às vezes o número de threads **ainda é menor que N**.
Para lidar com isso, usamos um padrão chamado **grid-stride loop**:

```cpp
int stride = blockDim.x * gridDim.x;
for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
```

**O que isso faz:**

* `stride` representa o número total de threads ativas na GPU.
* Cada thread processa múltiplos elementos separados por esse intervalo.
* Assim, mesmo que `N` seja muito grande, todas as posições do vetor são tratadas.

---

### Kernel completo com múltiplos blocos

```cpp
#include <iostream>
#include <math.h>

// Kernel CUDA para somar dois vetores usando múltiplos blocos
__global__
void add(int n, float *x, float *y)
{
  int index  = blockIdx.x * blockDim.x + threadIdx.x; // índice global da thread
  int stride = blockDim.x * gridDim.x;                // salto entre iterações

  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1 milhão de elementos
  float *x, *y;

  // Memória unificada (CPU + GPU)
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // Inicialização dos vetores
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Configuração da execução
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  // Lançamento do kernel
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Sincronização
  cudaDeviceSynchronize();

  // Verificação do resultado
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));

  std::cout << "Erro máximo: " << maxError << std::endl;

  cudaFree(x);
  cudaFree(y);
  return 0;
}
```

Versão com prints didáticos 
```cpp
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

// Kernel CUDA para somar dois vetores usando múltiplos blocos
__global__
void add(int n, float *x, float *y)
{
  // Calcula o índice global e o salto entre iterações
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Imprime informações das primeiras threads de cada bloco
  if (threadIdx.x == 0) {
    printf("Bloco %d ativo com %d threads (index inicial global = %d)\n",
           blockIdx.x, blockDim.x, index);
  }

  // Cada thread processa múltiplos elementos separados por 'stride'
  for (int i = index; i < n; i += stride) {
    if (i < 10 && blockIdx.x == 0 && threadIdx.x < 5) {
      printf("[Thread %d | Bloco %d] somando x[%d]=%.1f + y[%d]=%.1f\n",
             threadIdx.x, blockIdx.x, i, x[i], i, y[i]);
    }
    y[i] = x[i] + y[i];
  }
}

int main(void)
{
  int N = 1<<20; // 1 milhão de elementos
  float *x, *y;

  std::cout << "=== Inicializando memória unificada ===" << std::endl;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // Inicialização dos vetores
  std::cout << "=== Inicializando vetores x e y ===" << std::endl;
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Configuração da execução
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  std::cout << "=== Configuração do kernel ===" << std::endl;
  std::cout << "Número total de elementos: " << N << std::endl;
  std::cout << "Threads por bloco (blockSize): " << blockSize << std::endl;
  std::cout << "Número de blocos (numBlocks): " << numBlocks << std::endl;
  std::cout << "Threads totais (gridDim*blockDim): "
            << numBlocks * blockSize << std::endl;
  std::cout << "=======================================" << std::endl;

  // Lançamento do kernel
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Sincronização
  cudaDeviceSynchronize();

  std::cout << "\n=== Kernel finalizado, verificando resultados ===" << std::endl;

  // Verificação do resultado
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));

  std::cout << "Erro máximo: " << maxError << std::endl;

  // Mostra os primeiros 10 resultados
  std::cout << "\n=== Amostra dos primeiros 10 valores de y ===" << std::endl;
  for (int i = 0; i < 10; i++)
    std::cout << "y[" << i << "] = " << y[i] << std::endl;

  cudaFree(x);
  cudaFree(y);
  std::cout << "\n=== Execução completa ===" << std::endl;

  return 0;
}

```



### Visualização do Modelo

| Nível  | Identificador | Exemplo | Função                 |
| ------ | ------------- | ------- | ---------------------- |
| Thread | `threadIdx.x` | 0–255   | Uma linha de execução  |
| Bloco  | `blockIdx.x`  | 0–4095  | Agrupa 256 threads     |
| Grade  | `gridDim.x`   | 4096    | Agrupa todos os blocos |

Cada thread calcula:

```
index = blockIdx.x * blockDim.x + threadIdx.x
```

e acessa `y[index]`.

### Compilando e executando no HPC

No seu ambiente:

```bash
module load cuda/12.6_sequana
nvcc add_grid.cu -o add_grid
srun --partition=sequana_gpu_dev --gres=gpu:1 ./add_grid
```

Saída esperada:

```
srun: job 11402255 queued and waiting for resources
srun: job 11402255 has been allocated resources
Erro máximo: 0
```

### Conceitos-chave dessa etapa

| Conceito                                  | Explicação                                               |
| ----------------------------------------- | -------------------------------------------------------- |
| **Streaming Multiprocessor (SM)**         | Unidade de processamento paralela da GPU                 |
| **Grid**                                  | Conjunto de blocos lançados na GPU                       |
| **Block**                                 | Grupo de threads que compartilham memória local          |
| **Grid-Stride Loop**                      | Técnica para percorrer grandes vetores com menos threads |
| **blockIdx.x * blockDim.x + threadIdx.x** | Fórmula padrão de indexação CUDA                         |

## Exercícios:

1. Teste outros tamanhos de vetor.
2. Modifique o número de blocos e threads para observar o impacto.
3. Adicione medições de tempo com as funções CUDA (`cudaEventRecord`).
4. Teste kernels com mais dimensões.


## Se quiser aprender mais

1. Explore a [documentação do CUDA Toolkit](https://docs.nvidia.com/cuda/index.html).
   Veja o [Guia Rápido de Instalação](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) e o [Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
2. Teste o uso de `printf()` dentro do kernel para imprimir `threadIdx.x` e `blockIdx.x`.
3. Experimente `threadIdx.y`, `threadIdx.z` e `blockIdx.y`.
   Descubra como definir grids e blocos em múltiplas dimensões.
4. Leia sobre a [Unified Memory no CUDA 8](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/) e o mecanismo de migração de páginas da arquitetura Pascal.



obs: Material adaptado do Deep Learning Institute NVIDIA e do NVIDIA Teaching kit - Accelerated Computing

```
