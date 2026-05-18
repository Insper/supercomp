
# Passando uma função da APS para GPU

Vamo aplicar passo a passo a paralelização de uma das funções do código base da APS utilizando CUDA.

O objetivo não é otimizar a função da melhor forma possível, mas é servir como um ponto de partida para entender como transformar um código sequencial executado na CPU em um código paralelo executado na GPU.

### Fluxo básico para portar um código CPU → GPU

De forma geral, praticamente toda aplicação CUDA segue o mesmo fluxo:

**1. Alocar memória na GPU**

Reservamos espaço na memória da GPU utilizando:

```cpp
cudaMalloc(...)
```

**2. Copiar dados da CPU para a GPU**

Transferimos os dados necessários da memória RAM para a memória da GPU:

```cpp
cudaMemcpy(...)
```

**3. Criar o kernel CUDA**

Transformamos a função sequencial em uma função paralela utilizando:

```cpp
__global__
void FunçãoQualquer(int* variaveis_uteis, int dentro_do_kernel)
{
    // Índice global da thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Verifica se está dentro do vetor
    if (variaveis_uteis < dentro_do_kernel)
    {
        // Cada thread opera em uma posição diferente do vetor
        variaveis_uteis[idx] += 1;
    }
}
```

**4. Executar o kernel**

Chamamos o kernel para efetivamente realizar a computação na GPU:

```cpp 
kernel<<<blocks, threads>>>();
```

**5. Copiar o resultado da GPU para a CPU**

Após o processamento, trazemos os resultados de volta para a CPU:

```cpp
cudaMemcpy(...)
```

**6. Liberar a memória da GPU**

Ao final, liberamos os recursos utilizados:

```cpp
cudaFree(...)
```

Esse fluxo será utilizado praticamente em qualquer aplicação CUDA.


### Entendendo o programa

O código base da APS2 implementa uma simulação do comportamento das ações da NVIDIA utilizando o modelo de Black-Scholes aliado ao Método de Monte Carlo.

A aplicação utiliza dados históricos de fechamento das ações para calcular a volatilidade do mercado e, a partir disso, gerar múltiplas projeções probabilísticas para o preço futuro do ativo.

Quando você configura os parâmetros desta forma, como no exemplo:

```cpp
# Execuções
runCPU:
        ./$(CPU_EXE)  100 100 100 100 0.5 0.5

```
Você está configurando:

| Parâmetro           | Valor | Significado                                                            |
| ------------------- | ----- | ---------------------------------------------------------------------- |
| `inLoops`           | `100` | Quantidade de simulações |
| `outLoops`          | `100` | Quantidade total de execuções do método de Monte Carlo                 |
| `timeStepsHistory`  | `100` | Janela de dados históricos utilizados para calcular a volatilidade |
| `timeStepsForecast` | `100` | Janela de tempo para a previsão futura                      |
| `spotPrice`         | `0.5` | Preço inicial da ação no instante inicial                       |
| `riskRate`          | `0.5` | Taxa livre de risco utilizada no modelo matemático                     |

---

Na prática, configurar:

```cpp
# Execuções
runCPU:
        ./$(CPU_EXE)  100 100 100 100 0.5 0.5

```
significa:

```text id=
- Executar 100 simulações
- Executar o Monte Carlo 100 vezes
- Utilizar 100 amostras históricas do mercado
- Projetar 100 passos temporais futuros
- Considerar um preço inicial da ação igual a 0.5
- Utilizar taxa de risco de 0.5
```


Lembrando que para a entrega da [APS2](../../projetos/2026-1/APS2.md) a configuração dos testes muda de acordo com a rúbrica, para validar a pontuação inicial, você deve cumprir os seguintes critérios:

![alt text](image-8.png)

### Passando para GPU
Observe esta função no código base:

```cpp 
/** ---------------------------------------------------------------------------
    Encontra a média de uma matriz 2D ao longo do primeiro índice (numLoops)
    numLoops representa a quantidade de simulações e timeSteps o tempo
----------------------------------------------------------------------------*/

float* find2dMean(float** matrix, int32_t numLoops, int32_t timeSteps)
{
    int32_t j;
    float* avg = new float[timeSteps];
    float sum = 0.0f;

    for (int32_t i = 0; i < timeSteps; i++)
    {
        for (j = 0; j < numLoops; j++)
        {
            sum += matrix[j][i];
        }

        avg[i] = sum / numLoops;
        sum = 0.0f;
    }

    return avg;
}
```

Esta função calcula a média das colunas de uma matriz 2D.

Por exemplo:

```text
matrix:

[ 1 2 3 ]
[ 4 5 6 ]
[ 7 8 9 ]

Resultado:

avg[0] = (1 + 4 + 7)/3
avg[1] = (2 + 5 + 8)/3
avg[2] = (3 + 6 + 9)/3
```

Para migrar a função `find2dMean` para a GPU, seguiremos os passos do fluxo básico. 

### Passo 0 - Preparação dos dados

A GPU não lida bem com `float**` (ponteiros de ponteiros), pois a memória precisa ser contígua. No código base, cada linha da matriz pode estar em um lugar diferente da RAM. Para a GPU, trataremos a matriz como um **vetor único** para facilitar a nossa vida e ganhar desempenho no acesso aos dados.


 `Matriz[row][col]` torna-se `Vetor[row * total_colunas + col]`.


Para a lógica da função `find2dMean` ser executada na GPU, precisaremos aplicar as modificações abaixo;

```cpp
__global__ 
void find2dMeanKernel(float* d_matrix, float* d_avg, int numLoops, int timeSteps) {
    // O Kernel identifica sua coluna pelo ID da Thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < timeSteps) {
        float sum = 0.0f;
        for (int row = 0; row < numLoops; row++) {
            // Acesso linearizado
            sum += d_matrix[row * timeSteps + col];
        }
        // resultado
        d_avg[col] = sum / numLoops;
    }
}
```

Podemos criar uma função de suporte para organizar os dados para a GPU:

```cpp
float* find2dMeanGPU(float* h_matrix, int numLoops, int timeSteps) {
    size_t matrixSize = numLoops * timeSteps * sizeof(float);
    size_t avgSize = timeSteps * sizeof(float);
    
    float *d_matrix, *d_avg;
    float *h_avg = new float[timeSteps];

    // Alocar memória na GPU
    cudaMalloc(&d_matrix, matrixSize);
    cudaMalloc(&d_avg, avgSize);

    // Copiar dados da CPU para a GPU
    cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);

    // Executar o kernel (Configuração de Grid e Bloco)
    int threadsPerBlock = 256;
    int blocksPerGrid = (timeSteps + threadsPerBlock - 1) / threadsPerBlock;
    
    find2dMeanKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_avg, numLoops, timeSteps);

    // Copiar o resultado da GPU para a CPU
    cudaMemcpy(h_avg, d_avg, avgSize, cudaMemcpyDeviceToHost);

    // Liberar a memória da GPU
    cudaFree(d_matrix);
    cudaFree(d_avg);

    return h_avg;
}
```

Agora você precisa ajustar a declaração das funções em `base.hpp` e precisa chamar `find2dMeanGPU` e `find2dMeanKernel` adequadamente em `main_gpu.cu`


**Dica Extra:** O maior "gargalo" agora está na transferência dos dados, não aplicamos nenhuma otimização, apenas levamos o código para a GPU.

**Outra Dica:*** É auspicioso gerar os números aleatórios e os caminhos (Black-Scholes) dentro da GPU.