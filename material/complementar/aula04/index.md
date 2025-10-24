# **Como ganhar desempenho com um código em Python?”**

Agora que já entendemos com funciona um sistema de HPC, já realizamos otimizações me códigos sequenciais utilizando a bibliteca nativa do Pyhton a multiprocessing, estamos prontos para conhecer Numba, uma ferramenta que permite compilar trechos do código em python, permite paralelismo em CPU e paralelismo em GPU, mas antes de começar, é importante prestar atenção nesses pontos:


## Usando um ambiente virutal

Quando queremos garantir isolamento do nosso código para ter compatibilidade de bibliotecas e não ter problemas de interferências com outras aplicações em que estamos trabalhando no ambiente, é sempre importante utilizar as venv's.

Então, se ainda não criou sua venv no Cluster Franky, crie a sua:

```
python -m venv venv
```

Logo em seguida, ative ela:

```
source venv/bin/activate
```

Vamos garantir que nosso ambiente está com as ferramentas atualizadas:

```
python -m pip install --upgrade pip
```

Agora que temos nosso ambiente isolado e atualizado, podemos realizar as instalações necessárias para realizar o nossos testes com numba:

```
pip install numba
```


## **Contexto: O Problema dos Cristais de energia**

Imagine que você trabalha num Laboratório de HPC, onde acontece uma pesquisa sobre cristais que armazenam energia.
Cada cristal tem milhões de átomos, e precisamos calcular sua energia total com base em interações físicas não lineares.

A função de energia é definida como:

$$
E(x) = \sin^2(x) + \sqrt{|x|} + e^{-x^2/50}
$$

Para um cristal com **10 milhões de átomos**, o cálculo em Python puro leva alguns longos segundos.

Vamos otimizar isso!

**O código original (Python puro)**
Desta forma, o Python executa cada iteração linha por linha, sem utilizar nenhum recuro de otimização.

```python
#puro.py
import math, random, time, numpy as np

# ---------- Calculo demorado ---------------
def energy(x):
    return math.sin(x)**2 + math.sqrt(abs(x)) + math.exp(-x**2 / 50)

# --- Preparação dos dados ---
# Número de elementos
N = 100_000_000
# Define a semente para garantir reprodutibilidade
np.random.seed(42)
# Gera sempre o mesmo conjunto de números
data = np.random.uniform(0, 1000, N).astype(np.float32)



start = time.time()
results = [energy(x) for x in data]
end = time.time()

print(f"Os dados: {results[:10]}")
print(f"Tempo total: {end - start:.2f}s")
```

Para testar adequadamente em um ambiente HPC:

```
srun --partition=gpu python puro.py
```

## Otimizações com Numba

Numba é um **compilador just-in-time (JIT)** que traduz funções Python numéricas em **código nativo otimizado**, usando o compilador **LLVM**, o mesmo backend usado por linguagens como C e Julia.
vamo entender como fazer algumas modificações no código para gerar otimizações de desempenho.


### Compilação JIT (Just-in-Time)

A forma mais simples de usar Numba é aplicando o decorator `@njit`, que converte automaticamente uma função Python em código compilado.

```python
#compilado.py
import math, random, time, numpy as np
from numba import njit


#------ Função otimizada ---------
@njit
def energy(x):
    return math.sin(x)**2 + math.sqrt(abs(x)) + math.exp(-x**2 / 50)


# --- Preparação dos dados ---
# Número de elementos
N = 100_000_000
# Define a semente para garantir reprodutibilidade
np.random.seed(42)
# Gera sempre o mesmo conjunto de números
data = np.random.uniform(0, 1000, N).astype(np.float32)

start = time.time()
results = [energy(x) for x in data]
end = time.time()

print(f"Os dados: {results[:10]}")
print(f"Tempo total: {end - start:.2f}s")
```

Para testar adequadamente em um ambiente HPC:

```
srun --partition=gpu python compilado.py
```


Na **primeira execução**, o LLVM compila o trecho de código, nas próximas execuções, o binário otimizado é reutilizado diretamente da cache.

O código continua sequencial, mas agora cada a função otimizada é executada em baixo nível, sem o interpretador Python.


### Paralelismo em CPU com `prange`

O Numba também permite paralelizar **loops independentes** entre os núcleos da CPU.
Para isso, usamos `@njit(parallel=True)` junto com `prange` (parallel range).

```python
#paralelo.py
from numba import njit, prange
import math, random, time, numpy as np

#----- Função otimizada ------
@njit(parallel=True)
def total_energy(data):
    out = np.empty_like(data)
    for i in prange(data.size):  # loop paralelo
        out[i] = math.sin(data[i])**2 + math.sqrt(abs(data[i])) + math.exp(-data[i]**2 / 50)
    return out

# --- Preparação dos dados ---
# Número de elementos
N = 100_000_000
# Define a semente para garantir reprodutibilidade
np.random.seed(42)
# Gera sempre o mesmo conjunto de números
data = np.random.uniform(0, 1000, N).astype(np.float32)

start = time.time()
results = total_energy(data)
end = time.time()

print(f"Os dados: {results[:10]}")
print(f"Tempo (CPU paralela): {end - start:.2f}s")
```

Para testar adequadamente em um ambiente HPC:

```
srun --partition=gpu --cpus-per-task=8 python compilado.py
```


O Numba divide automaticamente o loop entre os núcleos disponíveis do processador,
executando cada pedaço do vetor em paralelo.

### Paralelismo em GPU com `@cuda.jit`

As **GPUs** (Unidades de Processamento Gráfico) foram projetadas originalmente para renderização de imagens, mas, ao longo dos anos, as GPUs foram se especializando cada vez mais em realizar operações matemáticas de multiplas dimensões.

> Enquanto uma CPU tem de 4 a 64 núcleos de computação,
> uma GPU pode ter milhares de núcleos mais simples, capazes de executar o mesmo cálculo em milhares de dados simultaneamente.

Com Numba é possível escrever kernels CUDA dentro do código em Python, usando o decorador `@cuda.jit`.

Cada thread CUDA executa uma instância da função. Essas threads são organizadas em **blocos (blocks)** e **grades (grids)**, que determinam como o trabalho é dividido entre os núcleos da GPU.

**Estrutura básica de um kernel CUDA em Numba**


* `cuda.grid(1)` retorna o **índice global da thread** na grade 1D;
* Cada thread acessa **uma posição diferente** do vetor;
* Nenhuma thread interfere em outra, não existe dependência entre os dados.

```python
from numba import cuda

@cuda.jit
def meu_kernel(x, y):
    i = cuda.grid(1)       # calcula o índice global da thread
    if i < x.size:
        y[i] = x[i] * 2    # cada thread faz uma operação independente
```


Agora vamos passar o nosso código de calculo de energia em cristais para a GPU:

```python
#paralelo_cuda.py
import math, random, time, numpy as np
from numba import cuda

# --- Função otimizada ---
@cuda.jit
def energy_gpu(data, out):
    i = cuda.grid(1)  # índice global da thread em 1 dimensão
    if i < data.size:
        out[i] = math.sin(data[i])**2 + math.sqrt(abs(data[i])) + math.exp(-data[i]**2 / 50)

# --- Preparação dos dados ---
N = 100_000_000
# Define a semente para garantir reprodutibilidade
np.random.seed(42)
# Gera sempre o mesmo conjunto de números
data = np.random.uniform(0, 1000, N).astype(np.float32)
out = np.zeros_like(data)

# --- Definição da grade ---
threads = 256                                 # threads por bloco
blocks = (N + threads - 1) // threads         # número de blocos (grid)

# --- Transferência CPU → GPU ---
d_data = cuda.to_device(data)                # Transfere os dados da CPU para a GPU
d_out = cuda.device_array_like(out)          # Reserva um espaço na memória para trazer os dados de volta para a CPU

# --- Execução ---
start = time.time()
energy_gpu[blocks, threads](d_data, d_out)    # lança o kernel
cuda.synchronize()                            # espera todas as threads terminarem suas missões
end = time.time()

# --- Transferência GPU → CPU ---
results = d_out.copy_to_host()                 # Transfere os dados da GPU para a CPU
print(f"Os dados: {results[:10]}")
print(f"Tempo (GPU): {end - start:.2f}s")
```

Como estamos em um ambiente de HPC, para utilizar devidamente a GPU do sistema, primeiro, precisamos carregar o modulo do driver no ambiente:

```
module load cuda/12.8.1
```

O comando de solicitação de hardware para o SLURM também muda:

```
srun --partition=gpu --gres=gpu:1 python paralelo_cuda.py
```

Como cada thread CUDA processa um elemento do vetor.
Com 100 milhões de elementos, temos **milhões de cálculos simultâneos** acontecendo dentro da GPU.


A distribuição de trabalho em CUDA segue esta lógica:

$$
\text{indice global} = \text{threadIdx.x} + \text{blockIdx.x} \times \text{blockDim.x}
$$

O comando `cuda.grid(1)` faz exatamente esse cálculo automaticamente, assim, cada thread tem a sua fatia de dado correspondente.


Um ponto crítico é a transferência de dados entre CPU (host) e GPU (device):

```python
d_data = cuda.to_device(data)          # host → device
energy_gpu[blocks, threads](...)       # execução no device
result = d_out.copy_to_host()          # device → host
```

Essas cópias têm custo computacional e gera latência. Por isso, o ideal é minimizar transferências e maximizar o tempo de cálculo na GPU.

*De forma geral:*

* Transfira os blocos de dados que serão utilizados pela sua função de uma só vez;
* Faça o máximo de operações possível **antes** de trazer os dados de volta;


### **Comparando o desempenho**

Para avaliar o ganho de desempenho, calculamos o speedup:

$$
Speedup = \frac{Código.Base}{Código.Otimizado}
$$

| Versão             | Tempo (s)  | Speedup |
| ------------------ | ---------  | ------- |
| Python puro        | 38.98      | 1×      |
| Numba compilado    | 15.61      | 2.5x    |
| Numba CPU paralela | 0.67       | 58x     |
| Numba GPU (CUDA)   | 0.21       | 185x    |


De forma geral, o desempenho do código depende da forma como o código é executado no hardware.

Na primeira versão, o código padrão, sem otimizações, é interpretado e sequencial, desta forma, ele não utiliza todo o potencial do hardware disponível.

Com o uso do **Numba**, um trecho deste código é convertido em binário via compilação, eliminando o overhead do interpretador e permitindo a execução otimizada pelo backend **LLVM**, resultando em ganhos imediatos de performance.

Ao adicionar o paralelismo com `@njit(parallel=True)` e o uso de `prange`, o Numba distribui automaticamente as iterações de loops independentes entre os núcleos da CPU, aproveitando a arquitetura de processamento paralelo disponível. 

O próximo nível de otimização foi executar parte do código em **GPU** com o uso do decorator `@cuda.jit`. Nesse modelo, a computação é executada por milhares de threads simultâneas organizadas em blocos e grades, permitindo a execução várias operações ao mesmo. Cada thread CUDA calcula uma parte do problema de forma independente. 

Com isso, espero que conceitos os conceitos de otimização e paralelismo tenham ficado menos abstratos, e que vocês possam utilizar essas ferramentas para melhorar o desempenho das aplicações de vocês, ou pelo menos, ter um ponto de partida para se aprofundar depois.

Se quiser saber mais sobre Numba, a [documentação está disponível aqui](https://numba.readthedocs.io/en/stable/).