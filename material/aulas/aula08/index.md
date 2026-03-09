# Sincronização em CPU com OpenMP

Nos exemplos anteriores vimos alguns problemas que aparecem quando múltiplas threads executam ao mesmo tempo:

* **condições de corrida (race condition)**
* **dependência de dados**
* **overhead de criação de tarefas**

Quando várias threads precisam acessar dados compartilhados, às vezes é necessário controlar o acesso a esses dados para evitar resultados incorretos.

OpenMP fornece alguns mecanismos para isso:

* `atomic`
* `critical`
* `reduction`
* `barrier`
* `ordered`

Esses mecanismos garantem correção do programa, mas têm um custo importante: **Eles reduzem o paralelismo.**

Sempre que usamos um mecanismo de sincronização, estamos dizendo para as threads:

> "Esperem um pouco, apenas uma de cada vez pode fazer isso."

Ou seja, parte do código volta a se comportar **de forma sequencial**.


## O caso clássico: atualização de variável compartilhada

Considere novamente o exemplo da soma:

```cpp
double soma = 0.0;

#pragma omp parallel for
for (int i = 0; i < N; i++) {
    soma += a[i];
}
```

Esse código tem o problema da condição de corrida.

Várias threads podem tentar atualizar `soma` ao mesmo tempo.

O resultado final fica inconsistente.

## Usando `critical`

Se usamos `critical`, apenas uma thread por vez pode executar o trecho dentro da região `critical`.

Isso garante a segurança na leitura e na escrita da variável, mas cria um problema.

Cada atualização de `soma` precisa:

1. esperar outras threads
2. entrar na região crítica
3. executar
4. liberar o acesso

Se o laço tem milhões de iterações, essa espera pode se tornar muito cara.

Na prática, o código pode ficar mais lento que a versão sequencial.


```cpp
double soma = 0.0;

#pragma omp parallel for
for (int i = 0; i < N; i++) {

    #pragma omp critical
    soma += a[i];
}
```


## Usando `atomic`

Para operações simples, OpenMP oferece uma alternativa menos drástica:

```cpp
double soma = 0.0;

#pragma omp parallel for
for (int i = 0; i < N; i++) {

    #pragma omp atomic
    soma += a[i];
}
```

`atomic` garante que a atualização da variável será feita de forma indivisível. Que quer dizer que essa operação acontece como se fosse um único passo, que não pode ser interrompido ou intercalado por outra thread.

Ou seja, enquanto uma thread está executando a operação atômica, nenhuma outra thread pode interferir naquela mesma variável.

A diferença é que `atomic` é mais "barata" que `critical`, pois:

* funciona diretamente com instruções atômicas do hardware
* evita bloqueios mais pesados

Mesmo assim, ainda existe custo.

Se muitas threads atualizam a mesma variável, todas competem por esse acesso.

## Usando variáveis `private`

Outra estratégia muito comum em programação paralela é evitar compartilhar dados entre threads.

Em vez de várias threads competirem pela mesma variável, cada thread trabalha com sua própria cópia local.

No OpenMP isso pode ser feito com variáveis `private`.

```cpp
double soma = 0.0;
double soma_local;

#pragma omp parallel private(soma_local)
{
    soma_local = 0.0;

    #pragma omp for
    for (int i = 0; i < N; i++) {
        soma_local += a[i];
    }

    #pragma omp critical
    soma += soma_local;
}
```



## Comparando as três abordagens

O problema é:

> Dado um vetor grande de valores inteiros, queremos contar quantas vezes cada valor aparece.

Isso gera múltiplas threads atualizando o mesmo contador, o que causa condição de corrida.


```cpp
//hist.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

// tamanho do vetor de dados
#define N 20000000

// quantidade de categorias do histograma
#define BINS 2

/*
Função que preenche o vetor com valores aleatórios
entre 0 e BINS-1
*/
void fill_data(std::vector<int> &data) {

    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, BINS-1);

    for (int i = 0; i < N; i++) {
        data[i] = dist(gen);
    }
}

/*
Zera o histograma antes de cada execução
*/
void reset_histogram(std::vector<int> &hist) {
    for (int i = 0; i < BINS; i++)
        hist[i] = 0;
}

/*
Versão usando ATOMIC
Cada incremento no histograma é protegido
por uma operação atômica.

Isso garante proteção contra condição de corrida, mas cria contenção
quando muitas threads tentam atualizar
o mesmo contador.
*/
double run_atomic(std::vector<int> &data, std::vector<int> &hist) {

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {

        int bin = data[i];


        #pragma omp atomic
        hist[bin]++;
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end-start).count();
}

/*
Versão usando CRITICAL

A região critical permite que apenas
uma thread por vez execute o bloco.

Isso resolve a condição de corrida,
mas pode criar um gargalo forte,
pois o acesso fica praticamente sequencial.
*/
double run_critical(std::vector<int> &data, std::vector<int> &hist) {

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {

        int bin = data[i];

        // apenas uma thread pode executar
        #pragma omp critical
        {
            hist[bin]++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end-start).count();
}

/*
Versão usando histogramas privados

Cada thread cria um histograma local,
evitando concorrência durante o processamento.

No final, os resultados são combinados.

Essa abordagem reduz drasticamente
a necessidade de sincronização.
*/
double run_private_merge(std::vector<int> &data, std::vector<int> &hist) {

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {

        // cada thread cria seu próprio histograma
        std::vector<int> local_hist(BINS, 0);

        // divide as iterações entre as threads
        #pragma omp for
        for (int i = 0; i < N; i++) {

            int bin = data[i];

            local_hist[bin]++;
        }

        // combinação final dos resultados
        #pragma omp critical
        {
            for (int i = 0; i < BINS; i++)
                hist[i] += local_hist[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end-start).count();
}

int main() {

    std::vector<int> data(N);
    std::vector<int> histogram(BINS);

    fill_data(data);

    // diferentes quantidades de threads para testar
    std::vector<int> thread_tests = {1,2,4,8,16};

    for (int threads : thread_tests) {

        omp_set_num_threads(threads);

        std::cout << "\nThreads: " << threads << std::endl;

        reset_histogram(histogram);
        double t_atomic = run_atomic(data, histogram);

        reset_histogram(histogram);
        double t_critical = run_critical(data, histogram);

        reset_histogram(histogram);
        double t_private = run_private_merge(data, histogram);

        std::cout << "atomic:   " << t_atomic << " s\n";
        std::cout << "critical: " << t_critical << " s\n";
        std::cout << "private:  " << t_private << " s\n";
    }

    return 0;
}
```

Para compilar use

```cpp
g++ -fopenmp -O3 hist.cpp -o hist
```

Para testar no SLURM

```bash
srun --partition=normal --cpus-per-task=16 ./hist
```

ou

```bash
#!/bin/bash
#SBATCH --job-name=hist   # nome do job
#SBATCH --output=hist.out  # arquivo de saída
#SBATCH --cpus-per-task=16              # 4 threads para usar
#SBATCH --time=00:05:00                 # tempo máximo de execução
#SBATCH --mem=2G                        # Memória 

# garante que o OpenMP use exatamente os recursos alocados pelo SLURM
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# executa o binário
./hist

```

## Conclusão

Os resultados mostram que mecanismos de sincronização podem reduzir o desempenho de um programa paralelo, principalmente quando muitas threads precisam acessar o mesmo dado compartilhado.

As versões com `atomic` e `critical` não escalam bem conforme aumentamos o número de threads.

No caso de `atomic`, todas as threads precisam atualizar a mesma posição do histograma:

```cpp
hist[bin]++;
```

A diretiva `atomic` garante que essa operação seja executada de forma indivisível, evitando condições de corrida. Para isso, o hardware precisa garantir que apenas uma thread por vez modifique aquela posição de memória. Como resultado, quando várias threads tentam atualizar a mesma variável simultaneamente, ocorre contenção de memória. As threads passam a competir pelo mesmo endereço, gerando invalidações de cache e serialização da operação no nível do hardware. 

Já na versão com `critical`, o comportamento é mais restritivo. A diretiva cria uma região crítica, permitindo que apenas uma thread execute aquele trecho de código por vez. Assim, quando várias threads chegam nesse ponto do programa, elas precisam esperar sua vez para entrar na região crítica. Isso cria uma espécie de fila de execução, tornando aquela parte do código sequencial. Quanto maior o número de threads, maior tende a ser o tempo de espera.

Na versão que utiliza `private`, cada thread mantém sua própria estrutura de dados local durante o processamento e só ocorre sincronização no momento final de combinar os resultados. Dessa forma, o número de operações que exigem coordenação entre threads é reduzido, diminuindo o overhead nesta aplicação.

> Sempre que possível, devemos **evitar acesso concorrente a dados compartilhados**, preferindo estruturas locais e reduzindo ao mínimo os pontos de sincronização.

