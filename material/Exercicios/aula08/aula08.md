
## Questão 1 — Teórica (MPI + OpenMP)
**Enunciado:**  
a) Explique por que faz sentido combinar **MPI** e **OpenMP** em clusters de HPC.  
b) Quais são as vantagens dessa abordagem híbrida em relação a usar apenas MPI puro?  
c) Dê um exemplo de problema em que **somente OpenMP** não seria suficiente.  
d) Cite uma dificuldade extra que surge ao programar em MPI+OpenMP em comparação com usar apenas OpenMP.  

??? note "Ver resposta"
    
    a) Os clusters HPC têm múltiplos nós (máquinas diferentes) e múltiplos núcleos por nó (vários core dentro da CPU). O MPI é usado para comunicação entre nós, O OpenMP é usado para paralelismo dentro de cada nó, usandos as threads e a memória compartilhada (L1, L2, L3 e RAM). Juntos, aproveitam ao máximo a hierarquia do hardware (distribuído + compartilhado).

    b) Reduz o número de processos MPI: menos overhead de comunicação entre nós.  
     
     Aproveita melhor a memória compartilhada dentro de cada nó, que é mais rápida que a troca de mensagens MPI.

    Aproveita melhor os recursos disponíveis no cluster: MPI distribui entre nós, OpenMP paraleliza dentro de cada nó.  
    
  
    c) Problemas que não cabem na memória de um único nó. Exemplo: simulação de clima global, que precisa de terabytes de memória, impossível em apenas um nó. Necessário MPI para dividir os dados entre nós diferentes.

    d) Maior complexidade de programação: o programador precisa lidar tanto com gerenciamento de threads (OpenMP) quanto com comunicação entre processos(MPI). Debug e balanceamento de carga ficam mais difíceis em códigos híbridos.  


## Questão 2 — Somas parciais em matriz híbrida
**Enunciado:**  
Implemente um programa que calcula a soma de todos os elementos de uma matriz `NxN` de forma híbrida:  
- O `rank 0` inicializa a matriz com valores aleatórios.  
- Cada processo recebe um bloco de linhas usando `MPI_Scatter`.  
- Dentro de cada processo, use **OpenMP** para somar os elementos do seu bloco.  
- Combine os resultados parciais em `rank 0` com `MPI_Reduce`.  

```cpp
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024;
    std::vector<int> A;
    if (rank == 0) {
        A.resize(N*N);
        for (int i = 0; i < N*N; i++) A[i] = rand() % 10;
    }

    // TODO: MPI_Scatter blocos de linhas
    // TODO: soma parcial local com OpenMP
    // TODO: MPI_Reduce para combinar resultados

    MPI_Finalize();
}
```

??? note "Ver Resposta"

        #include <mpi.h>
        #include <omp.h>
        #include <iostream>
        #include <vector>
        #include <cstdlib>

        int main(int argc, char** argv) {
            // Inicializa o MPI (cria o comunicador global e o ambiente de processos)
            MPI_Init(&argc, &argv);

            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // identificador do processo (0..size-1)
            MPI_Comm_size(MPI_COMM_WORLD, &size);  // total de processos MPI

            // Dimensão da matriz NxN 
            int N = 1024;

            // Apenas o rank 0 mantém a matriz completa; os demais só recebem o seu pedaço
            std::vector<int> A;
            if (rank == 0) {
                A.resize(N * N);
                // Preenche a matriz com inteiros aleatórios simples (0..9)
                for (int i = 0; i < N * N; i++) A[i] = rand() % 10;
            }

            // Número de linhas por processo (versão simples: exige N % size == 0)
            int rows_per_proc = N / size;

            // Buffer local do processo: "rows_per_proc" linhas por N colunas
            std::vector<int> local(rows_per_proc * N);

            // Distribui blocos de linhas de A (contíguas em memória) para todos os processos
            // - No rank 0: envia pedaços de A
            // - Nos demais ranks: recebe em "local"
            MPI_Scatter(
                A.data(),                 // buffer de envio (apenas significativo no rank 0)
                rows_per_proc * N,        // elementos enviados para CADA processo
                MPI_INT,                  // tipo dos elementos
                local.data(),             // buffer de recepção (todos os ranks)
                rows_per_proc * N,        // elementos recebidos por processo
                MPI_INT,                  // tipo dos elementos
                0,                        // root (rank que envia)
                MPI_COMM_WORLD            // comunicador
            );

            // Soma parcial local (vai acumular a soma do bloco deste processo)
            long long soma_local = 0;

            // Paraleliza a soma dentro do nó com OpenMP:
            // - Cada thread percorre um pedaço do vetor "local"
            // - reduction(+:soma_local) evita condição de corrida (cada thread tem acumulador próprio)
            #pragma omp parallel for reduction(+:soma_local)
            for (int i = 0; i < rows_per_proc * N; i++) {
                soma_local += local[i];
            }

            // Reduz (soma) todas as somas locais em "soma_total" no rank 0
            long long soma_total = 0;
            MPI_Reduce(
                &soma_local,              // dado local
                &soma_total,              // resultado no root
                1,                        // quantidade
                MPI_LONG_LONG,            // tipo do dado
                MPI_SUM,                  // operação de redução
                0,                        // root
                MPI_COMM_WORLD
            );

            if (rank == 0) {
                std::cout << "Soma total = " << soma_total << std::endl;
            }

            MPI_Finalize(); // Finaliza MPI
        }



## Questão 3 — Produto escalar distribuído
**Enunciado:**  
Implemente um programa híbrido para calcular o **produto escalar** entre dois vetores `v` e `w`:  

`s = sum(v[i] * w[i])`  

- O `rank 0` inicializa os vetores.  
- Cada processo recebe um pedaço dos vetores via `MPI_Scatter`.  
- Dentro de cada processo, paralelize o cálculo do produto parcial com **OpenMP**.  
- Use `MPI_Reduce` para reunir a soma no `rank 0`.  

```cpp
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1<<20;
    std::vector<float> v, w;
    if (rank == 0) {
        v.resize(N, 1.0f);
        w.resize(N, 2.0f);
    }

    // TODO: Scatter pedaços dos vetores
    // TODO: produto parcial com OpenMP
    // TODO: MPI_Reduce para juntar resultados

    MPI_Finalize();
}
```

??? note "Ver resposta"

    #include <mpi.h>
    #include <omp.h>
    #include <iostream>
    #include <vector>

    int main(int argc, char** argv) {
        // Inicializa o ambiente MPI (processos distribuídos)
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // identificador do processo (0..size-1)
        MPI_Comm_size(MPI_COMM_WORLD, &size);  // total de processos

        // Tamanho global dos vetores (2^20 ~ 1 milhão de elementos)
        int N = 1 << 20;

        // Apenas o rank 0 guarda os vetores completos
        std::vector<float> v, w;
        if (rank == 0) {
            v.resize(N, 1.0f);  // exemplo simples: todos 1.0
            w.resize(N, 2.0f);  // exemplo simples: todos 2.0
        }

        // Particionamento simples: assume N % size == 0 (cada rank recebe o mesmo tamanho)
        int elems_per_proc = N / size;

        // Buffers locais (cada processo recebe um pedaço contíguo de v e w)
        std::vector<float> v_local(elems_per_proc), w_local(elems_per_proc);

        // Distribui um pedaço de v para cada processo
        MPI_Scatter(
            v.data(),                 // buffer origem (válido no rank 0)
            elems_per_proc,           // nº de elementos enviados a cada rank
            MPI_FLOAT,                // tipo
            v_local.data(),           // buffer destino (todos os ranks)
            elems_per_proc,           // nº de elementos recebidos
            MPI_FLOAT,                // tipo
            0, MPI_COMM_WORLD
        );

        // Distribui um pedaço de w para cada processo
        MPI_Scatter(
            w.data(),
            elems_per_proc,
            MPI_FLOAT,
            w_local.data(),
            elems_per_proc,
            MPI_FLOAT,
            0, MPI_COMM_WORLD
        );

        // Soma parcial local do produto escalar desse processo
        // Observação: usar 'double' para acumular reduz erro numérico
        double soma_local = 0.0;

        // Paraleliza o laço local com OpenMP:
        // - Cada thread processa um bloco de índices [0..elems_per_proc)
        // - reduction(+:soma_local) evita condição de corrida (cada thread acumula localmente)
        #pragma omp parallel for reduction(+:soma_local)
        for (int i = 0; i < elems_per_proc; i++) {
            soma_local += static_cast<double>(v_local[i]) * static_cast<double>(w_local[i]);
        }

        // Redução global: soma todas as parcelas locais em 'soma_total' no rank 0
        double soma_total = 0.0;
        MPI_Reduce(
            &soma_local, &soma_total,
            1, MPI_DOUBLE, MPI_SUM,
            0, MPI_COMM_WORLD
        );

        // Rank 0 imprime o resultado final
        if (rank == 0) {
            std::cout << "Produto escalar = " << soma_total << std::endl;
        }

        // Encerra o ambiente MPI
        MPI_Finalize();
    }


## Questão 4 — Filtro 1D paralelo com halos
**Enunciado:**  
Implemente um **filtro de média móvel 1D** sobre um vetor:  

`out[i] = (v[i-1] + v[i] + v[i+1]) / 3`  

- O `rank 0` inicializa o vetor `v`.  
- O vetor é dividido entre os processos (`MPI_Scatter`).  
- Cada processo precisa das **bordas vizinhas (halos)** — troque os elementos das extremidades com `MPI_Sendrecv`.  
- Paralelize o cálculo local com **OpenMP**.  
- Reúna o vetor final no `rank 0` com `MPI_Gather`.  

```cpp
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024;
    std::vector<float> v, out;
    if (rank == 0) {
        v.resize(N, 1.0f);
        out.resize(N);
    }

    // TODO: Scatter blocos do vetor
    // TODO: troca de halos com MPI_Sendrecv
    // TODO: cálculo local com OpenMP
    // TODO: MPI_Gather para juntar resultado

    MPI_Finalize();
}
```
??? note "Ver Resposta"

    #include <mpi.h>
    #include <omp.h>
    #include <iostream>
    #include <vector>

    int main(int argc, char** argv) {
        // ------------------------------------------------------------
        // Inicialização do MPI e descoberta de (rank, size)
        // ------------------------------------------------------------
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // ID do processo (0..size-1)
        MPI_Comm_size(MPI_COMM_WORLD, &size); // nº total de processos

        // ------------------------------------------------------------
        // Dimensão global do vetor e buffers no rank 0
        // (v inicializado, out receberá o resultado global)
        // ------------------------------------------------------------
        int N = 1024;
        std::vector<float> v, out;
        if (rank == 0) {
            v.resize(N, 1.0f); // exemplo simples: todos 1.0
            out.resize(N);
        }

        // ------------------------------------------------------------
        // Particionamento simples: assume N % size == 0
        // Cada processo recebe elems_per_proc elementos contíguos
        // ------------------------------------------------------------
        int elems_per_proc = N / size;
        std::vector<float> local(elems_per_proc), local_out(elems_per_proc);

        // ------------------------------------------------------------
        // Distribui partes de v para todos os processos
        // (rank 0 envia; demais recebem em 'local')
        // ------------------------------------------------------------
        MPI_Scatter(v.data(), elems_per_proc, MPI_FLOAT,
                    local.data(), elems_per_proc, MPI_FLOAT,
                    0, MPI_COMM_WORLD);

        // ------------------------------------------------------------
        // Troca de HALOS (bordas vizinhas):
        //   - Cada processo precisa do elemento da esquerda e da direita
        //     pertencentes aos vizinhos (para computar out[i] nas extremidades).
        //   - Política de borda simples: se não há vizinho, usamos o próprio valor.
        // ------------------------------------------------------------
        float left_halo  = local[0];                    // padrão: próprio valor (borda)
        float right_halo = local[elems_per_proc - 1];   // padrão: próprio valor (borda)

        // Envia/recebe com o vizinho à esquerda (rank-1), se existir
        if (rank > 0) {
            MPI_Sendrecv(&local[0],               1, MPI_FLOAT, rank - 1, 0,
                        &left_halo,              1, MPI_FLOAT, rank - 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Envia/recebe com o vizinho à direita (rank+1), se existir
        if (rank < size - 1) {
            MPI_Sendrecv(&local[elems_per_proc - 1], 1, MPI_FLOAT, rank + 1, 0,
                        &right_halo,                1, MPI_FLOAT, rank + 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // ------------------------------------------------------------
        // Cálculo local do filtro (paralelo com OpenMP)
        //   out[i] = (v[i-1] + v[i] + v[i+1]) / 3
        //   - Para i=0 e i=último, usamos halos recebidos (ou próprio valor na borda).
        // ------------------------------------------------------------
        #pragma omp parallel for
        for (int i = 0; i < elems_per_proc; i++) {
            float left  = (i == 0) ? left_halo : local[i - 1];
            float right = (i == elems_per_proc - 1) ? right_halo : local[i + 1];
            local_out[i] = (left + local[i] + right) / 3.0f;
        }

        // ------------------------------------------------------------
        // Reúne os pedaços filtrados em 'out' no rank 0
        // ------------------------------------------------------------
        MPI_Gather(local_out.data(), elems_per_proc, MPI_FLOAT,
                out.data(),       elems_per_proc, MPI_FLOAT,
                0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Filtro 1D concluído." << std::endl;
            // (Opcional) validar/inspecionar alguns elementos de 'out'
            // for (int i = 0; i < 10; ++i) std::cout << out[i] << " ";
            // std::cout << "\n";
        }

        MPI_Finalize();
    }
