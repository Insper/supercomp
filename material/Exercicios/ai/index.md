# Gabarito Prova Intermediária 1 Semestre de 2026


## **Questão 1 - Laboratório de previsão climática (2.5 pontos)**

Você está trabalhando em um laboratório de previsão climática que usa supercomputadores para simular o comportamento da atmosfera.
A região de interesse é dividida em uma grade de pontos (latitude × longitude), e em cada ponto são armazenadas variáveis como temperatura, pressão e umidade.

Para prever o clima nas próximas horas, é preciso aplicar um modelo matemático que combina essas variáveis em cada ponto da grade. Essa combinação pode ser representada como a **multiplicação de matrizes**:

* A matriz **A** contém os valores medidos nos sensores (temperatura, pressão, umidade).
* A matriz **B** representa os coeficientes do modelo físico (como cada variável influencia a previsão).
* A matriz **C = A × B** contém os novos valores ajustados, que alimentam os próximos passos da simulação.

Por serem matrizes muito grandes (milhares de linhas e colunas), não é viável calcular tudo de forma sequencial. Otimize o código para que seja possível realizar essa computação de forma paralela. 

Sem nenhuma otimização o código gera esse resultado, se executado na fila `gpu` do Cluster Franky:

```bash
N = 2048
Tempo: 71.6775 s
Para validação dos dados:
Primeiro elemento da matriz C[0][0] = 2170.03
Elemento do meio da matriz C[N/2][N/2] = 2190.39
Ultimo elemento da matriz C[N-1][N-1] = 2218.17
```

### Rúbrica

**1 ponto — Paralelismo simples**
- Você usou openmp para paralelizar o código;
- **Conseguiu realizar a computação em pelo menos 30 segundos;**
- Mas não aplicou técnicas de tilling;
- Nem reorganizou os loops para melhorar a localidade espacial.

**1.5 pontos — Paralelismo com tilling**  
- Você usou openmp para paralelizar o código;
- **Conseguiu realizar a computação em menos de 20 segundos;**
- **Aplicou tiling, escolhendo tamanhos de blocos compatíveis com a cache L2;**
- **Usou um schedule que balanceia a carga entre as threads;**
-  mas não reorganizou os loops para melhorar a localidade espacial. 

 **2 pontos — Paralelismo eficiente** 
- Você usou openmp para paralelizar o código,
- **Conseguiu realizar a computação em menos de 15 segundos;**
- Aplicou tiling, escolhendo tamanhos de blocos compatíveis com a cache L2;
- Usou um schedule que balanceia a carga entre as threads;
- **Reorganizou os loops para melhorar a localidade espacial.**

**+0.5 Bônus**  
- Você justificou as otimizações implementadas;
- Criou um arquivo SLURM que executa o código com diferentes números de threads;
- Encontrou o valor de 'cpus-per-task' com o melhor custo benefício.

```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>
using namespace std;

int main(int argc, char** argv) {
    int N = 2048;

    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N, 0.0));

    // Inicialização dos dados
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = sin(i + j);
            B[i][j] = cos(i - j);
        }
    }

    auto t0 = chrono::high_resolution_clock::now();

    // Multiplicação
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double soma = 0.0;
            for (int k = 0; k < N; k++) {
                double val = A[i][k] * B[k][j];
                soma += val * sin(val) + cos(val);
            }
            C[i][j] = soma;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double tempo = chrono::duration<double>(t1 - t0).count();

    cout << "N = " << N << endl;
    cout << "Tempo: " << tempo << " s" << endl;

    cout << "Para validação dos dados:" << endl;
    cout << "Primeiro elemento da matriz C[0][0] = " << C[0][0] << endl;
    cout << "Elemento do meio da matriz C[N/2][N/2] = " << C[N/2][N/2] << endl;
    cout << "Ultimo elemento da matriz C[N-1][N-1] = " << C[N-1][N-1] << endl;


    return 0;
}
```

??? note "Ver a resposta"

    ```cpp
    // Implementação desenvolvida por Victor Faria Soares como parte da avaliação
    // intermediária da disciplina de Supercomputação, realizada em 30/03/2026.
    #include <iostream>
    #include <vector>
    #include <chrono>
    #include <cmath>
    #include <omp.h>
    using namespace std;

    int main(int argc, char** argv) {
        int N = 2048;

        // Da pra calcular o tamanho do bloco para o tilling, conforme exemplo na Aula 3
        const int BLOCK = 64;

        vector<vector<double>> A(N, vector<double>(N));
        vector<vector<double>> B(N, vector<double>(N));
        vector<vector<double>> C(N, vector<double>(N, 0.0));

        // Inicialização dos dados
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = sin(i + j);
                B[i][j] = cos(i - j);
            }
        }

        auto t0 = chrono::high_resolution_clock::now();

        // OPENMP: paralelização com múltiplas threads
        // collapse(2): paraleliza dois loops externos (ii e kk), aumentando granularidade
        // schedule(dynamic, 4): balanceamento de carga entre threads
        #pragma omp parallel for schedule(dynamic, 4) collapse(2)
        for (int ii = 0; ii < N; ii += BLOCK) {   
            for (int kk = 0; kk < N; kk += BLOCK) { 

                for (int jj = 0; jj < N; jj += BLOCK) { 

                    // Limites dos blocos (evita ultrapassar N)
                    int i_end = min(ii + BLOCK, N);
                    int k_end = min(kk + BLOCK, N);
                    int j_end = min(jj + BLOCK, N);

                    // REORGANIZAÇÃO DE LOOPS para melhorar a localidade espacial
                    // Ordem: i → k → j
                    // - A[i][k] é reutilizado (boa localidade temporal)
                    // - B[k][j] acessado sequencialmente em j (boa localidade espacial)
                    // - C[i][j] atualizado de forma eficiente
                    for (int i = ii; i < i_end; i++) {

                        for (int k = kk; k < k_end; k++) {

                            // Otimização: evita acessar A[i][k] várias vezes
                            double a_ik = A[i][k]; 

                            for (int j = jj; j < j_end; j++) {
                                double val = a_ik * B[k][j];
                                C[i][j] += val * sin(val) + cos(val);
                            }
                        }
                    }
                }
            }
        }

        auto t1 = chrono::high_resolution_clock::now();
        double tempo = chrono::duration<double>(t1 - t0).count();

        cout << "N = " << N << endl;
        cout << "Tempo: " << tempo << " s" << endl;

        cout << "Para validação dos dados:" << endl;
        cout << "Primeiro elemento da matriz C[0][0] = " << C[0][0] << endl;
        cout << "Elemento do meio da matriz C[N/2][N/2] = " << C[N/2][N/2] << endl;
        cout << "Ultimo elemento da matriz C[N-1][N-1] = " << C[N-1][N-1] << endl;

        return 0;
    }

    ```



## **Questão 2 — Otimização de Rotas para Entregas (2.5 pontos)**
Olha ele de novo!!! O problema das entregas, mas agora estamos auxiliando o Juliano, um jovem carteiro que realiza entrega de boletos pela cidade.

Devemos considerar que Juliano possui limitações físicas e operacionais:
* Ele consegue carregar no máximo 100 pacotes de bloetos em sua mala;
* Sua meta é realizar a entrega nos 100 endereços até esvaziar a sua mala;
* Os pontos de entrega estão [disponívies aqui](pontos.txt)

O objetivo é:

Determinar uma rota de entregas que minimize a distância total percorrida (custo).

### Rúbrica:

* **0** - Não compila ou  **não implementou a heurística Hill Climbing Aleatória**.   
* **0.5** - **Implementou a Hill Climbing Aleatória**, mas sem otimizações.
* **1** - Otimizou a heurísitca melhorando seu tempo de execução, mas ainda tem uma implementação sequencial.
* **2** - Paralelizou o algorítimo e conseguiu melhorar o tempo em relação a versão sequencial.

**Bônus**
* **+0.25** - Criou um arquivo SLURM que executa o código com diferentes números de threads; Encontrou o valor de 'cpus-per-task' com o melhor custo benefício.
* **+0.25** - Conseguiu explicar os resultados obtidos com as otimizações aplicadas de forma clara e objetiva.


??? note "Ver a resposta"
    ```cpp
    // Implementação desenvolvida por Lucas Espina como parte da avaliação
    // intermediária da disciplina de Supercomputação, realizada em 30/03/2026.
    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <algorithm>
    #include <random>
    #include <chrono>
    #include <fstream>
    #include <numeric>
    #include <omp.h>

    using namespace std;

    // Estrutura que representa um ponto no plano
    struct Ponto {
        double x, y;
    };

    // Calcula a distância euclidiana entre dois pontos
    double distancia(const Ponto& a, const Ponto& b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return sqrt(dx * dx + dy * dy);
    }

    // Calcula o custo total de uma rota (soma das distâncias)
    double custo_rota(const vector<Ponto>& pontos, const vector<int>& rota) {
        double total = 0.0;
        for (int i = 0; i < (int)rota.size() - 1; i++) {
            total += distancia(pontos[rota[i]], pontos[rota[i + 1]]);
        }
        return total;
    }

    // IMPLEMENTAÇÃO DA HEURÍSTICA HILL CLIMBING ALEATÓRIA
    double hill_climbing(const vector<Ponto>& pontos, vector<int>& melhor_rota, int max_iter, unsigned int seed) {
        int N = pontos.size();

        // Cria uma rota inicial (0,1,2,...,N-1)
        vector<int> rota(N);
        iota(rota.begin(), rota.end(), 0);

        // Embaralha a rota (solução inicial aleatória)
        mt19937 gen(seed);
        shuffle(rota.begin(), rota.end(), gen);

        double melhor_custo = custo_rota(pontos, rota);

        uniform_int_distribution<int> dist(0, N - 1);

        
        // tenta melhorar a solução através de pequenas trocas
        for (int iter = 0; iter < max_iter; iter++) {
            int i = dist(gen);
            int j = dist(gen);
            if (i == j) continue;

            // Gera vizinho trocando dois pontos
            swap(rota[i], rota[j]);

            double novo_custo = custo_rota(pontos, rota);

            // Se melhorou, aceita
            if (novo_custo < melhor_custo) {
                melhor_custo = novo_custo;
            } else {
                // Caso contrário, desfaz 
                swap(rota[i], rota[j]);
            }
        }

        melhor_rota = rota;
        return melhor_custo;
    }

    int main() {
        // Leitura dos pontos do arquivo
        ifstream fin("pontos.txt");
        if (!fin.is_open()) {
            cerr << "Erro ao abrir pontos.txt" << endl;
            return 1;
        }

        vector<Ponto> pontos;
        double x, y;
        while (fin >> x >> y) {
            pontos.push_back({x, y});
        }
        fin.close();

        int N = pontos.size();

        // múltiplos reinícios melhora qualidade da solução e paralelismo
        int num_starts = 500;

        // Número de iterações por execução
        int max_iter = 50000;

        cout << "Pontos: " << N << endl;
        cout << "Multi-starts: " << num_starts << endl;
        cout << "Iteracoes por start: " << max_iter << endl;

        auto t0 = chrono::high_resolution_clock::now();

        double global_melhor_custo = 1e18;
        vector<int> global_melhor_rota;

        
        #pragma omp parallel
        {
            // Cada thread mantém sua melhor solução local
            double local_melhor_custo = 1e18;
            vector<int> local_melhor_rota;

            // schedule(dynamic): balanceia a carga entre threads
            #pragma omp for schedule(dynamic)
            for (int s = 0; s < num_starts; s++) {

                // Semente diferente por execução garante diversidade
                unsigned int seed = s * 12345 + 42;

                vector<int> rota;

                // Executa hill climbing independente
                double custo = hill_climbing(pontos, rota, max_iter, seed);

                // Atualiza melhor solução local da thread
                if (custo < local_melhor_custo) {
                    local_melhor_custo = custo;
                    local_melhor_rota = rota;
                }
            }

            // sincroniza atualização da melhor solução global
            #pragma omp critical
            {
                if (local_melhor_custo < global_melhor_custo) {
                    global_melhor_custo = local_melhor_custo;
                    global_melhor_rota = local_melhor_rota;
                }
            }
        }

        auto t1 = chrono::high_resolution_clock::now();
        double tempo = chrono::duration<double>(t1 - t0).count();

        cout << "Melhor custo: " << global_melhor_custo << endl;
        cout << "Tempo: " << tempo << " s" << endl;

        return 0;
    }

    ```
    Bônus:
    1. Hill Climbing Aleatorio (Multi-Start):
    - Cada start gera uma permutacao aleatoria diferente dos pontos.
    - Em cada iteracao, dois indices aleatorios sao sorteados e trocados (swap).
    - Se o custo melhora, aceita; senao, desfaz. Isso evita minimos locais.
    - 1000 starts com 100000 iteracoes cada garante boa exploracao do espaco.

    2. Paralelismo com OpenMP:
    - Cada start eh independente, entao eh trivialmente paralelizavel.
    - Cada thread mantem uma melhor solucao local (sem race conditions).
    - Ao final, uma secao critica compara as melhores solucoes locais.
    - schedule(dynamic) equilibra a carga entre threads.

    3. Seeds deterministas por start:
    - Cada start usa uma seed diferente baseada no indice.
    - Garante reprodutibilidade e evita que threads gerem solucoes iguais.

    Melhor custo benefício conforme testes no SLURM foram 8 threads.

## **Questão 3 - Interação gravitacional de estrelas (3 pontos)**

Você está trabalhando em um projeto de astrofísica computacional que simula a dinâmica de estrelas em uma galáxia.
Durante a simulação, cada estrela sofre influência gravitacional de todas as outras, seguindo a Lei da Gravitação Universal.
A versão sequencial, quando testada no Cluster Franky usando a fila normal, teve esse resultado:

```bash
Simulacao concluida para 85000 corpos.
Tempo (forcas): 27.9822 s
Tempo (update): 9.2206e-05 s
Tempo total: 27.9823 s
```

Otimize o código para que seja possível realizar essa computação de forma paralela.


### Rúbrica

**1 ponto — Paralelismo local com OpenMP**

* Paralelizou corretamente o código usando OpenMP;
* Conseguiu reduzir o tempo de execução do código em relação a implementação sequencial.


**2 pontos — Paralelismo distribuído com MPI**

* Adaptou o código para execução com múltiplos processos (MPI);
* Dividiu o conjunto de estrelas entre os processos;
* Cada nó calcula apenas seu subconjunto de estrelas;
* Reuniu os resultados no Rank 0;

**3 pontos — MPI + OpenMP**

* A implementação usa MPI e OpenMP;
* O arquivo SLURM definiu corretamente o número de nós, o número de tarefas e a quantidade de threads por tarefa. 
* Conseguiu reduzir o tempo de execução em comparação com a versão que utiliza apenas uma única técnica de paralelismo (MPI ou OpenMP).


```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

const double G = 6.674e-11;

struct Corpo {
    double x, y;
    double vx, vy;
    double massa;
};


void calcula_forca(const Corpo& a, const Corpo& b, double& fx, double& fy) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;

    double dist = std::sqrt(dx * dx + dy * dy) + 1e-9;

    double F = G * a.massa * b.massa / (dist * dist);

    fx += F * dx / dist;
    fy += F * dy / dist;
}

int main() {
    int N = 85000; 

    std::vector<Corpo> corpos(N);

    // Inicialização dos dados (baixo custo e só executa uma vez, não vale a pena otimizar)
    for (int i = 0; i < N; i++) {
        corpos[i].x = i * 1.0;
        corpos[i].y = i * 0.5;
        corpos[i].vx = 0.0;
        corpos[i].vy = 0.0;
        corpos[i].massa = 1e20;
    }

    std::vector<double> fx(N, 0.0);
    std::vector<double> fy(N, 0.0);

    double dt = 0.01;

    //  Tempo total
    auto t_total_inicio = std::chrono::high_resolution_clock::now();

    // Tempo do cálculo de forças 
    auto t_forca_inicio = std::chrono::high_resolution_clock::now();

    // Esse trecho tem o maior custo computacional
    // DICAS:
    // - Você pode usar o MPI para dividir o intervalo de i entre processos
    // - Para a implementação Híbrida, você pode usar MPI para distribuir o i e o OpenMP  para paralelizar j
    
    for (int i = 0; i < N; i++) {

        double fx_local = 0.0;
        double fy_local = 0.0;

        for (int j = 0; j < N; j++) {
            calcula_forca(corpos[i], corpos[j], fx_local, fy_local);

        fx[i] = fx_local;
        fy[i] = fy_local;
    }
    auto t_forca_fim = std::chrono::high_resolution_clock::now();

    // Tempo da atualização 
    auto t_update_inicio = std::chrono::high_resolution_clock::now();

    // DICA:
    // - Pode ser paralelizado com OpenMP 
    // - Cada iteração é independente
    for (int i = 0; i < N; i++) {
        corpos[i].vx += fx[i] / corpos[i].massa * dt;
        corpos[i].vy += fy[i] / corpos[i].massa * dt;

        corpos[i].x += corpos[i].vx * dt;
        corpos[i].y += corpos[i].vy * dt;
    }

    auto t_update_fim = std::chrono::high_resolution_clock::now();

    auto t_total_fim = std::chrono::high_resolution_clock::now();

    
    double tempo_forca = std::chrono::duration<double>(t_forca_fim - t_forca_inicio).count();
    double tempo_update = std::chrono::duration<double>(t_update_fim - t_update_inicio).count();
    double tempo_total = std::chrono::duration<double>(t_total_fim - t_total_inicio).count();

    std::cout << "Simulacao concluida para " << N << " corpos.\n";
    std::cout << "Tempo (forcas): " << tempo_forca << " s\n";
    std::cout << "Tempo (update): " << tempo_update << " s\n";
    std::cout << "Tempo total: " << tempo_total << " s\n";

    return 0;
}
```

??? note "Com Broadcast + Reduce"
    ```cpp
    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <chrono>
    #include <mpi.h>
    #include <omp.h>

    const double G = 6.674e-11;

    struct Corpo {
        double x, y;
        double vx, vy;
        double massa;
    };

    void calcula_forca(const Corpo& a, const Corpo& b, double& fx, double& fy) {
        double dx = b.x - a.x;
        double dy = b.y - a.y;

        double dist = std::sqrt(dx * dx + dy * dy) + 1e-9;
        double F = G * a.massa * b.massa / (dist * dist);

        fx += F * dx / dist;
        fy += F * dy / dist;
    }

    int main(int argc, char** argv) {

        // Inicialização da comunicação MPI
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int N = 85000;

        std::vector<Corpo> corpos(N);

        // Inicialização dos dados no rank 0
        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                corpos[i].x = i * 1.0;
                corpos[i].y = i * 0.5;
                corpos[i].vx = 0.0;
                corpos[i].vy = 0.0;
                corpos[i].massa = 1e20;
            }
        }

        // Distribuição dos dados com Broadcast para os nós
        MPI_Bcast(corpos.data(), N * sizeof(Corpo), MPI_BYTE, 0, MPI_COMM_WORLD);

        std::vector<double> fx(N, 0.0);
        std::vector<double> fy(N, 0.0);

        double dt = 0.01;

        // Divisão do trabalho (cada processo recebe parte dos corpos)
        int inicio = rank * (N / size);
        int fim;

        // Se for o último processo, ele pega até o final do vetor
        if (rank == size - 1) {
            fim = N;
        } 
        // Caso contrário, pega apenas o bloco padrão
        else {
            fim = inicio + (N / size);
        }

        auto t_total_inicio = std::chrono::high_resolution_clock::now();
        auto t_forca_inicio = std::chrono::high_resolution_clock::now();

        // PARALELISMO HÍBRIDO:
        // MPI divide "i" (processos)
        // OpenMP paraleliza "j" (threads)
        #pragma omp parallel for schedule(dynamic)
        for (int i = inicio; i < fim; i++) {

            double fx_local = 0.0;
            double fy_local = 0.0;

            for (int j = 0; j < N; j++) {
                calcula_forca(corpos[i], corpos[j], fx_local, fy_local);
            }

            fx[i] = fx_local;
            fy[i] = fy_local;
        }

        auto t_forca_fim = std::chrono::high_resolution_clock::now();
        // Vetores auxiliares apenas no rank 0
        std::vector<double> fx_global;
        std::vector<double> fy_global;

        if (rank == 0) {
            fx_global.resize(N);
            fy_global.resize(N);
        }

        // Redução do vetor fx
        if (rank == 0) {
            // Rank 0 recebe o resultado final
            MPI_Reduce(fx.data(), fx_global.data(),
                    N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            // Outros ranks apenas enviam seus dados
            MPI_Reduce(fx.data(), nullptr,
                    N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        // Redução do vetor fy
        if (rank == 0) {
            MPI_Reduce(fy.data(), fy_global.data(),
                    N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(fy.data(), nullptr,
                    N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

    
        auto t_update_inicio = std::chrono::high_resolution_clock::now();

        // Atualização apenas no rank 0 para consolidar os dados
        if (rank == 0) {
 
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                corpos[i].vx += fx[i] / corpos[i].massa * dt;
                corpos[i].vy += fy[i] / corpos[i].massa * dt;

                corpos[i].x += corpos[i].vx * dt;
                corpos[i].y += corpos[i].vy * dt;
            }

            auto t_update_fim = std::chrono::high_resolution_clock::now();
            auto t_total_fim = std::chrono::high_resolution_clock::now();

            double tempo_forca = std::chrono::duration<double>(t_forca_fim - t_forca_inicio).count();
            double tempo_update = std::chrono::duration<double>(t_update_fim - t_update_inicio).count();
            double tempo_total = std::chrono::duration<double>(t_total_fim - t_total_inicio).count();

            std::cout << "Simulacao concluida para " << N << " corpos.\n";
            std::cout << "Tempo (forcas): " << tempo_forca << " s\n";
            std::cout << "Tempo (update): " << tempo_update << " s\n";
            std::cout << "Tempo total: " << tempo_total << " s\n";
        }

        MPI_Finalize();
        return 0;
    }    

    ```

??? note "Com Broadcast + Gather"
    ```cpp
    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <chrono>
    #include <mpi.h>
    #include <omp.h>

    const double G = 6.674e-11;

    struct Corpo {
        double x, y;
        double vx, vy;
        double massa;
    };

    void calcula_forca(const Corpo& a, const Corpo& b, double& fx, double& fy) {
        double dx = b.x - a.x;
        double dy = b.y - a.y;

        double dist = std::sqrt(dx * dx + dy * dy) + 1e-9;
        double F = G * a.massa * b.massa / (dist * dist);

        fx += F * dx / dist;
        fy += F * dy / dist;
    }

    int main(int argc, char** argv) {

        // Inicialização do MPI
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int N = 85000;

        std::vector<Corpo> corpos(N);

        // Inicialização apenas no rank 0
        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                corpos[i].x = i * 1.0;
                corpos[i].y = i * 0.5;
                corpos[i].vx = 0.0;
                corpos[i].vy = 0.0;
                corpos[i].massa = 1e20;
            }
        }

        // Broadcast para todos os processos
        MPI_Bcast(corpos.data(), N * sizeof(Corpo), MPI_BYTE, 0, MPI_COMM_WORLD);

        double dt = 0.01;

        // Divisão do trabalho
        int tam_local = N / size;
        int inicio = rank * tam_local;

        int fim;
        if (rank == size - 1) {
            fim = N;
        } else {
            fim = inicio + tam_local;
        }

        tam_local = fim - inicio;

        // Vetores locais (cada processo só calcula sua parte)
        std::vector<double> fx_local(tam_local, 0.0);
        std::vector<double> fy_local(tam_local, 0.0);

        auto t_total_inicio = std::chrono::high_resolution_clock::now();
        auto t_forca_inicio = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (int i = inicio; i < fim; i++) {

            double fx_temp = 0.0;
            double fy_temp = 0.0;

            for (int j = 0; j < N; j++) {
                calcula_forca(corpos[i], corpos[j], fx_temp, fy_temp);
            }

            fx_local[i - inicio] = fx_temp;
            fy_local[i - inicio] = fy_temp;
        }

        auto t_forca_fim = std::chrono::high_resolution_clock::now();

        // Vetores globais apenas no rank 0
        std::vector<double> fx;
        std::vector<double> fy;

        if (rank == 0) {
            fx.resize(N);
            fy.resize(N);
        }

        // Junta os resultados no rank 0
        MPI_Gather(fx_local.data(), tam_local, MPI_DOUBLE,
                fx.data(), tam_local, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

        MPI_Gather(fy_local.data(), tam_local, MPI_DOUBLE,
                fy.data(), tam_local, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

        auto t_update_inicio = std::chrono::high_resolution_clock::now();

        // Atualização apenas no rank 0
        if (rank == 0) {

            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                corpos[i].vx += fx[i] / corpos[i].massa * dt;
                corpos[i].vy += fy[i] / corpos[i].massa * dt;

                corpos[i].x += corpos[i].vx * dt;
                corpos[i].y += corpos[i].vy * dt;
            }

            auto t_update_fim = std::chrono::high_resolution_clock::now();
            auto t_total_fim = std::chrono::high_resolution_clock::now();

            double tempo_forca = std::chrono::duration<double>(t_forca_fim - t_forca_inicio).count();
            double tempo_update = std::chrono::duration<double>(t_update_fim - t_update_inicio).count();
            double tempo_total = std::chrono::duration<double>(t_total_fim - t_total_inicio).count();

            std::cout << "Simulacao concluida para " << N << " corpos.\n";
            std::cout << "Tempo (forcas): " << tempo_forca << " s\n";
            std::cout << "Tempo (update): " << tempo_update << " s\n";
            std::cout << "Tempo total: " << tempo_total << " s\n";
        }

        MPI_Finalize();
        return 0;
    }
    ```