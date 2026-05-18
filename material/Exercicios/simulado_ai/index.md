
# Simulado da Avaliação Intermediária de Supercomp, 1 semestre de 2026

## Questões Teóricas:
A resposta para questões teóricas devem estar em um arquivo "respostas_teoricas.txt"

1) O que é um sistema de HPC? (0.5 ponto)
??? note "Gabarito"
    High-Performance Computing (HPC) refere-se ao uso de supercomputadores e clusters de computadores para resolver problemas computacionalmente complexos. HPC é essencial em campos como ciência, engenharia e finanças, onde grandes volumes de dados precisam ser processados rapidamente.[Fonte](https://insper.github.io/supercomp/teoria/aula02/)


2) Qual é a função do SLURM em um sistema de HPC? (0.5 ponto)
??? note "Gabarito"
    O Slurm é responsável por gerenciar a execução das tarefas que você submete, distribuindo-as eficientemente pelos recursos de computação disponíveis. [Fonte](https://insper.github.io/supercomp/teoria/aula02/)

3) Explique por que um loop com dependência entre iterações não pode ser paralelizado diretamente. (0.5 ponto)
??? note "Gabarito"
    Se um loop tem dependência entre as iterações, não dá pra paralelizar direto. Isso acontece porque uma iteração precisa do resultado da anterior, então as threads teriam que esperar — e aí perde a vantagem da paralelização.
    Uma forma de resolver é paralelizar só as partes que não dependem umas das outras e usar sincronização quando necessário. Outra ideia é dividir o loop em blocos, deixando as dependências dentro de cada bloco e executando os blocos em paralelo. [Fonte](https://insper.github.io/supercomp/aulas/aula07/)

4) Qual a diferença entre paralelismo com memória compartilhada e paralelismo com memória distribuída. (0.5 ponto)
??? note "Gabarito"
    No paralelismo com memória compartilhada, os threads acessam a mesma RAM e executam na mesma máquina, usando múltiplos núcleos, cada core pode executar uma ou mais threads simultaneamente. 
    No paralelismo com memória distribuída, os processos executam em máquinas diferentes. A comunicação é feita por troca de mensagens, usando protocolos como TCP/IP. Com o MPI cada processo possui um identificador único (rank), que é usado para organizar a comunicação e a distribuição das tarefas.

    O modelo de memória compartilhada favorece a escalabilidade vertical (mais recursos na mesma máquina), enquanto o modelo distribuído permite escalabilidade horizontal (uso de várias máquinas).[fonte](https://insper.github.io/supercomp/teoria/slides/Aula09.pdf)

## Questões práticas
Para cada questão, você deve organizar sua resposta da seguinte forma:
* Questões de implementação devem ser entregues em arquivos no formato: Qx.cpp (onde x é o número da questão)

* Questões que exigem execução em cluster devem incluir também o respectivo arquivo de submissão: run_Qx.slurm

* Questões teóricas devem ser respondidas em arquivos de texto: Qx.txt


## **Exercício 1 - Otimizações em CPU - (2 pontos)**

Abaixo temos um código base implementado de forma sequencial em CPU, da forma menos eficiente que eu consegui imaginar, ele realiza o cálculo da soma, média, quantidade de picos e quantidade de vales de um sensor fictício. Quando executado na fila **gpu** do cluster Franky, o programa produz os seguintes resultados:

```bash
Processando vetor de 123456789 elementos...
Soma final: 7.60016e+09
Média final: 61.5613
Quantidade de picos: 123456
Quantidade de vales: 123456
Tempo CPU sequencial: 991.454 ms
```

### Implemente as seguintes otimizções:

- Melhore o acesso aos dados, usando passagem por referência ou por ponteiro **+0.2 pontos**
- Aplicar Tilling para melhorar a localidade espacial e temporal e aproveitar melhor a memória cache **+0.5 pontos**
- Aplicar paralelismo em CPU usando OpenMP **+0.8 pontos**
- Criar um arquivo 'run.slurm' que solicita adequadamete os recursos de hardware para o Cluster Franky executar a versão com paralelismo em **CPU**. **+0.5 pontos**

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// Função ineficiente: inicializa vetor por valor
std::vector<float> inicializar_vetor(std::vector<float> v, size_t N) {
    for (size_t i = 0; i < N; i++) {
        v[i] = (i % 1000) * 0.123f;
    }
    return v; // retorna uma cópia
}

// Função muito ruim: processa os dados por valor
std::vector<float> processar_dados(
    std::vector<float> v,
    size_t N,
    double &soma,
    size_t &num_picos,
    size_t &num_vales
) {
    std::vector<float> out(N, 0.0f);

    soma = 0.0;
    num_picos = 0;
    num_vales = 0;

    for (size_t i = 1; i < N - 1; i++) {

        float a = v[i - 1];
        float b = v[i];
        float c = v[i + 1];

        // Média local
        float media_local = (a + b + c) / 3.0f;

        // Pico
        bool pico = (b > a && b > c);

        // Vale
        bool vale = (b < a && b < c);

        if (pico) num_picos++;
        if (vale) num_vales++;

        out[i] = media_local + (pico ? b : 0.0f);

        soma += out[i];
    }

    return out; // retorna outra cópia gigante </3
}

// Função horrivel: calcula média por valor
double calcular_media(std::vector<float> out, double soma, size_t N) {
    return soma / N;
}


int main() {
    const size_t N = 123'456'789;

    // Vetor inicial
    std::vector<float> v(N);

    std::cout << "Processando vetor de " << N << " elementos...\n";

    // Inicialização
    v = inicializar_vetor(v, N);

    auto t0 = std::chrono::high_resolution_clock::now();

    double soma = 0.0;
    size_t num_picos = 0;
    size_t num_vales = 0;

    // Processamento sequencial do vetor
    std::vector<float> out = processar_dados(v, N, soma, num_picos, num_vales);

    // Mais uma cópia desnecessária de dados para calcular média :'(
    double media_final = calcular_media(out, soma, N);

    auto t1 = std::chrono::high_resolution_clock::now();
    double tempo_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Soma final: "           << soma        << "\n";
    std::cout << "Média final: "          << media_final << "\n";
    std::cout << "Quantidade de picos: "  << num_picos   << "\n";
    std::cout << "Quantidade de vales: "  << num_vales   << "\n";
    std::cout << "Tempo CPU sequencial: " << tempo_ms    << " ms\n";

    return 0;
}


```
??? note "Gabarito"
    ```cpp
    // Autor: Emil
    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <chrono>
    #include <omp.h>
    #include <iomanip>

    // Função ineficiente: inicializa vetor por valor
    // O que foi feito: 
    // foi trocado a cópia do objeto vetor por uma referência
    // N mantive como valor pois size_t é um dado muito pequeno, e não estamos
    // alterando ele
    void inicializar_vetor(std::vector<float> &v, size_t N) {
        //Como não preciso recuperar dados, é uma inicialização simples, não é
        //necessário fazer tilling
        //Utilizado simd para inicialização mais rápida do vetor
        #pragma omp simd
        for (size_t i = 0; i < N; i++) {
            v[i] = (i % 1000) * 0.123f;
        }
    }

    // Função muito ruim: processa os dados por valor
    // O que foi feito: 
    // Modificado para que a função para que não seja necessário fazer cópias dos
    // vetores, no lugar de retornar utilizamos uma referência de saida também, que
    // estará pré alocada.
    void processar_dados(
                        std::vector<float> &v,
                        size_t N,
                        double &soma,
                        size_t &num_picos,
                        size_t &num_vales,
                        std::vector<float> &out
                        ) {

        soma = 0.0;
        num_picos = 0;
        num_vales = 0;

        //O que  foi feito:
        //Tilling aplicado no próximo loop

        size_t tile_size = 2048;


        //Paralelização do for externo do tilling
        #pragma omp parallel for reduction(+:soma, num_picos, num_vales)
        for (size_t i = 1; i < N - 1; i+=tile_size) {

            //Simd utilizado para pois já fizemos ele caber no Cahe L1
            #pragma omp simd reduction(+:soma, num_picos, num_vales)
            for(size_t j = i; j < std::min(i+tile_size, N-1); j++){

                float a = v[j - 1];
                float b = v[j];
                float c = v[j + 1];

                // Média local
                float media_local = (a + b + c) / 3.0f;

                // Pico
                bool pico = (b > a && b > c);

                // Vale
                bool vale = (b < a && b < c);

                if (pico) num_picos++;
                if (vale) num_vales++;

                out[j] = media_local + (pico ? b : 0.0f);

                soma += out[j];
            }
        }
    }

    // Função horrivel: calcula média por valor
    // Removi o parametro não utilizado e inseri a diretiva inline que indica para
    // o compilador que pode substituir o chamada da função por seu corpo.
    inline double calcular_media(double soma, size_t N) {
        return soma / N;
    }


    int main() {
        const size_t N = 123'456'789;

        // Vetor inicial
        std::vector<float> v(N);
        std::vector<float> out(N, 0);

        std::cout << "Processando vetor de " << N << " elementos...\n";

        // Inicialização
        inicializar_vetor(v, N);

        auto t0 = std::chrono::high_resolution_clock::now();

        double soma = 0.0;
        size_t num_picos = 0;
        size_t num_vales = 0;

        // Processamento sequencial do vetor
        processar_dados(v, N, soma, num_picos, num_vales, out);

        // Mais uma cópia desnecessária de dados para calcular média :'(
        double media_final = calcular_media(soma, N);

        auto t1 = std::chrono::high_resolution_clock::now();
        double tempo_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "Soma final: "           << soma        << "\n";
        std::cout << "Média final: "          << media_final << "\n";
        std::cout << "Quantidade de picos: "  << num_picos   << "\n";
        std::cout << "Quantidade de vales: "  << num_vales   << "\n";
        std::cout << "Tempo paralelo: " << tempo_ms    << " ms\n";

        return 0;
    }

    /**
    * Log:
    * 1. Melhorando o acesso dos dados passando por referência
    * 2. Aplicar Tilling, selecionei 2048 com tile_size
    * 3. Aplicar o OpemMP para paralelização de loops
    **/

    ```
    Para gerar o binário corretamente

    ```cpp
    g++ -fopenmp -O3 Q1.cpp -o Q1
    ```
    Arquivo SLURM

    ```bash
    #!/bin/bash
    #SBATCH --job-name=Q1           # nome do job
    #SBATCH --output=Q1.out         # arquivo de saída
    #SBATCH --cpus-per-task=8               # 8 threads para usar
    #SBATCH --time=00:05:00                 # tempo máximo de execução
    #SBATCH --mem=2G                        # Memória 
    #SBATCH --partition=gpu              # fila

    # garante que o OpenMP use exatamente os recursos alocados pelo SLURM
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

    # executa o binário
    ./Q1
    ```
    Resultado Paralelo
    ```bash
    Processando vetor de 123456789 elementos...
    Soma final: 7.60016e+09
    Média final: 61.5613
    Quantidade de picos: 123456
    Quantidade de vales: 123456
    Tempo paralelo: 20.9825 ms
    ```

    Sequencial
    ```bash
    Processando vetor de 123456789 elementos...
    Soma final: 7.60016e+09
    Média final: 61.5613
    Quantidade de picos: 123456
    Quantidade de vales: 123456
    Tempo CPU sequencial: 991.454 ms
    ```



## **Exercício 2 — Cálculo de PI com Monte Carlo**

O código abaixo implementa o cálculo de PI usando o método de Monte Carlo de forma **sequencial**. Seu objetivo é otimizar este programa usando OpenMP.

### **Parte 1 - Paralelismo com OpenMP (2 pontos)**

* **0 pontos** → O aluno **não paralelizou** o código, ou a versão apresentada **não compila/não executa**.
 
* **0.5 ponto** → O aluno tentou paralelizar, mas não tratou corretamente a variável compartilhada (`dentro`) causando **condição de corrida**. O programa roda, mas os resultados são inconsistentes.

* **1 ponto** → O aluno paralelizou corretamente, garantindo que o código execute sem conflitos.

* **2 pontos** -> O aluno paralelizou corretamente, garantindo que o código execute sem conflitos, ajustou o número de pontos e configurou corretamente os recursos no SLURM, de modo que o programa gera uma estimativa de PI com precisão 3,1415 em menos de 10 segundos.

??? note "Gabarito"
    ```cpp
    // Autor: Emil
    #include <iostream>
    #include <random>
    #include <chrono>
    #include <iomanip>
    #include <omp.h>

    using namespace std;

    int main(int argc, char** argv) {
        long long N = (argc > 1 ? atoll(argv[1]) : 850000000); // quantidade de pontos
                                                    
        mt19937 rng(42); // gerador de números aleatórios
        uniform_real_distribution<double> dist(0.0, 1.0);

        long long dentro = 0;

        chrono::high_resolution_clock::time_point t0 = chrono::high_resolution_clock::now();

    
        #pragma omp parallel for reduction(+:dentro)
        for (long long i = 0; i < N; i++) {
            double x = dist(rng);
            double y = dist(rng);
            if (x * x + y * y <= 1.0) {
                dentro++;
            }
        }

        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

        double pi = 4.0 * (double)dentro / (double)N;
        double tempo = chrono::duration<double>(t1 - t0).count();
        cout << fixed << setprecision(4);
        cout << "N = " << N << "  pi = " << pi << "  tempo = " << tempo << "s" << endl;

        return 0;
    }

    /** 
    * Mudanças:
    * 1. Adicionado o Pragma parallel com reduction no loop que calcula o dentro
    * 2. Subi o N para aumentar a precisão no cálculo do PI
    **/

    ```
    Para compilar corretamente:

    ```cpp
    g++ -fopenmp -O3 Q2_p1.cpp -o Q2_p1 
    ```

    Arquivo SLURM
    ```bash
    #!/bin/bash
    #SBATCH --job-name=Q2           # nome do job
    #SBATCH --output=saida.out         # arquivo de saída
    #SBATCH --cpus-per-task=4               # 4 threads para usar
    #SBATCH --time=00:05:00                 # tempo máximo de execução
    #SBATCH --mem=2G                        # Memória
    #SBATCH --partition=normal              # fila

    # garante que o OpenMP use exatamente os recursos alocados pelo SLURM
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

    # executa o binário
    ./Q2_p1

    ```

    Resultado:

    ```
    N = 850000000  pi = 3.1415  tempo = 6.0315s
    ```


---

### **Parte 2 - Distribuição com MPI (2 pontos)**
Você deve:
    - Distribuir o número total de pontos N entre os processos MPI
    - Cada processo deve calcular sua contagem local de pontos 
    - Utilizar sementes diferentes para o gerador de números aleatórios em cada processo
    - Centralizar as respostas no Rank0 para exibir o valor de PI calculado.

* **0 pontos** → Não implementou MPI ou o código não compila/não executa

* **0.5 ponto** → Implementação incompleta, apresentando problemas como:
    Todos os processos utilizam a mesma seed;
    Problema de distribuição de dados (todos os nós fazem o mesmo trabalho)

* **2 pontos** → Implementação correta, incluindo:
    Distribuição adequada de N entre os processos;
    Uso de seeds independentes por processo;
    Cálculo correto de PI exibido pelo Rank 0 usando o resultado das contas realizadas nos nós de computação

??? note "Gabarito"
    ```cpp
    #include <iostream>
    #include <random>
    #include <chrono>
    #include <iomanip>
    #include <omp.h>
    #include <mpi.h>

    using namespace std;

    int main(int argc, char** argv) {
        // Inicializa o ambiente MPI
        MPI_Init(&argc, &argv);

        // rank → identifica o processo atual
        // size → quantidade total de processos
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Quantidade de pontos
        long long N = 850000000;

        // Divide o trabalho total (N) entre os processos
        long long chunk = N / size;   // parte igual para todos
        long long resto = N % size;   // sobra

        long long localN = chunk;

        // Rank 0 fica com o resto 
        if (rank == 0) {
            localN = localN + resto;
        }

        // Os outros Ranks processam o localN que receberam da divisão de trabalho
        long long dentro = 0;

        auto t0 = chrono::high_resolution_clock::now();
        // parallel → cria as threads
        // reduction → cada thread usa sua própria cópia de "dentro" e soma tudo no final
        #pragma omp parallel reduction(+:dentro)
        {
            int thread_id = omp_get_thread_num();

            unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
          
          // cada thread tem uma seed diferente para explorar diferentes intervalos do espaço de busca
            seed = seed + rank * 1000 + thread_id;
            mt19937 rng(seed);
            uniform_real_distribution<double> dist(0.0, 1.0);
          
          // omp for → divide as iterações do loop entre as threads
            #pragma omp for
            for (long long i = 0; i < localN; i++) {

                double x = dist(rng);
                double y = dist(rng);
               
                if (x * x + y * y <= 1.0) {
                    dentro = dentro + 1;
                }
            }
        }


        long long total_dentro = 0;

        // Redução MPI:
        // soma os valores de "dentro" de todos os processos
        // o resultado final vai para o processo de rank 0
        MPI_Reduce(&dentro,
                &total_dentro,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                0,
                MPI_COMM_WORLD);

        auto t1 = chrono::high_resolution_clock::now();

        // Apenas o processo 0 imprime o resultado final
        if (rank == 0) {

            double pi = 4.0 * (double)total_dentro / (double)N;
            double tempo = chrono::duration<double>(t1 - t0).count();

            cout << fixed << setprecision(4);
            cout << "N = " << N << endl;
            cout << "PI = " << pi << endl;
            cout << "Tempo = " << tempo << " s" << endl;
        }

        MPI_Finalize();
        return 0;
    }
    ```
    Para compilar corretamente:
    ```bash
    mpic++ -fopenmp -O3 Q2_p2.cpp -o Q2_p2
    ```
    Arquivo SLURM
    ```bash
    #!/bin/bash
    #SBATCH --job-name=Q2_p2           # nome do job
    #SBATCH --output=Q2_p2.out         # arquivo de saída
    #SBATCH --partition=normal
    #SBATCH --time=00:05:00                 # tempo máximo de execução
    #SBATCH --mem=2G                        # Memória
    #SBATCH --nodes=4                       # Nós de computação
    #SBATCH --ntasks=4                      # Processos
    #SBATCH --cpus-per-task=4               #  threads para usar

    # garante que o OpenMP use exatamente os recursos alocados pelo SLURM
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

    # executa o binário
    mpirun -np $SLURM_NTASKS ./Q2_p2
    ```
    Resultado:
    ```bash
    N = 850000000
    PI = 3.1415
    Tempo = 1.897940 s
    ```


```cpp
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
using namespace std;

int main(int argc, char** argv) {
    long long N = (argc > 1 ? atoll(argv[1]) : 100000); // quantidade de pontos

    mt19937 rng(42); // gerador de números aleatórios
    uniform_real_distribution<double> dist(0.0, 1.0);

    long long dentro = 0;

    chrono::high_resolution_clock::time_point t0 = chrono::high_resolution_clock::now();

    // Código sequencial
    for (long long i = 0; i < N; i++) {
        double x = dist(rng);
        double y = dist(rng);
        if (x * x + y * y <= 1.0) {
            dentro++;
        }
    }

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    double pi = 4.0 * (double)dentro / (double)N;
    double tempo = chrono::duration<double>(t1 - t0).count();
    cout << fixed << setprecision(4);
    cout << "N = " << N << "  pi = " << pi << "  tempo = " << tempo << "s" << endl;

    return 0;
}
```

## **Exercício 3 — Otimização de Rotas para Entregas com Bicicleta (2 pontos)**

Voltamos ao problema das entregas, mas agora estamos auxiliando o Rodolfo, um jovem audacioso que realiza entregas de bicicleta.

Devemos considerar que Rodolfo possui limitações físicas e operacionais:
    - Ele consegue carregar no máximo 2 pacotes por vez
    - Precisa retornar ao ponto de coleta sempre que esvaziar a carga
    - Sua meta diária é realizar 60 entregas
    - Os pontos de entrega estão [disponívies aqui](pontos.txt)

O objetivo é:

Determinar uma rota de entregas que minimize a distância total percorrida (custo), respeitando a limitação de carga.

Rúbrica:

* **0** - Não compila, não é possível testar a implementação.   
* **+0.2** - Implementou a heuristica hill climbing aleatória, mas sem otimizações.
* **+0.2** - Otimizou a heurísitca melhorando seu tempo de execução, mas ainda tem uma implementação sequencial.
* **+0.6** - Paralelizou o algorítimo e conseguiu melhorar o tempo em relação a versão sequencial.
* **+0.4** - Montou o arquivo SLURM de forma correta e exibiu uma tabela comprovando que a versão paralela é mais eficiente que a versão sequencial
* **+0.6** - Conseguiu explicar os resultados obtidos nos seus testes com as otimizações aplicadas de forma clara e objetiva.

??? note "Gabarito"
    ```cpp
    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <random>
    #include <algorithm>
    #include <numeric>
    #include <chrono>
    #include <fstream>
    #include <omp.h>

    using namespace std;

    struct Ponto {
        double x, y;
    };

    // Lê os pontos do arquivo
    vector<Ponto> lerPontos(string nome) {
        ifstream arq(nome);
        vector<Ponto> pontos;
        double x, y;

        while (arq >> x >> y) {
            pontos.push_back({x, y});
        }

        return pontos;
    }

    // Distância entre dois pontos
    double distancia(Ponto a, Ponto b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return sqrt(dx * dx + dy * dy);
    }

    // Calcula o custo da rota considerando capacidade = 2
    double calcularCusto(vector<int>& rota,
                        vector<Ponto>& entregas,
                        Ponto casa,
                        Ponto coleta) {

        double custo = 0.0;

        custo += distancia(casa, coleta);

        int carga = 2;
        Ponto atual = coleta;

        for (int idx : rota) {

            // se acabou a carga → volta para coleta
            if (carga == 0) {
                custo += distancia(atual, coleta);
                atual = coleta;
                carga = 2;
            }

            custo += distancia(atual, entregas[idx]);
            atual = entregas[idx];
            carga--;
        }

        custo += distancia(atual, casa);
        return custo;
    }

    // Gera uma rota inicial 
    vector<int> rotaAleatoria(int n, mt19937& rng) {
        vector<int> rota(n);
        iota(rota.begin(), rota.end(), 0);
        shuffle(rota.begin(), rota.end(), rng);
        return rota;
    }

    // ===============================
    // HILL CLIMBING ALEATÓRIO
    // ===============================
    double hillClimbing(vector<Ponto>& entregas,
                        Ponto casa,
                        Ponto coleta,
                        mt19937& rng) {

        int n = entregas.size();

        vector<int> atual = rotaAleatoria(n, rng);
        double melhor = calcularCusto(atual, entregas, casa, coleta);

        uniform_int_distribution<int> dist(0, n - 1);

        bool melhorou = true;

        while (melhorou) {
            melhorou = false;

            // tenta vizinhos aleatórios (troca de posições)
            for (int t = 0; t < 2000; t++) {

                int i = dist(rng);
                int j = dist(rng);

                if (i == j) continue;

                swap(atual[i], atual[j]);

                double custo = calcularCusto(atual, entregas, casa, coleta);

                if (custo < melhor) {
                    melhor = custo;
                    melhorou = true;
                    break;
                } else {
                    swap(atual[i], atual[j]); // desfaz
                }
            }
        }

        return melhor;
    }

    // ===============================
    // SEQUENCIAL
    // ===============================
    double resolverSequencial(vector<Ponto>& entregas,
                            int restarts,
                            unsigned seed) {

        mt19937 rng(seed);
        double melhor = 1e18;

        for (int i = 0; i < restarts; i++) {

            // cada iteração executa um hill climbing completo
            double custo = hillClimbing(entregas, {0,0}, {5,5}, rng);

            if (custo < melhor) {
                melhor = custo;
            }
        }

        return melhor;
    }

    // ===============================
    // PARALELO 
    // ===============================
    double resolverParalelo(vector<Ponto>& entregas,
                        int restarts,
                        unsigned seed) {

    double melhor = 1e18;

    // paraleliza o loop de restarts
    #pragma omp parallel for
    for (int i = 0; i < restarts; i++) {

        // cada iteração explora um intervalo do espaço de busca diferente
        mt19937 rng(seed + i);

        double custo = hillClimbing(entregas, {0,0}, {5,5}, rng);

        // Operação critica para não bagunçar os resultados de custo de cada thread
        #pragma omp critical
        {
            if (custo < melhor) {
                melhor = custo;
            }
        }
    }

    return melhor;
    }
    // ===============================
    // MAIN
    // ===============================
    int main() {

        vector<Ponto> entregas = lerPontos("pontos.txt");

        cout << "Pontos: " << entregas.size() << endl;

        int restarts = 200;

        // -------- SEQUENCIAL --------
        auto t0 = chrono::high_resolution_clock::now();
        double custoSeq = resolverSequencial(entregas, restarts, 42);
        auto t1 = chrono::high_resolution_clock::now();

        // -------- PARALELO --------
        auto t2 = chrono::high_resolution_clock::now();
        double custoPar = resolverParalelo(entregas, restarts, 42);
        auto t3 = chrono::high_resolution_clock::now();

        double tempoSeq = chrono::duration<double>(t1 - t0).count();
        double tempoPar = chrono::duration<double>(t3 - t2).count();

        cout << "\nSequencial:\n";
        cout << "Custo: " << custoSeq << endl;
        cout << "Tempo: " << tempoSeq << endl;

        cout << "\nParalelo:\n";
        cout << "Custo: " << custoPar << endl;
        cout << "Tempo: " << tempoPar << endl;

        return 0;
    }

    ```

    Para compilar corretamente

    ```bash
    g++ -fopenmp -O3 Q3.cpp -o Q3
    ```

    Arquivo SLURM

    ```bash
    #!/bin/bash
    #SBATCH --job-name=Q3
    #SBATCH --output=S_Q3.out
    #SBATCH --cpus-per-task=16
    #SBATCH --time=00:05:00
    #SBATCH --mem=2G
    #SBATCH --partition=normal

    echo "Teste de escalabilidade"
    echo "========================"

    # lista de threads que queremos testar
    for t in 2 4 6 8 10 12 14 16
    do
        export OMP_NUM_THREADS=$t

        echo ""
        echo "Executando com $t threads..."

        # Binário recebendo a lista de pontos
        ./Q3 pontos.txt
    done
    ```

    Resultado dos testes

    | Threads | Tempo Sequencial (s) | Tempo Paralelo (s) | 
    |--------|----------------------|---------------------|
    | 2      | 0.168033             | 0.0862287           |
    | 4      | 0.168010             | 0.0444519           |  
    | 6      | 0.168018             | 0.0312003           | 
    | **8**      | **0.168029**             | **0.0251245**           |
    | 10     | 0.168049             | 0.0322822           | 
    | 12     | 0.167883             | 0.0295553           |
    | 14     | 0.167998             | 0.0301920           | 
    | 16     | 0.162080             | 0.0333125           |

    O ganho de desempenho foi obtido ao paralelizar os restarts do algoritmo Hill Climbing. Foi implementado o paralelismo nos restarts permitindo que múltiplas buscas ocorram simultaneamente.Além disso, foram utilizadas sementes diferentes para evitar que as threads trabalhem no mesmo espaço de busca, uma região crítica foi criada apenas para atualizar o melhor resultado sem sobrescrever ou bagunçar os dados. O melhor desempenho ocorreu com 8 threads, pois a partir desse ponto o overhead de criação e sincronização das threads passa a reduzir os ganhos do paralelismo.