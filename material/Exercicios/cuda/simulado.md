

# Simulado da Avaliação Final de Supercomp.

Este simulado tem como objetivo ajudar você a identificar quais conteúdos estudados até aqui foram bem compreendidos e quais ainda precisam de maior atenção antes da Avaliação Final.


Para que o resultado represente seu nível atual de preparação, realize as atividades com seriedade e honestidade, seguindo as mesmas condições de uma prova.


A pontuação de cada exercício será atribuída mediante a apresentação de uma tentativa de resolução, mesmo que parcial. O feedback indicará os pontos que precisam ser corrigidos ou aprimorados, sem desconto na pontuação. Entretanto, exercícios enviados sem qualquer tentativa de resolução não receberão nota.


Durante as avaliações desta disciplina, é permitida a consulta ao site da disciplina, ao PrairieLearn, aos seus próprios códigos e às suas anotações.


Consulta a colegas, a Inteligência artificial ou a recursos que não estão explicitamente indicados neste texto não é permitido e constituirão violações ao Código de Ética e de Conduta e acarretarão sanções nele previstas. Faça o seu trabalho de maneira ética!

Faça a submissão das suas respostas pelo blackboard.

## **Monitoramento de Sensores em Tempo Real (4,0 pontos)**

Você está desenvolvendo um sistema de monitoramento ambiental que recebe continuamente dados de sensores distribuídos por uma cidade. Cada sensor registra um valor de intensidade de poluição, armazenado em um vetor de números reais (float).

Para analisar a contribuição acumulada dessas medições, o sistema precisa realizar duas etapas de processamento:

Soma acumulada (Prefix Sum): calcular a soma dos valores desde o primeiro sensor até cada posição do vetor.

Normalização: dividir cada soma acumulada pelo valor total das medições, obtendo resultados entre 0 e 1.

Por exemplo:

* Entrada:        [2, 1, 3, 4]
* Soma acumulada: [2, 3, 6, 10]
* Normalização:   [0.2, 0.3, 0.6, 1.0]

Como o sistema processa grandes volumes de dados, essas operações devem ser executadas em GPU utilizando CUDA.
Implemente um programa que recebe um vetor de floats e realiza **duas etapas de processamento em GPU**:

1. Calcule o prefix sum (soma cumulativa) de todos os elementos.
   Exemplo: `[2, 1, 3, 4] → [2, 3, 6, 10]`.

2. Use o último elemento (a soma total) para normalizar todos os valores:
   `[2, 3, 6, 10] → [0.2, 0.3, 0.6, 1.0]`.



### **Rubrica de Avaliação — Total: 4,0 pontos**

| Pontuação      | Critério                        | Descrição                                                                                                                  |
| -------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **+1,0 ponto**  | Implementação em GPU Síncrona   | Implementa corretamente as etapas de Prefix Sum e normalização em GPU, utilizando execução síncrona.                       |
| **+2,5 pontos** | Implementação em GPU Assíncrona | Utiliza CUDA Streams e operações assíncronas, respeitando as dependências entre as etapas e otimizando o uso de CPU e GPU. |
| **+0,5 ponto** | Slurm — Cluster Franky          | Configura corretamente a execução no ambiente HPC (`srun` ou `sbatch`), com parâmetros adequados para utilização da GPU.   |
| **4,0 pontos** | **Pontuação máxima**            |                                                                                                                            |


??? Note "Código base"
    '```cpp
    #include <iostream>
    #include <vector>
    #include <chrono>

    int main() {
        const size_t N = 87654;
        std::vector<float> v(N);
        std::vector<float> prefix(N);

        auto inicio = std::chrono::high_resolution_clock::now();

        // Gera o vetor
        for (size_t i = 0; i < N; ++i)
            v[i] = static_cast<float>((i + 1) * 2);

        // prefix sum
        prefix[0] = v[0];
        for (size_t i = 1; i < N; ++i)
            prefix[i] = prefix[i - 1] + v[i];

        // Normalização
        float total = prefix.back();  // soma total
        for (size_t i = 0; i < N; ++i)
            prefix[i] /= total;

        auto fim = std::chrono::high_resolution_clock::now();
        double tempo = std::chrono::duration<double, std::milli>(fim - inicio).count();

        // RESULTADOS
        std::cout << "\nÚltimos 10 valores do prefix sum:\n[ ";
        for (size_t i = N - 10; i < N; ++i)
            std::cout << prefix[i] * total << " ";
        std::cout << "]\n";

        std::cout << "Últimos 10 valores normalizados:\n[ ";
        for (size_t i = N - 10; i < N; ++i)
            std::cout << prefix[i] << " ";
        std::cout << "]\n";

        std::cout << "\nTempo CPU: " << tempo << " ms\n";
    }
    ```
        

### **Laboratório de previsão climática (3 pontos)**

Você está trabalhando em um laboratório de previsão climática que usa supercomputadores para simular o comportamento da atmosfera.
A região de interesse é dividida em uma grade de pontos (latitude × longitude), e em cada ponto são armazenadas variáveis como temperatura, pressão e umidade.

Para prever o clima nas próximas horas, é preciso aplicar um modelo matemático que combina essas variáveis em cada ponto da grade. Essa combinação pode ser representada como a **multiplicação de matrizes**:

* A matriz **A** contém os valores medidos nos sensores (temperatura, pressão, umidade).
* A matriz **B** representa os coeficientes do modelo físico (como cada variável influencia a previsão).
* A matriz **C = A × B** contém os novos valores ajustados, que alimentam os próximos passos da simulação.

Por serem matrizes muito grandes (milhares de linhas e colunas), não é viável calcular tudo de forma sequencial. Otimize o código para que seja possível realizar essa computação de forma paralela em GPU. 

Sem nenhuma otimização o código gera esse resultado, se executado no Cluster Franky:

```bash
N = 2048
Tempo: 71.6775 s
Para validação dos dados:
Primeiro elemento da matriz C[0][0] = 2170.03
Elemento do meio da matriz C[N/2][N/2] = 2190.39
Ultimo elemento da matriz C[N-1][N-1] = 2218.17
```

### **Rubrica de Avaliação**

| Pontuação      | Critério               | Descrição                                                                                                                    |
| -------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **2,0 pontos** | Paralelismo Simples    | Solução CUDA funcional, com kernel implementado corretamente e transferência adequada de dados entre CPU e GPU.              |
| **3,0 pontos** | Paralelismo com Tiling | Aplicação correta da técnica de *tiling*, apresentando melhor desempenho em relação à implementação com paralelismo simples. |


??? Note "Código base"
    ```cpp
    #include <iostream>
    #include <vector>
    #include <chrono>
    #include <cmath>

    using namespace std;

    // Multiplicação sequencial em CPU.
    void multiplicacaoCPU(
        const vector<double>& A,
        const vector<double>& B,
        vector<double>& C,
        int N
    ) {

        for (int i = 0; i < N; i++) {

            for (int j = 0; j < N; j++) {

                double soma = 0.0;

                for (int k = 0; k < N; k++) {

                    // Acessa A[i][k]
                    double a = A[i * N + k];

                    // Acessa B[k][j]
                    double b = B[k * N + j];

                    // Produto intermediário
                    double val = a * b;

                    // Operação computacionalmente custosa
                    soma += val * sin(val) + cos(val);
                }

                // Armazena o resultado em C[i][j]
                C[i * N + j] = soma;
            }
        }
    }

    int main() {

        const int N = 2048;

        vector<double> A(N * N);
        vector<double> B(N * N);
        vector<double> C(N * N, 0.0);

        // Inicialização dos dados
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {

                int idx = i * N + j;

                A[idx] = sin(i + j);
                B[idx] = cos(i - j);
            }
        }

        // Marca o início da medição
        auto t0 = chrono::high_resolution_clock::now();

        // Executa a multiplicação sequencial
        multiplicacaoCPU(A, B, C, N);

        // Marca o final da medição
        auto t1 = chrono::high_resolution_clock::now();

        // Calcula o tempo de execução
        double tempo =
            chrono::duration<double>(t1 - t0).count();

        cout << "N = " << N << endl;
        cout << "Tempo: " << tempo << " s" << endl;

        // Exibe alguns valores para validação
        cout << "Para validacao dos dados:" << endl;

        cout << "Primeiro elemento da matriz C[0][0] = "
            << C[0] << endl;
        
        cout << "Elemento do meio da matriz C[N/2][N/2] = "
        << C[(N / 2) * N + (N / 2)] << endl;

        cout << "Ultimo elemento da matriz C[N-1][N-1] = "
            << C[(N - 1) * N + (N - 1)] << endl;

        return 0;
    }
    ```




### **Soma distribuída com MPI (3 pontos)**

**Enunciado:**
O programa abaixo distribui um vetor de inteiros entre os processos MPI, cada processo calcula a soma local dos elementos que recebeu, e o processo `rank 0` reúne os resultados para calcular a soma total.

Atualmente, a implementação utiliza chamadas **coletivas** (`MPI_Scatter` e `MPI_Gather`).

Seu objetivo é **reescrever a comunicação** para usar **ponto a ponto não bloqueante**, tanto para a distribuição inicial dos dados quanto para a coleta das somas locais no processo `rank 0`.

### **Rubrica de Avaliação**

| Pontuação      | Critério               | Descrição                                                                                                                                                                                                                                                                  |
| -------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0 pontos**   | Não implementado       | O aluno não implementou a versão com comunicação ponto a ponto ou a solução não funciona.                                                                                                                                                                                  |
| **1,5 pontos** | Implementação parcial  | O aluno utilizou comunicação ponto a ponto, mas não corretamente. Exemplos: utilizou funções MPI bloqueantes ou utilizou funções não bloqueantes sem `MPI_Wait`, causando problemas de sincronização.                                                                            |
| **3,0 pontos** | Implementação completa | O aluno implementou corretamente a comunicação ponto a ponto não bloqueante, garantindo que o `rank 0` distribua os blocos de dados, cada processo calcule sua soma local, o `rank 0` receba os resultados por comunicações não bloqueantes e a soma total esteja correta. |


??? Note "Código base"
    ```cpp
    #include <mpi.h>
    #include <iostream>
    #include <vector>

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int N = 16;
        std::vector<int> dados;
        if (rank == 0) {
            dados.resize(N);
            for (int i = 0; i < N; i++) dados[i] = i+1;
        }

        int tam_local = N / size;
        std::vector<int> local(tam_local);

        // Distribuição dos dados
        MPI_Scatter(dados.data(), tam_local, MPI_INT,
                    local.data(), tam_local, MPI_INT,
                    0, MPI_COMM_WORLD);

        int soma_local = 0;
        for (int x : local) soma_local += x;

        std::vector<int> somas;
        if (rank == 0) somas.resize(size);

        // Coleta das somas locais
        MPI_Gather(&soma_local, 1, MPI_INT,
                somas.data(), 1, MPI_INT,
                0, MPI_COMM_WORLD);

        if (rank == 0) {
            int soma_total = 0;
            for (int s : somas) soma_total += s;
            std::cout << "Soma total = " << soma_total << "\n";
        }

        MPI_Finalize();
    }
    ```