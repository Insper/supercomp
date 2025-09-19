# Prova — Paralelismo com OpenMP

## Questão 1 — Teórica (Dependência de dados e paralelismo)
**Enunciado:**  
Explique por que um loop com **dependência entre iterações** não pode ser paralelizado ingenuamente com OpenMP.  
- Dê um exemplo de um loop **não paralelizável** por dependência.  
- Dê um exemplo de um loop **paralelizável** (independente).  
- Explique como a escolha do escalonamento (`static`, `dynamic`, `guided`) pode influenciar no desempenho **mesmo quando não há dependências**.  

??? note "Ver resposta"

    1. **Dependência de dados vs paralelização**  
    Um loop só pode ser paralelizado com OpenMP se cada iteração puder ser executada de forma independente.  
   
    **Exemplo não paralelizável:**  
            
        for (int i = 1; i < N; i++) {
            A[i] = A[i-1] + 1;  // dependência de dado em A[i-1]
        }
        
    **Exemplo paralelizável:**  

        for (int i = 0; i < N; i++) {
            B[i] = C[i] * 2;    // cada i usa apenas sua própria posição
        }
        

    2. **Influência do escalonamento no desempenho**  
    Mesmo sem dependências, a forma de dividir o trabalho entre threads afeta o desempenho:  
        - `static`: divide iterações em blocos fixos → bom quando o custo por iteração é uniforme.  
        - `dynamic`: atribui novas iterações às threads que terminam mais cedo → melhor para cargas desbalanceadas, mas tem overhead maior.  
        - `guided`: semelhante ao `dynamic`, mas blocos vão diminuindo de tamanho → equilíbrio entre overhead e adaptabilidade.  
    Além disso, podemos usar `schedule(static, tamanho_do_bloco)` para aplicar tiling: quebramos o loop em blocos que caibam na memória cache, melhorando o aproveitamento da hierarquia de memória.  
    Assim, a escolha do escalonamento pode **minimizar tempo de espera** entre threads, evitar subutilização da cache e melhorar o uso do cluster.  


---

## Questão 2 — Normalização de vetor com OpenMP
**Enunciado:**  
Implemente em C++ um programa que normaliza um vetor `a` em `b`, de forma que:  

`b[i] = a[i] / max(a)`


- Paralelize a busca de `max(a)` com **redução**.  
- Paralelize o cálculo de `b[i]` com `#pragma omp parallel for`.  
- Compare os tempos de execução com `OMP_SCHEDULE=static`, `dynamic` e `guided`.  

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::stoi(argv[1]) : 10000000);

    std::vector<float> a(N), b(N);

    // TODO: inicializar vetor a com números aleatórios em [0,1)

    float max_val = 0.0f;

    double t0 = omp_get_wtime();

    // TODO: paralelizar busca do máximo usando redução

    double t1 = omp_get_wtime();

    // TODO: paralelizar cálculo de b[i] = a[i] / max_val

    double t2 = omp_get_wtime();

    std::cout << "Tempo max = " << (t1 - t0) << "s\n";
    std::cout << "Tempo normalização = " << (t2 - t1) << "s\n";
}
```

??? note "Ver Resposta"

        #include <iostream>
        #include <vector>
        #include <random>
        #include <omp.h>

        int main(int argc, char** argv) {
            // Tamanho do vetor: pode ser passado na linha de comando
            int N = (argc > 1 ? std::stoi(argv[1]) : 10000000);

            // Vetores de entrada (a) e saída (b)
            std::vector<float> a(N), b(N);

            // Geração reprodutível de valores em a ∈ [0,1)
            std::mt19937 rng(123);
            std::uniform_real_distribution<> U(0.0, 1.0);
            for (int i = 0; i < N; i++) {
                a[i] = static_cast<float>(U(rng));
            }

            float max_val = 0.0f;

            double t0 = omp_get_wtime();

            // ------------------------------------------------------------------
            // PARTE 1: Encontrar o máximo com OpenMP + redução
            //
            // - Cada thread calcula um máximo local e o OpenMP combina (reduction)
            //   usando o operador 'max', evitando condições de corrida.
            // - 'parallel for' distribui as iterações i entre as threads.
            // ------------------------------------------------------------------
            #pragma omp parallel for reduction(max:max_val)
            for (int i = 0; i < N; i++) {
                if (a[i] > max_val) {
                    max_val = a[i];
                }
            }

            double t1 = omp_get_wtime();

            // ------------------------------------------------------------------
            // PARTE 2: Normalização em paralelo
            //
            // - Cada iteração é independente (não há dependência entre i's),
            //   então o 'parallel for' é naturalmente seguro e escalável.
            // - Acesso sequencial a a[i] e b[i] → boa localidade de cache.
            // ------------------------------------------------------------------
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                b[i] = a[i] / max_val;
            }

            double t2 = omp_get_wtime();

            std::cout << "Tempo max = " << (t1 - t0) << "s\n";
            std::cout << "Tempo normalizacao = " << (t2 - t1) << "s\n";
        }




---

## Questão 3 — Contagem de elementos pares
**Enunciado:**  
Implemente uma função que conta quantos elementos pares existem em um vetor de inteiros.  
- Paralelize com `#pragma omp parallel for reduction(+:contador)`.  
- Varie `OMP_NUM_THREADS` em {1, 2, 4, 8}.  
- Compare resultados com `schedule(static,4)` e `schedule(dynamic,4)`.  

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::stoi(argv[1]) : 10000000);

    std::vector<int> v(N);

    // TODO: inicializar vetor v com números inteiros aleatórios

    long long contador = 0;

    double t0 = omp_get_wtime();

    // TODO: paralelizar contagem de pares com redução

    double t1 = omp_get_wtime();

    std::cout << "Total pares = " << contador << "\n";
    std::cout << "Tempo = " << (t1 - t0) << "s\n";
}
```

??? note "Ver implementação"

        #include <iostream>
        #include <vector>
        #include <random>
        #include <omp.h>

        int main(int argc, char** argv) {
            // Tamanho do vetor (padrão: 10 milhões). Pode ser passado via linha de comando.
            int N = (argc > 1 ? std::stoi(argv[1]) : 10000000);

            // Vetor de inteiros a ser analisado
            std::vector<int> v(N);

            // Geração reprodutível de dados inteiros uniformes em [0, 1000]
            std::mt19937 rng(123);
            std::uniform_int_distribution<int> U(0, 1000);
            for (int i = 0; i < N; i++) {
                v[i] = U(rng);
            }

            // Contador global de elementos pares (tipo largo para evitar overflow)
            long long contador = 0;

            double t0 = omp_get_wtime();

            // ------------------------------------------------------------------
            // Contagem de pares paralelizada
            //
            // - 'parallel for' distribui as iterações entre as threads.
            // - 'reduction(+:contador)' cria um contador local por thread e
            //    no final soma tudo no 'contador' global.
            // - Cada iteração é independente (não há dependências entre i's).
            // ------------------------------------------------------------------
            #pragma omp parallel for reduction(+:contador)
            for (int i = 0; i < N; i++) {
                if (v[i] % 2 == 0) {
                    contador++;  // soma no acumulador local da thread
                }
            }

            double t1 = omp_get_wtime();

            std::cout << "Total pares = " << contador << "\n";
            std::cout << "Tempo = " << (t1 - t0) << "s\n";
        }




---

## Questão 4 — Convolução 1D com OpenMP
**Enunciado:**  
Implemente uma convolução 1D de um vetor `a` (N elementos) com um kernel fixo de tamanho `K`.  

$$ c[i] = \sum_{j=0}^{K-1} a[i+j] \cdot kernel[j] $$


- Paralelize o loop externo (`i`).  
- Varie `OMP_NUM_THREADS` em {2, 4, 8, 16}.  
- Qual o menor custo de hardware para o melhor beneficio de desempenho?


```cpp
#include <iostream>
#include <vector>
#include <omp.h>

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::stoi(argv[1]) : 1000000);
    int K = 5; // tamanho do kernel

    std::vector<float> a(N, 1.0f), kernel(K, 0.2f), c(N-K+1, 0.0f);

    double t0 = omp_get_wtime();

    // TODO: paralelizar o loop externo da convolução 1D
    for (int i = 0; i < N - K + 1; i++) {
        float soma = 0.0f;
        for (int j = 0; j < K; j++) {
            soma += a[i + j] * kernel[j];
        }
        c[i] = soma;
    }

    double t1 = omp_get_wtime();

    std::cout << "Tempo convolução = " << (t1 - t0) << "s\n";
}
```
??? note "Ver Implementação"

        #include <iostream>
        #include <vector>
        #include <omp.h>

        int main(int argc, char** argv) {
            // Tamanho do vetor de entrada (default: 1 milhão). Pode ser passado na linha de comando.
            int N = (argc > 1 ? std::stoi(argv[1]) : 1000000);
            int K = 5; // tamanho do kernel (filtro convolucional)

            // Vetor de entrada 'a' preenchido com 1.0
            std::vector<float> a(N, 1.0f);
            // Kernel de tamanho K preenchido com 0.2 (simples média móvel de 5 pontos)
            std::vector<float> kernel(K, 0.2f);
            // Vetor de saída 'c' com tamanho (N-K+1), inicializado em 0.0
            std::vector<float> c(N-K+1, 0.0f);

            double t0 = omp_get_wtime();

            // ------------------------------------------------------------------
            // Convolução paralelizada
            //
            // - 'parallel for' distribui as iterações de i (posições da saída) entre threads.
            // - Cada iteração calcula c[i] = soma(a[i..i+K-1] * kernel[0..K-1]).
            // - As iterações são independentes → não há dependência entre diferentes c[i].
            // - Isso torna o loop um bom candidato para paralelismo com OpenMP.
            // ------------------------------------------------------------------
            #pragma omp parallel for
            for (int i = 0; i < N - K + 1; i++) {
                float soma = 0.0f;
                // loop interno: acumula o produto de K elementos
                for (int j = 0; j < K; j++) {
                    soma += a[i + j] * kernel[j];
                }
                c[i] = soma; // resultado da convolução no ponto i
            }

            double t1 = omp_get_wtime();

            std::cout << "Tempo convolução = " << (t1 - t0) << "s\n";
        }
