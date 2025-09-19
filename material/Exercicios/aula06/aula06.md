# Prova — Efeitos Colaterais do Paralelismo em OpenMP

## Questão 1 — Teórica (Race condition e dependência de dados)
**Enunciado:**  
a) Explique com suas palavras o que é uma **condição de corrida** em um programa paralelo.  
b) Por que um loop que acumula resultados em uma mesma variável global é vulnerável a esse problema?  
c) Dê um exemplo de loop que **não** pode ser paralelizado por depender de valores calculados em iterações anteriores.  
d) O que significa “reformular o algoritmo” para eliminar uma dependência? Cite um exemplo.  

??? note "Ver respostas"

    a) É quando duas ou mais threads acessam a mesma variável compartilhada simultaneamente, e pelo menos uma delas altera o valor. 

    b) Um loop em paralelo que escreve em uma única variável global cria acessos concorrentes de várias threads. Diferentes threads podem sobrescrever valores, perdendo operações.

    c) Loops em que cada iteração depende do valor produzido pela iteração anterior.  
    Ex.: cálculo de séries recursivas, onde `A[i]` usa `A[i-1]`.  
    
    d) É alterar a forma de cálculo para remover dependências entre iterações e permitir paralelismo.  

## Questão 2 — Média de vetor com e sem redução (OpenMP)
**Enunciado:**  
Implemente em C++ duas versões de um programa que calcula a **média** dos valores em um vetor:  
1. **Versão ingênua (errada):** paralelize o loop de soma sem usar redução.  
2. **Versão corrigida:** use `reduction(+:soma)` para evitar condição de corrida.  
Encontre o melhor custo de hardware para o melhor benefício de tempo   

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

int main() {
    int N = 1000000;
    std::vector<float> v(N);

    std::mt19937 rng(123);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (int i = 0; i < N; i++) v[i] = dist(rng);

    double soma = 0.0;

    // TODO: versão ingênua paralelizada sem redução (vai dar erro numérico)

    // TODO: versão corrigida usando reduction
    // #pragma omp parallel for reduction(+:soma)

    double media = soma / N;
    std::cout << "Media = " << media << "\n";
}
```

??? note "Ver implementação"

        #include <iostream>
        #include <vector>
        #include <random>
        #include <omp.h>

        int main() {
            int N = 1000000;
            std::vector<float> v(N);

            std::mt19937 rng(123);
            std::uniform_real_distribution<> dist(0.0, 1.0);
            for (int i = 0; i < N; i++) v[i] = dist(rng);

            double soma = 0.0;

            // Versão errada (sem reduction, resultado inconsistente)
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                soma += v[i];
            }
            std::cout << "[Errada] Media = " << soma / N << "\n";

            soma = 0.0;
            // Versão correta com reduction
            #pragma omp parallel for reduction(+:soma)
            for (int i = 0; i < N; i++) {
                soma += v[i];
            }
            std::cout << "[Correta] Media = " << soma / N << "\n";
        }


---

## Questão 3 — Prefixo acumulado (dependência de dados)
**Enunciado:**  
O programa abaixo calcula o prefixo acumulado (`p[i] = p[i-1] + a[i]`), que possui dependência sequencial.  

a) Explique por que a versão abaixo **não pode** ser paralelizada diretamente.  
b) Reescreva o algoritmo de forma **paralelizável**, eliminando a dependência.  

```cpp
#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    int N = 20;
    std::vector<int> a(N), p(N);

    for (int i = 0; i < N; i++) a[i] = 1;

    // Versão com dependência
    p[0] = a[0];
    for (int i = 1; i < N; i++) {
        p[i] = p[i-1] + a[i];  // depende do anterior
    }

    // TODO: versão reformulada paralelizável
    // dica: perceba que o resultado é uma progressão

    for (int i = 0; i < N; i++) std::cout << p[i] << " ";
}
```

??? note "Ver resposta"
    a) O loop tem dependência entre iterações, para calcular p[i] é obrigatório conhecer p[i-1] já calculado.
    Se você colocar #pragma omp parallel for nesse loop, threads diferentes tentariam computar p[i] e p[i-1] ao mesmo tempo, violando a ordem necessária; o resultado fica incorreto

    b) Como o programa define a[i] = 1 para todo i, então o prefixo acumulado vira uma progressão:

    p[i] = 1 + 1 + … + 1 (i+1 vezes) = i + 1.

    Logo podemos paralelizar calculando diretamente a fórmula fechada:

        #include <iostream>
        #include <vector>
        #include <omp.h>

        int main() {
            int N = 20;
            std::vector<int> a(N), p(N);

            for (int i = 0; i < N; i++) a[i] = 1;

            // Versão reformulada (progressão) — VÁLIDA porque a[i] = 1 para todo i
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                // Como a[i]=1, p[i] = (i+1)*1
                p[i] = (i + 1) * a[0];  // eliminada a dependência (fórmula fechada)
            }

            for (int i = 0; i < N; i++) std::cout << p[i] << " ";
            std::cout << "\n";
        }


---

## Questão 4 — Recursão paralela em árvore binária (tasks)
**Enunciado:**  
Considere uma **árvore binária** em que cada nó contém um número inteiro.  
Desejamos calcular a **soma de todos os nós** da árvore.  

Matematicamente, se `raiz` é o nó atual:  

`S(raiz) = valor(raiz) + S(filho_esq) + S(filho_dir)`

**Tarefa:**  
- Implemente uma função recursiva `soma_arvore` que use **OpenMP tasks** para calcular a soma:  
  - Uma task para o filho esquerdo  
  - Uma task para o filho direito  
  - Use `taskwait` para sincronizar.  

```cpp
#include <iostream>
#include <omp.h>

struct No {
    int valor;
    No* esq;
    No* dir;
    No(int v) : valor(v), esq(nullptr), dir(nullptr) {}
};

// TODO: implementar soma_arvore recursiva com tasks
int soma_arvore(No* raiz) {
    if (!raiz) return 0;
    int soma_esq = 0, soma_dir = 0;

    // #pragma omp task shared(soma_esq)
    // soma_esq = soma_arvore(raiz->esq);

    // #pragma omp task shared(soma_dir)
    // soma_dir = soma_arvore(raiz->dir);

    // #pragma omp taskwait

    return raiz->valor + soma_esq + soma_dir;
}

int main() {
    // exemplo: árvore pequena
    No* raiz = new No(1);
    raiz->esq = new No(2);
    raiz->dir = new No(3);
    raiz->esq->esq = new No(4);
    raiz->esq->dir = new No(5);

    double t0 = omp_get_wtime();
    int soma = soma_arvore(raiz);
    double t1 = omp_get_wtime();

    std::cout << "Soma dos nós = " << soma << " tempo = " << (t1 - t0) << "s\n";
}
```

??? note "Ver Implementação"

        #include <iostream>
        #include <omp.h>

        // Nó básico de árvore binária
        struct No {
            int valor;
            No* esq;
            No* dir;
            No(int v) : valor(v), esq(nullptr), dir(nullptr) {}
        };

        // -------------------------------------------------------------
        // soma_arvore: soma os valores de todos os nós de uma árvore.
        // Estratégia paralela: criar TAREFAS (OpenMP tasks) para cada
        // subárvore esquerda e direita.
        // -------------------------------------------------------------
        int soma_arvore(No* raiz) {
            // Caso base da recursão: árvore vazia soma 0 (execução sequencial)
            if (!raiz) return 0;

            int soma_esq = 0, soma_dir = 0;

            // Cria uma TAREFA para computar a soma da subárvore ESQUERDA.
            // 'shared(soma_esq)' permite que a tarefa escreva no escopo atual.
            // Observação: cada tarefa escreve em uma variável distinta → sem corrida.
            #pragma omp task shared(soma_esq)
            soma_esq = soma_arvore(raiz->esq);

            // Cria outra TAREFA para a subárvore DIREITA.
            #pragma omp task shared(soma_dir)
            soma_dir = soma_arvore(raiz->dir);

            // Sincronização: espera as DUAS tarefas terminarem antes de somar.
            // Sem taskwait, poderíamos retornar antes das subtarefas concluírem.
            #pragma omp taskwait
            return raiz->valor + soma_esq + soma_dir;
        }

        int main() {
            // Monta uma arvorezinha de exemplo
            No* raiz = new No(1);
            raiz->esq = new No(2);
            raiz->dir = new No(3);
            raiz->esq->esq = new No(4);
            raiz->esq->dir = new No(5);

            double t0 = omp_get_wtime();
            int soma = 0;

            // Região paralela: cria o time de threads.
            #pragma omp parallel
            {
                // single: apenas UMA thread entra na chamada inicial.
                // A partir daqui, as tarefas criadas recursivamente poderão
                // ser executadas por QUALQUER thread do time.
                #pragma omp single
                soma = soma_arvore(raiz);
            }
            double t1 = omp_get_wtime();

            std::cout << "Soma dos nós = " << soma << " tempo = " << (t1 - t0) << "s\n";
        }


**Perguntas adicionais:**  
a) O que acontece com o tempo de execução se a árvore for muito pequena?  
b) Por que criar muitas tasks pequenas pode ser pior do que calcular de forma sequencial?  

??? note "Ver Respostas"

    a)O tempo tende a piorar ou empatar com o sequencial, porque o overhead de paralelismo (criar região paralela, criar/agendar/sincronizar tasks, work-stealing, barreiras) pode ser maior que o trabalho útil (somar poucos nós). Em árvores pequenas, não há trabalho suficiente para “compensar” o custo de paralelizar.

    b) Cada `#pragma omp task` tem custo de criação, enfileiramento e agendamento. Se o trabalho é minúsculo, você gasta mais tempo organizando do que trabalhando.

