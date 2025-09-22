# Prova — Programação Distribuída com MPI

## Questão 1 — Teórica (Conceitos fundamentais de MPI)
**Enunciado:**  
a) Diferencie **paralelismo com memória compartilhada** e **paralelismo com memória distribuída**.  
b) Explique o que é um **rank** no contexto do MPI e como ele é utilizado em comunicações.  
c) Cite uma vantagem e uma desvantagem de usar **comunicação coletiva** (ex.: `MPI_Bcast`, `MPI_Reduce`) em comparação com várias chamadas ponto a ponto (`MPI_Send`/`MPI_Recv`).  
d) O que acontece se dois processos MPI tentarem executar `MPI_Send` um para o outro sem `MPI_Recv` correspondente?  

??? note "Ver Resposta"

    a) No paralelismo com memória compartilhada (como no OpenMP), todas as threads acessam a mesma região de memória. Isso facilita a troca de dados. Já no paralelismo com memória distribuída (como no MPI), a comunicação é feita por troca explícita de mensagens. 


    b) Rank em MPI
    O rank é o identificador único de cada processo dentro de um comunicador MPI, geralmente numerado de `0` até `size-1`. Ele é usado para diferenciar processos e definir o papel de cada um durante a execução. 


    c) Comunicação coletiva vs. ponto a ponto
    Uma vantagem das comunicações coletivas é a simplicidade de implementação, pois a biblioteca MPI já otimiza a troca de dados internamente, evitando que o programador tenha de escrever múltiplos `MPI_Send` e `MPI_Recv`. A desvantagem é o overhead, todos os processos precisam participar da operação coletiva ao mesmo tempo, o que pode causar esperas desnecessárias se algum processo atrasar. Já a comunicação ponto a ponto dá mais flexibilidade, e controle dos processos e trocas de mensagens.


    d) Dois processos executando apenas `MPI_Send`
    Se dois processos chamarem `MPI_Send` simultaneamente um para o outro sem que haja chamadas correspondentes de `MPI_Recv`, o programa pode entrar em deadlock. Isso acontece porque cada processo fica esperando o envio ser concluído, mas não há recepção ativa para liberar os dados. Para evitar isso, é importante garantir que um processo envie enquanto o outro recebe.



## Questão 2 — Broadcast manual vs MPI_Bcast
**Enunciado:**  
Implemente duas versões para distribuir um mesmo valor inteiro a todos os processos:  

1. **Versão manual:** o `rank 0` envia o valor individualmente para cada outro processo usando `MPI_Send`.  
2. **Versão coletiva:** use `MPI_Bcast` para o mesmo objetivo.  

Compare os tempos de execução à medida que o número de processos aumenta.  

```cpp
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int valor;
    if (rank == 0) {
        valor = 42;
        // TODO: enviar valor manualmente com MPI_Send
    } else {
        // TODO: receber valor com MPI_Recv
    }

    std::cout << "[Manual] Processo " << rank << " recebeu valor = " << valor << "\n";

    // TODO: implementar versão com MPI_Bcast

    std::cout << "[Bcast] Processo " << rank << " recebeu valor = " << valor << "\n";

    MPI_Finalize();
}
```
??? note "Ver Resposta"

        #include <mpi.h>
        #include <iostream>

        int main(int argc, char** argv) {
            MPI_Init(&argc, &argv);
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            int valor;
            if (rank == 0) {
                valor = 42;
                // versão manual
                for (int dest = 1; dest < size; dest++) {
                    MPI_Send(&valor, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                }
            } else {
                MPI_Recv(&valor, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            std::cout << "[Manual] Processo " << rank << " recebeu valor = " << valor << "\n";

            // versão com MPI_Bcast
            if (rank == 0) valor = 99;
            MPI_Bcast(&valor, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::cout << "[Bcast] Processo " << rank << " recebeu valor = " << valor << "\n";

            MPI_Finalize();
        }


## Questão 3 — Redução manual vs MPI_Reduce
**Enunciado:**  
Implemente a soma de inteiros distribuídos entre os processos de duas formas:  

1. **Versão manual:** cada processo envia seu valor ao `rank 0`, que acumula.  
2. **Versão coletiva:** use `MPI_Reduce` para calcular a soma diretamente.  

Verifique se os resultados são iguais.  

```cpp
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int meu_valor = rank + 1; // cada processo tem um valor diferente
    int soma = 0;

    if (rank == 0) {
        soma = meu_valor;
        // TODO: receber valores dos outros processos com MPI_Recv e acumular
    } else {
        // TODO: enviar valor ao processo 0 com MPI_Send
    }

    if (rank == 0) {
        std::cout << "[Manual] Soma total = " << soma << "\n";
    }

    // TODO: implementar versão com MPI_Reduce

    if (rank == 0) {
        std::cout << "[Reduce] Soma total = " << soma << "\n";
    }

    MPI_Finalize();
}
```
??? note "Ver Resposta"

        #include <mpi.h>
        #include <iostream>

        int main(int argc, char** argv) {
            MPI_Init(&argc, &argv);
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            int meu_valor = rank + 1; // só para ter valores diferentes
            int soma = 0;

            if (rank == 0) {
                soma = meu_valor;
                for (int src = 1; src < size; src++) {
                    int temp;
                    MPI_Recv(&temp, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    soma += temp;
                }
                std::cout << "[Manual] Soma total = " << soma << "\n";
            } else {
                MPI_Send(&meu_valor, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }

            // versão com MPI_Reduce
            int soma_reduce = 0;
            MPI_Reduce(&meu_valor, &soma_reduce, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                std::cout << "[Reduce] Soma total = " << soma_reduce << "\n";
            }

            MPI_Finalize();
        }


## Questão 4 — Scatter + cálculo local + Gather
**Enunciado:**  
Implemente um programa que divide um vetor de inteiros igualmente entre os processos, cada processo calcula a soma dos seus elementos, e depois os resultados são reunidos no processo 0:  

1. Use `MPI_Scatter` para distribuir partes do vetor.  
2. Cada processo calcula sua soma parcial.  
3. Use `MPI_Gather` para enviar os resultados ao `rank 0`.  
4. O `rank 0` calcula a soma total e imprime.  

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

    // TODO: usar MPI_Scatter para enviar partes do vetor

    int soma_local = 0;
    // TODO: calcular soma local

    std::vector<int> somas;
    if (rank == 0) somas.resize(size);

    // TODO: usar MPI_Gather para enviar resultados locais ao rank 0

    if (rank == 0) {
        int soma_total = 0;
        // TODO: acumular somas e imprimir resultado final
    }

    MPI_Finalize();
}
```

??? note "Ver Resposta"

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

            MPI_Scatter(dados.data(), tam_local, MPI_INT,
                        local.data(), tam_local, MPI_INT,
                        0, MPI_COMM_WORLD);

            int soma_local = 0;
            for (int x : local) soma_local += x;

            std::vector<int> somas;
            if (rank == 0) somas.resize(size);

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
