Message Passing Interface (MPI) é um padrão para comunicação de dados em computação paralela. Existem várias modalidades de computação paralela, e dependendo do problema que se está tentando resolver, pode ser necessário passar informações entre os vários processadores ou nós de um cluster, e o MPI oferece uma infraestrutura para essa tarefa.

Para iniciarmos o nosso estudo de MPI, implemente os desafios abaixo, entendendo como encadear sends e receives, e o impacto nos resultados.


## **Ping-pong**

A ideia é medir a **latência de comunicação ponto a ponto**.

Implemente o ping-pong: rank 0 envia uma mensagem ao rank 1, que responde.
Faça duas versões:

   * **Bloqueante** (`MPI_Send/MPI_Recv`)
   * **Não-bloqueante** (`MPI_Isend/Irecv + MPI_Wait`)

Rode os testes:

   * Mensagens de **16 B, 1 KB, 64 KB, 1 MB**
   * Em 2, 3 e 4 nós 

Analise: 

   * Para mensagens pequenas, o que domina: **latência fixa** ou **tamanho da mensagem**?
   * A partir de que tamanho de mensagem o gargalo passa a ser a **largura de banda** da rede?

Código base:
```cpp
#include <mpi.h>        // Biblioteca principal do MPI para comunicação entre processos
#include <iostream>    
#include <cstring>      

int main(int argc, char** argv) {
    int rank;               // Variável que armazenará o "rank" (identificador) do processo
    MPI_Status status;      // Estrutura que armazenará o status da comunicação MPI
    char mensagem[100];     // Vetor de caracteres para armazenar a mensagem a ser enviada/recebida

    // Inicializa o ambiente MPI (todos os processos são iniciados)
    MPI_Init(&argc, &argv);

    // Descobre o "rank" do processo atual dentro do comunicador global (MPI_COMM_WORLD)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Se este for o processo de rank 0 (emissor inicial)
    if (rank == 0) {
        // Copia a string "Olá" para a variável mensagem
        std::strcpy(mensagem, "Olá");

        // Envia a mensagem para o processo de rank 1
        // Parâmetros: buffer, tamanho, tipo, destino, tag, comunicador
        MPI_Send(mensagem, std::strlen(mensagem) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        // Imprime no terminal que a mensagem foi enviada
        std::cout << "Processo 0 enviou: " << mensagem << std::endl;

        // Aguarda a resposta do processo 1
        // Parâmetros: buffer, tamanho máximo, tipo, origem, tag, comunicador, status
        MPI_Recv(mensagem, 100, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &status);

        // Imprime a mensagem recebida
        std::cout << "Processo 0 recebeu: " << mensagem << std::endl;
    }

    // Se este for o processo de rank 1 (receptor inicial)
    else if (rank == 1) {
        // Recebe a mensagem enviada pelo processo 0
        MPI_Recv(mensagem, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

        // Imprime a mensagem recebida
        std::cout << "Processo 1 recebeu: " << mensagem << std::endl;

        // Prepara a resposta "Oi"
        std::strcpy(mensagem, "Oi");

        // Envia a resposta de volta ao processo 0
        MPI_Send(mensagem, std::strlen(mensagem) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

        // Imprime que a mensagem foi enviada
        std::cout << "Processo 1 enviou: " << mensagem << std::endl;
    }

    else {
        // Todos os outros processos apenas informam que estão ociosos
        std::cout << "Processo " << rank << " está ocioso neste exercício." << std::endl;
    }

    // Finaliza o ambiente MPI (todos os processos encerram)
    MPI_Finalize();

    return 0;
}
```

### Compile o programa:
```bash
mpic++ -FlagdeOtimização seu_codigo.cpp -o seu_binario
```


### Script SLURM

```bash
#!/bin/bash
#SBATCH --job-name=mpi_hello
#SBATCH --output=saida_%j.txt
#SBATCH --nodes=2   # 2 nós (2 computadores)
#SBATCH --ntasks=5  # 5 processos (5 task MPI)
#SBATCH --cpus-per-task=1 # 1 thread
#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH –mem=2G

mpirun -np $SLURM_NTASKS ./seu_binario

```

### Submeta o job com SLURM:
```bash
sbatch SeuSlurm.slurm
```


## **Token em anel**

A ideia é perceber o custo de **coletar informações sequencialmente**.

Implemente o token em anel: cada rank adiciona uma informação nova ao vetor e passa adiante.
Execute em 2, 3 e 4 nós .
Compare com o mesmo problema usando `MPI_Gather`.

Analise:

   * Como cresce o tempo do anel com o número de processos?
   * Qual o gargalo de percorrer todos em sequência?
   * Qual a vantagem de usar `MPI_Gather`?

---


## **Alternância**

A ideia é entender como funciona uma **distribuição de tarefas**.

Rank 0 possui 13 tarefas cada uma com diferentes niveis de complexidade. Implemente uma distribuição dinâmica de tarefas, o worker que terminar primeiro recebe a próxima tarefa.

O que acontece quando as tarefas variam muito de custo?




# O que você deve entregar:

Para cada exercício:

1. **Código** (disponível no repositório do github).
2. **Tabela de resultados** (parâmetros usados, tempos medidos).
3. **Discussão**: Análise dos resultados

**Envie o seu relatório com as suas análises até as 23h59 de 04/09 pelo [GitHub Classroom](https://classroom.github.com/a/ilH4AsH_)** 
