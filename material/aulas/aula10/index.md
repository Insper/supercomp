# Programação distribuída com MPI

Na programação com memória compartilhada, threads de um mesmo processo podem acessar as mesmas variáveis. OpenMP é uma ferramenta comum para esse modelo.

Com MPI, cada processo tem seu próprio espaço de memória. Uma variável criada no rank 0 não se torna automaticamente disponível no rank 1: para compartilhar seu conteúdo, o programa precisa usar uma operação de comunicação.

MPI é um padrão; Open MPI e MPICH são implementações. A implementação escolhe os mecanismos disponíveis para trocar mensagens, como memória compartilhada dentro do nó e rede entre nós. 

### Primeiro programa: quem sou eu e onde estou?

Todos os processos executam o mesmo binário. O rank permite que cada processo escolha uma ação diferente. Neste primeiro exemplo, o rank 0 apresenta a tarefa e os demais calculam o quadrado do próprio ID.


```cpp
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Inicializa o MPI.
    MPI_Init(&argc, &argv);

    char nome[MPI_MAX_PROCESSOR_NAME];
    int tamanho, rank, total;
    // Pega informações relevantes do ecossistema MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Get_processor_name(nome, &tamanho);

    std::cout << "Rank " << rank << " | Nó: " << nome << std::endl;

    if (rank == 0) {
        // O rank 0 apresenta a tarefa.
        std::cout << "Tarefa: calcular o quadrado de cada ID.\n"
                  << "Temos " << total << " nós.\n";
    } else {
        // Cada nó calcula usando seu próprio ID.
        int resultado = rank * rank;

        std::cout << ": Minha computação é = " << resultado << std::endl;
    }

    // Finaliza o MPI.
    MPI_Finalize();
    return 0;
}
```

### Entendendo as chamadas

Um cluster é formado por vários computadores conectados, chamados **nós**. Cada nó, como o `compute10`, possui memória RAM e uma ou mais CPUs. Essas CPUs têm **núcleos**, também chamados de *cores*, que executam as instruções dos programas. Assim, um único nó pode executar vários processos usando seus diferentes núcleos.

Um **processo** é uma instância de um programa em execução, com seu próprio espaço de memória. Quando iniciamos um programa MPI com quatro processos, executamos quatro instâncias do mesmo binário. Cada uma possui suas próprias variáveis: alterar uma variável em um processo não altera automaticamente a variável correspondente nos demais, mesmo que eles estejam no mesmo nó.

Dentro de um processo podem existir **threads**, que são fluxos de execução que compartilham a memória desse processo. Essa é uma diferença importante: threads de um mesmo processo podem acessar as mesmas variáveis; processos MPI precisam usar operações de comunicação para trocar dados. Também é possível combinar os dois modelos, usando MPI entre processos e OpenMP para criar threads dentro de cada processo.

Para identificar os processos, o MPI atribui a cada um um **rank**, válido dentro de um **comunicador**. O comunicador define um grupo de processos e o contexto em que eles podem se comunicar. Nos exemplos da aula, usamos `MPI_COMM_WORLD`, que reúne os processos da execução MPI. Se esse grupo tiver quatro processos, seus ranks serão **0, 1, 2 e 3**.

**Rank e nó são informações diferentes.** O rank identifica o processo; o nome do nó identifica o computador onde ele executa. Por exemplo, os ranks 0 e 1 podem executar no `compute10`, enquanto os ranks 2 e 3 executam no `compute11`. Nesse caso, temos quatro processos MPI distribuídos entre dois nós.

Para organizar essa execução, cada processo começa chamando **`MPI_Init`**, que inicializa o ambiente MPI. Em seguida, **`MPI_Comm_rank`** informa o rank daquele processo, e **`MPI_Comm_size`** informa quantos processos fazem parte do comunicador. Todos recebem o mesmo total, mas cada processo recebe seu próprio rank.

Com essas informações, o programa pode atribuir comportamentos diferentes aos processos. Uma condição como `if (rank == 0)` permite que apenas o rank 0 apresente a tarefa, enquanto os demais realizam cálculos. O rank 0 não é automaticamente um coordenador: esse papel é definido pelo código.

A chamada **`MPI_Get_processor_name`** permite obter o nome do nó onde cada processo executa. Ao imprimir o rank junto com esse nome, conseguimos verificar como os processos foram distribuídos pelo cluster. Já **`MPI_Finalize`**, chamada por todos os processos ao final do programa, encerra o uso do ambiente MPI.

Portanto, ao ler a saída do programa, observe duas informações: **quantos ranks diferentes aparecem e quantos nomes de nós diferentes aparecem**. Elas mostram, respectivamente, quantos processos participaram da execução e em quantos computadores esses processos foram distribuídos.

Resumindo...

| Conceito | Significado | Exemplo |
| --- | --- | --- |
| Nó | Um computador do cluster | `compute10` |
| Núcleo (core) | Unidade de processamento da CPU | Um nó pode ter vários núcleos (cores) |
| Processo | Uma instância do programa com seu próprio espaço de memória | Uma cópia do binário |
| Thread | Um fluxo de execução dentro de um processo | Threads de um programa OpenMP |
| Rank | Identificador do processo dentro de um comunicador MPI | 0, 1, 2 e 3 |
| Comunicador | Grupo de processos e contexto de comunicação | `MPI_COMM_WORLD` |
| Inicialização (`MPI_Init`) | Inicializa o ambiente MPI em cada processo | `MPI_Init(&argc, &argv);` |
| Identificação do processo (`MPI_Comm_rank`) | Obtém o rank deste processo no comunicador | `MPI_Comm_rank(MPI_COMM_WORLD, &rank);` → `rank = 0` |
| Total de processos (`MPI_Comm_size`) | Obtém a quantidade de processos no comunicador | `MPI_Comm_size(MPI_COMM_WORLD, &total);` → `total = 4` |
| Identificação do nó (`MPI_Get_processor_name`) | Obtém o nome do nó onde o processo executa | `MPI_Get_processor_name(nome, &tamanho);` → `nome = "compute10"` |
| Finalização (`MPI_Finalize`) | Finaliza o uso do ambiente MPI em cada processo | `MPI_Finalize();` |



### Compilação

No head-node, dentro da pasta `SCRATCH`:

```bash
mpic++ -O2 hello.cpp -o hello
```

`mpic++` é um wrapper: chama o compilador C++ com as opções necessárias para incluir e vincular o MPI. Sempre recompile após alterar o código.


### Executando pelo Slurm

O Slurm reserva recursos e inicia tarefas nos nós de computação. `sbatch` submete um script; `srun` inicia as tarefas. Reservar quatro tarefas não significa que executar `./hello_mpi` sozinho criará quatro processos.

```bash
srun --partition=merry_cpu --mpi=pmix --mem=1G --nodes=2 \
     --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 ./hello
```


| Opção | Significado |
| --- | --- |
| `--partition=merry_cpu` | Partição/fila em que o job será executado |
| `--nodes=2` | Quantidade de nós solicitados |
| `--ntasks=2` | Total de tarefas; aqui, processos MPI |
| `--ntasks-per-node=1` | Distribuição de tarefas por nó |
| `--cpus-per-task=1` | Uma CPU Slurm por tarefa; não cria várias threads |
| `--mem=1G` | 1 GiB de memória solicitada por nó, não por processo |
| `--time=00:01:00` | Tempo máximo do job |
| `--mpi=pmix` | Integração usada pelo `srun` para iniciar o MPI |



Saída ilustrativa:

```text
Rank 1 | Nó: compute11
Minha computação é = 1
Rank 0 | Nó: compute10
Tarefa: calcular o quadrado de cada ID.
Temos 2 nós.
```

Os nós escolhidos e a ordem das mensagens podem variar, o rank não determina a ordem de execução das aplicações.

### Execução em lote: script `run.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=hello-mpi
#SBATCH --partition=merry_cpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --output=mpi.out

echo "Job: $SLURM_JOB_ID"
echo "Nós alocados: $SLURM_JOB_NODELIST"
echo "Tarefas: $SLURM_NTASKS"

# Usa os recursos solicitados acima para iniciar os processos MPI.
srun --mpi=pmix ./hello
```

Submeta o job com o comando:

```bash
sbatch run.slurm
```

Se abrir o arquivo de saída, deve visualizar algo como:

```bash
Job: 29791
Nós alocados: compute[10-11]
Tarefas: 2
Rank 0 | Nó: compute10
Rank 1 | Nó: compute11
: Minha computação é = 1
Tarefa: calcular o quadrado de cada ID.
Temos 2 nós.
```
### Para entender se você entendeu: processos não são nós

Nesta missão, você vai observar como o Slurm distribui os processos MPI entre os nós do cluster.

Para cada experimento da tabela:

1. **Tente imaginar:** quantos processos serão executados? Quantos nós diferentes vão aparecer no print da saída?
2. **Ajuste o script:** altere `--nodes`, `--ntasks` e `--ntasks-per-node` para os valores indicados na tabela.
3. **Execute e confira:** compare os ranks e os nomes dos nós impressos pelo programa com o que você imaginou que aconteceria.

Lembre-se: 

**`--nodes` indica quantos nós de computação serão usados; 

*`--ntasks` indica o total de processos MPI; 

e `--ntasks-per-node` indica quantos processos executarão em cada nó de computação.**

| Experimento | `--nodes` | `--ntasks` | `--ntasks-per-node` | Distribuição esperada |
| --- | ---: | ---: | ---: | --- |
| A | 1 | 4 | 4 | ????????????????????????????????????? |
| B | 2 | 2 | 1 | ????????????????????????????????????? |
| C | 2 | 4 | 2 | ????????????????????????????????????? |
| D | 3 | 6 | 2 | ????????????????????????????????????? |

No experimento A, por exemplo, você deverá encontrar **quatro ranks diferentes**, mas **apenas um nó de computação**, pois todos os processos executarão no mesmo computador.

Perguntas para discutir:

1. Quatro processos sempre exigem quatro computadores?
2. O que aconteceria se a última linha do script fosse apenas `./hello_mpi`?
3. Por que a ordem dos prints muda entre execuções?

**Esta atividade não precisa ser entregue.** Use as execuções e as perguntas para verificar seu entendimento e discutir os resultados em aula.
