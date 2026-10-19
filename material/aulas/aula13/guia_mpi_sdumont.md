# Uso de MPI no SDumont com Slurm

Este roteiro mostra como preparar o ambiente, compilar um programa MPI, solicitar recursos e conferir onde cada processo executou no SDumont.

Os exemplos foram adaptados dos comandos e da consulta à partição fornecidos para a aula.


### Entendendo os papéis de MPI e Slurm

O **MPI** permite que processos troquem mensagens. O **Slurm** recebe o pedido de recursos, coloca o job na fila e organiza sua execução nos nós de computação.

| Conceito | Significado | Exemplo |
| --- | --- | --- |
| Nó | Um computador do cluster | Um nó escolhido pelo Slurm |
| Processo MPI | Uma instância do programa com memória própria | Uma execução de `hello` |
| Rank | Identificador do processo no comunicador | 0 e 1 em uma execução com dois processos |
| Tarefa Slurm | Unidade de execução solicitada ao escalonador | Neste roteiro, uma tarefa para cada processo MPI |
| CPU por tarefa | Recurso de CPU solicitado para cada tarefa | `--cpus-per-task=1` |
| Partição | Conjunto de recursos e regras de execução | `sequana_cpu_dev` |
| Job | Pedido de recursos submetido ao Slurm | Um job identificado por um número |

Dois processos podem executar no mesmo nó. Portanto, pedir dois processos não significa automaticamente pedir dois computadores: a distribuição depende também das opções de nós e tarefas por nó.


Após conectar via VPN e SSH no Santos Dumont, acesse a sua pasta scratch:

```bash
cd /scratch/insperhpc/seu_usuario/
```

Como o SDumont é um ambiente compartilhado com vários usuários e várias necessidades de ferramentas diferentes, as bilbiotecas ficam instaladas mas não ficam disponíveis automaticamente para não gerar conflito entre os projetos. Para utilizar uma ferramenta ou biblioteca específica, é preciso carregar o modulo. Para verificar modulos disponpiveis no sistema utilize:

```bash
module avail
```

Para filtrar alguma biblioteca especifica e verificar as versões disponíveis, utilize:

```bash
module avail mpi
```

Para carregar o módulo, utilize:

```bash
module load openmpi/gnu/4.1.8_sequana
```

Para verificar os módulos carregados no seu ambiente, utilize
```bash
module list
```


??? info "hello.cpp"
    ```cpp
    #include <mpi.h>
    #include <iostream>
    #include <sstream>

    using namespace std;

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);

        int rank, total, tamanho;
        char nome[MPI_MAX_PROCESSOR_NAME];

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &total);
        MPI_Get_processor_name(nome, &tamanho);

        // Cada processo faz uma computação simples para identificar sua tarefa.
        int resultado = rank * rank;

        // Monta o registro de cada processo antes de imprimi-lo.
        ostringstream mensagem;
        mensagem << "Rank " << rank << " de " << total
                << " | Nó: " << nome
                << " | Quadrado do rank: " << resultado << '\n';
        cout << mensagem.str() << flush;

        MPI_Finalize();
        return 0;
    }
    ```

O código demo mostra o rank, o tamanho do comunicador, o nó e o resultado local. A ordem das linhas pode variar porque os processos executam concorrentemente. Montar uma mensagem antes de imprimir reduz a fragmentação dos prints, mas não estabelece uma ordem global.

Para compilar:
```bash
mpic++ -O3 hello.cpp -o hello
```
Se alterar `hello.cpp`, recompile antes de executar novamente.

Após carregar o módulo e compilar:

```bash
srun --partition=sequana_cpu_dev --mem=1G \
     --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --cpus-per-task=1 --time=00:01:00 ./hello
```

Uma saída possível seria:

```text
srun: job 11618883 queued and waiting for resources
srun: job 11618883 has been allocated resources
Rank 1 | Nó: sdumont6237
: Minha computação é = 1
Rank 0 | Nó: sdumont6237
Tarefa: calcular o quadrado de cada ID.
Temos 2 nós.
Rank 1 | Nó: sdumont6238
: Minha computação é = 1
Rank 0 | Nó: sdumont6238
Tarefa: calcular o quadrado de cada ID.
Temos 2 nós.
```

**Por que não adicionamos --mpi=pmix?**

A forma de integração depende de como o módulo MPI e o Slurm foram configurados. MPICH pode usar diferentes interfaces de inicialização. Por isso os comandos no Santos Dumont e no Franky tem algumas diferenças.


Para submeter um job com sbatch:

`run.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=hello-mpi
#SBATCH --partition=sequana_cpu_dev
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --output=mpi-%j.out

echo "Job: $SLURM_JOB_ID"
echo "Nós alocados: $SLURM_JOB_NODELIST"
echo "Tarefas: $SLURM_NTASKS"

# Inicia a mesma quantidade de processos solicitada ao Slurm.
mpiexec -np "$SLURM_NTASKS" ./hello
```

Submeta com o comando:

```bash
sbatch run.slurm
```

Para monitorar a fila:

```bash
squeue -u "$USER"
```

Para um job de exemplo com ID `11618863`:

```bash
cat mpi-11618863.out
```

Substitua o número pelo ID da sua submissão. Para cancelar um job seu que não é mais necessário:

```bash
scancel 11618863
```

