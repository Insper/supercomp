
Esta etapa amplia a introdução da aula e prepara o desafio do handout. Na comunicação ponto-a-ponto, um processo envia dados e outro os recebe.

```cpp
MPI_Send(buffer, quantidade, tipo, destino, tag, comunicador);
MPI_Recv(buffer, capacidade, tipo, origem, tag, comunicador, status);
```

- **Buffer:** região de memória com os dados a enviar ou receber.
- **Quantidade/capacidade:** número de elementos, não necessariamente bytes.
- **Tipo:** por exemplo, `MPI_INT` para inteiros ou `MPI_BYTE` para bytes.
- **Origem/destino:** rank do outro processo, não nome do computador.
- **Tag:** identificador que ajuda a distinguir mensagens.
- **Comunicador:** contexto no qual ocorre a troca, aqui `MPI_COMM_WORLD`.

`MPI_Recv` espera até que a mensagem correspondente seja recebida. `MPI_Send` é bloqueante: ao retornar, o buffer de envio pode ser reutilizado, mas isso não garante que o destinatário já tenha concluído o recebimento. Portanto, não dependa de buffers internos para evitar travamentos; planeje a ordem de envio e recebimento.

Comunicações coletivas envolvem todos os processos do comunicador. Exemplos são `MPI_Bcast`, `MPI_Scatter` e `MPI_Gather`; elas ficam como continuação do estudo.

### Coleta centralizada: cada trabalhador envia ao rank 0

O código abaixo reúne blocos de inteiros usando envios e recebimentos individuais. É uma **coleta centralizada**, não um anel: todos os trabalhadores enviam diretamente ao rank 0.

Em um anel verdadeiro, o token passaria de um rank para o próximo e retornaria ao inicial. A quantidade e o caminho das mensagens seriam diferentes. Aqui há `total - 1` mensagens de dados.

Salve como `coleta.cpp`:

```cpp
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);

    const int bloco = 2048;
    std::vector<int> local(bloco);

    // Cada processo cria seu próprio bloco, com valores identificáveis.
    for (int i = 0; i < bloco; ++i)
        local[i] = rank * 10000 + i;

    // Apenas o coordenador precisa armazenar todos os blocos.
    std::vector<int> reunidos;
    if (rank == 0)
        reunidos.resize(total * bloco);

    // Aproxima o início da etapa nos processos participantes.
    MPI_Barrier(MPI_COMM_WORLD);
    double inicio = MPI_Wtime();

    if (rank == 0) {
        // O bloco do rank 0 já está localmente disponível.
        std::copy(local.begin(), local.end(), reunidos.begin());

        // Recebe cada bloco na faixa correspondente ao rank de origem.
        for (int origem = 1; origem < total; ++origem) {
            MPI_Recv(reunidos.data() + origem * bloco,
                     bloco, MPI_INT, origem, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // Cada trabalhador envia seu bloco ao coordenador.
        MPI_Send(local.data(), bloco, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    double fim = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Coleta com " << total << " processos: "
                  << fim - inicio << " s\n";

        for (int p = 0; p < total; ++p) {
            std::cout << "Bloco do rank " << p << ": ";
            for (int i = 0; i < 5; ++i)
                std::cout << reunidos[p * bloco + i] << ' ';
            std::cout << "...\n";
        }
    }

    MPI_Finalize();
    return 0;
}
```

Compile e execute com dois processos em dois nós:

```bash
mpic++ -std=c++11 -O2 coleta.cpp -o coleta
srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --time=00:01:00 ./coleta
```

Também é possível reutilizar `run.slurm`, trocando o executável da última linha por `./coleta`.

**Leitura do resultado:** o primeiro bloco começa em 0; o segundo, em 10000; o terceiro, se houver, em 20000. O deslocamento `origem * bloco` coloca cada bloco na posição correta do vetor final.

`MPI_Barrier` é uma operação coletiva de sincronização: todos devem chamá-la. Ela foi introduzida aqui apenas para organizar a medição. O tempo impresso é o intervalo medido no rank 0, incluindo a cópia local e os recebimentos; não é uma medida isolada da latência da rede. Os prints ficam fora desse intervalo.

## 7. Desafio ping-pong: latência e largura de banda

Implemente `pingpong.cpp` usando **exatamente dois processos MPI**. A ideia é medir o tempo de uma mensagem de ida e de uma resposta com o mesmo tamanho.

### Sequência de uma repetição

| Rank 0 | Rank 1 |
| --- | --- |
| Envia o buffer ao rank 1 | Recebe o buffer do rank 0 |
| Recebe o buffer de volta | Envia o buffer de volta ao rank 0 |

Repita essa sequência várias vezes. O rank 0 envia primeiro; o rank 1 recebe primeiro. Isso evita que os dois esperem por uma mensagem que ninguém enviou.

### Roteiro de implementação

1. Inicialize o MPI e obtenha `rank` e `total`.
2. Se `total != 2`, faça o rank 0 mostrar uma orientação, finalize o MPI em todos os processos e encerre o programa.
3. Para cada tamanho da tabela abaixo, crie um buffer de bytes, como `std::vector<char>`.
4. Faça 100 trocas de aquecimento, sem incluí-las no tempo medido.
5. Faça todos os processos chamarem `MPI_Barrier` antes da medição.
6. No rank 0, registre `inicio = MPI_Wtime()`.
7. Execute 10000 repetições de ida e volta. Use `MPI_BYTE` e envie o número de bytes escolhido.
8. No rank 0, registre `fim = MPI_Wtime()` e calcule as métricas.
9. Imprima os resultados apenas depois de encerrar a medição. Não coloque prints dentro do laço.

Use os tempos inicial e final do **mesmo rank**. Não subtraia um instante medido no rank 0 de outro medido no rank 1: os relógios MPI não precisam estar sincronizados.

| Tamanho | Bytes por mensagem | Repetições medidas |
| --- | ---: | ---: |
| 8 B | 8 | 10000 |
| 64 B | 64 | 10000 |
| 512 B | 512 | 10000 |
| 4 KiB | 4096 | 10000 |
| 32 KiB | 32768 | 10000 |
| 256 KiB | 262144 | 10000 |

### Como calcular

Seja `T` o tempo total medido em segundos, `R` o número de repetições e `B` o tamanho de uma mensagem em bytes. Cada repetição transfere duas mensagens de tamanho `B`.

O tempo médio de ida e volta é:

$$
t_{ida\ e\ volta} = \frac{T}{R}
$$

Uma estimativa do tempo médio de transferência em um sentido é:

$$
t_{um\ sentido} \approx \frac{T}{2R}
$$

Para mensagens pequenas, essa estimativa é usada como aproximação de latência. Para mensagens maiores, inclui também o custo de transferir o conteúdo; não representa apenas a latência inicial.

Para apresentar o valor em microssegundos:

$$
t_{\mu s} = \frac{T}{2R} \times 10^6
$$

A largura de banda efetiva do ping-pong, em MB/s, é:

$$
BW_{MB/s} = \frac{2BR}{T \times 10^6}
$$

Aqui, **1 MB = 1000000 bytes**. Se usar MiB/s, divida por `2^20` em vez de `10^6` e identifique a unidade. A medida é efetiva para esse experimento; não é necessariamente a capacidade máxima do enlace.

**Exemplo de cálculo, com números fictícios:** para `B = 4096`, `R = 10000` e `T = 0,2 s`, o tempo estimado por sentido é `10 µs` e a largura de banda é `409,6 MB/s`.

### Compare dentro do nó e entre nós

Após compilar:

```bash
mpic++ -std=c++11 -O2 pingpong.cpp -o pingpong
```

Execute dois processos no mesmo nó:

```bash
srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=1 --ntasks=2 --ntasks-per-node=2 \
     --time=00:05:00 ./pingpong
```

Depois, um processo em cada nó:

```bash
srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --time=00:05:00 ./pingpong
```

Registre os nomes dos nós antes da medição, usando `MPI_Get_processor_name`. Repita cada configuração pelo menos três vezes e registre a mediana, mantendo as mesmas repetições e tamanhos. Assim você reduz o efeito de uma execução excepcional.

| Configuração | Bytes | Repetições | Tempo total (s) | Tempo por sentido (µs) | Banda efetiva (MB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mesmo nó | 8 | 10000 | | | |
| Nós diferentes | 8 | 10000 | | | |
| Mesmo nó | 4096 | 10000 | | | |
| Nós diferentes | 4096 | 10000 | | | |

Complete a tabela também para os demais tamanhos. Registre os valores de cada execução e indique quais valores da tabela representam a mediana.

### Perguntas para análise

1. Qual é a diferença entre latência e largura de banda?
2. Por que o custo inicial pesa mais nas mensagens pequenas?
3. A banda efetiva sempre aumenta com o tamanho da mensagem? O que os seus dados mostram?
4. Por que repetir e aquecer o experimento antes de medir?
5. Como os resultados mudam entre processos no mesmo nó e em nós diferentes?
6. O ping-pong mede apenas a rede física? Que outros custos estão envolvidos?
7. Por que aumentar o número de processos não garante reduzir o tempo de qualquer programa?

## 8. Se algo não funcionar

| Sintoma | O que verificar |
| --- | --- |
| Apenas rank 0 e `total = 1` | O programa foi chamado sozinho? Use `srun --mpi=pmix ./programa` no script |
| Ranks repetidos e várias mensagens do coordenador | Verifique se você usou `srun ... mpirun ...`; use um único lançador |
| Job pendente com `PartitionNodeLimit` | Consulte `scontrol show partition merry_cpu` e compare o pedido com os limites |
| `command not found` | Confira o ambiente e peça orientação ao responsável pelo cluster |
| `Stale file handle` | Há um problema de acesso ao sistema de arquivos compartilhado; informe ao administrador |
| Binário não encontrado no nó | Use uma pasta compartilhada e confira nome, caminho e permissão do executável |
| Código alterado, saída antiga | Recompile, submeta novamente e leia o arquivo do novo ID de job |
| Prints fora de ordem | É esperado em processos concorrentes; não indica rank incorreto |
| Aviso de versão de `libxml` | Informe ao responsável pelo ambiente; é distinto da quantidade de ranks ou nós |

Use dois hífens comuns nas opções: `--mem=1G`, e não um travessão. As diretivas `#SBATCH` devem aparecer antes do primeiro comando do script. Não coloque variáveis de shell dentro dessas diretivas esperando expansão automática.
