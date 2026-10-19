# Comunicação coletiva MPI: Scatter e Gather

Na aula anterior, usamos `MPI_Bcast` para compartilhar os mesmos dados com todos os processos e `MPI_Reduce` para combinar resultados. Agora vamos mudar a organização: em vez de enviar uma cópia completa do vetor a todos, queremos **entregar uma fatia diferente para cada processo**. Depois da computação, queremos **reunir as fatias calculadas**, preservando seus elementos.

**Scatter** é uma operação coletiva que distribui os dados de um processo raiz em blocos, entregando uma parte a cada processo. Por exemplo, um vetor com 12 números pode ser dividido entre quatro processos, cada um recebendo três números.

**Gather** é uma operação coletiva que reúne os blocos fornecidos pelos processos em um único vetor na raiz, organizados pela ordem dos ranks. Ele preserva os elementos recebidos, sem somá-los ou combiná-los matematicamente.


### Scatter: distribuir uma fatia para cada processo

Considere o vetor abaixo, inicialmente disponível apenas no rank 0:

```cpp
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
```

Com quatro processos e três elementos por bloco, o efeito do Scatter é:

| Rank | Posições no vetor da raiz | Bloco recebido |
| --- | --- | --- |
| 0 | 0 a 2 | `{1, 2, 3}` |
| 1 | 3 a 5 | `{4, 5, 6}` |
| 2 | 6 a 8 | `{7, 8, 9}` |
| 3 | 9 a 11 | `{10, 11, 12}` |


```cpp
MPI_Scatter(
    dados.data(),     // Vetor completo
    bloco,            // Quantidade de elementos enviada A CADA processo.
    MPI_INT,          // Tipo dos elementos enviados.
    local.data(),     // Buffer onde este processo recebe sua fatia.
    bloco,            // Quantidade de elementos recebida por este processo.
    MPI_INT,          // Tipo dos elementos recebidos.
    0,                // Rank da raiz que distribui os dados.
    MPI_COMM_WORLD    // Comunicador participante.
);
```
O MPI transfere os elementos do `vector`, não o objeto C++. Por isso, usamos `.data()` e reservamos espaço com o construtor ou `resize()` antes da chamada. Nos demais ranks, o buffer de envio do Scatter é ignorado. 


**Exemplo completo: receber e identificar uma fatia**

`scatter.cpp`:

```cpp
#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total, tamanho;
    char nome[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Get_processor_name(nome, &tamanho);

    // Cada processo receberá três números.
    int bloco = 3;

    vector<int> dados;
    vector<int> local(bloco);

    // Somente o rank 0 prepara o vetor completo.
    if (rank == 0) {
        int quantidade = total * bloco;
        dados.resize(quantidade);

        for (int i = 0; i < quantidade; i++) {
            dados[i] = i + 1;
        }
    }

    // Distribui três números para cada processo.
    // Todos chamam, inclusive o rank 0.
    MPI_Scatter(
        dados.data(),     // Vetor completo, utilizado apenas na raiz.
        bloco,            // Elementos enviados para cada processo.
        MPI_INT,          // Tipo dos elementos enviados.
        local.data(),     // Onde este processo recebe sua fatia.
        bloco,            // Elementos recebidos por este processo.
        MPI_INT,          // Tipo dos elementos recebidos.
        0,                // Rank da raiz.
        MPI_COMM_WORLD
    );

    // Mostra qual fatia foi recebida por este processo.
    cout << "Rank " << rank
         << " | Nó: " << nome
         << " | Fatia recebida: ";

    for (int i = 0; i < bloco; i++) {
        cout << local[i] << ' ';
    }

    cout << endl;

    MPI_Finalize();
    return 0;
}
```

```bash
mpic++ -O2 scatter.cpp -o scatter

srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=2 --ntasks=4 --ntasks-per-node=2 \
     --time=00:05:00 ./scatter
```

Com quatro processos, o vetor inicial tem 12 números. Com dois, tem seis: neste primeiro exemplo, mantemos três elementos por processo e adaptamos o tamanho do problema à quantidade de ranks. A ordem dos prints pode variar.

### Gather: reunir blocos na raiz

Depois da computação, cada processo pode ter um bloco de resultados. `MPI_Gather` reúne esses blocos em um vetor na raiz, organizados pela ordem dos ranks.

Se cada processo elevar ao quadrado os valores do exemplo anterior:

| Rank | Bloco calculado | Posições no vetor reunido |
| --- | --- | --- |
| 0 | `{1, 4, 9}` | 0 a 2 |
| 1 | `{16, 25, 36}` | 3 a 5 |
| 2 | `{49, 64, 81}` | 6 a 8 |
| 3 | `{100, 121, 144}` | 9 a 11 |

O resultado na raiz será:

```cpp
{1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144}
```

**Gather preserva os elementos; não os soma nem calcula um máximo.** Para combinar contribuições por uma operação matemática, usamos Reduce.

### Leitura da chamada

```cpp
MPI_Gather(
    local.data(),       // Bloco que este processo envia.
    bloco,              // Elementos enviados por este processo.
    MPI_INT,            // Tipo dos elementos enviados.
    reunidos.data(),    // Vetor final: significativo apenas na raiz.
    bloco,              // Elementos recebidos DE CADA processo.
    MPI_INT,            // Tipo dos elementos recebidos.
    0,                  // Rank da raiz que reúne os dados.
    MPI_COMM_WORLD      // Comunicador participante.
);
```

Para quatro ranks enviando três inteiros cada, a raiz reserva espaço para 12 elementos, mas informa `recvcount = 3`. **Esse parâmetro não é a quantidade total reunida.**

Somente a raiz precisa alocar o vetor final. Nos outros processos, o buffer de recebimento é ignorado e pode ser `nullptr`. Todos precisam fornecer seu bloco de envio.

Os blocos ocupam posições determinadas pelo rank, mesmo que os processos terminem seus cálculos em momentos diferentes. A ordem das mensagens impressas na tela não altera a organização do vetor reunido.

**Exemplo completo: reunir um valor de cada processo**

`gather.cpp`:

```cpp
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total, tamanho;
    char nome[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Get_processor_name(nome, &tamanho);

    // Cada rank calcula um valor diferente.
    int valor = (rank + 1) * (rank + 1);
    vector<int> reunidos;
    if (rank == 0)
        reunidos.resize(total);

    ostringstream mensagem;
    mensagem << "Rank " << rank << " | Nó: " << nome
             << " | Valor calculado: " << valor << '\n';
    cout << mensagem.str() << flush;

    // Um inteiro de cada rank ocupa uma posição na raiz.
    MPI_Gather(&valor, 1, MPI_INT,
               rank == 0 ? reunidos.data() : nullptr,
               1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Valores reunidos por ordem de rank: ";
        for (int item : reunidos)
            cout << item << ' ';
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
```

```bash
mpic++ -O2 gather.cpp -o gather

srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=2 --ntasks=4 --ntasks-per-node=2 \
     --time=00:05:00 ./gather
```

Com quatro processos, o vetor final contém `1 4 9 16`. O resultado global fica somente na raiz; os demais ranks mantêm seus valores locais.

## Broadcast, Scatter, Gather e Reduce: qual é a diferença?

| Operação | O que acontece com os dados? | Exemplo com quatro ranks |
| --- | --- | --- |
| Broadcast | Todos recebem os mesmos elementos da raiz | Todos recebem o vetor completo |
| Scatter | Cada rank recebe um bloco correspondente do vetor da raiz | Cada um recebe três de 12 números |
| Gather | Blocos dos ranks ocupam faixas distintas no vetor da raiz | Quatro blocos de três formam um vetor de 12 |
| Reduce | Contribuições são combinadas por uma operação | Quatro somas parciais produzem uma soma global |

Scatter e Gather lidam com blocos. Broadcast replica dados. Reduce combina contribuições. A escolha depende do que queremos obter ao final da comunicação.

### Tudo junto: distribuir, calcular e reunir

Vamos retomar o problema dos quadrados. Desta vez, o rank 0 cria os números de 1 a 12, cada processo recebe apenas uma fatia e a raiz reúne **todos os quadrados**, em vez de receber somente a soma.

| Etapa | Operação | Resultado |
| --- | --- | --- |
| Preparar o vetor | Computação na raiz | Números de 1 a 12 |
| Distribuir fatias | `MPI_Scatter` | Um bloco local por processo |
| Calcular os quadrados | Computação local | Bloco local transformado |
| Reunir blocos | `MPI_Gather` | Vetor dos quadrados na raiz |

O rank 0 recebe sua fatia e calcula como os demais. O vetor completo de entrada e o de saída ficam na raiz; os outros ranks precisam apenas de seus blocos locais.


`scatter_gather.cpp`:

```cpp
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total, tamanho;
    char nome[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Get_processor_name(nome, &tamanho);

    const int N = 14;

    int bloco = N / total;
    int sobra = N % total;

    // Quantidade de elementos distribuídos pelo Scatter.
    int distribuidos = bloco * total;

    vector<int> dados;
    vector<int> reunidos;
    vector<int> local(bloco);

    if (rank == 0) {
        dados.resize(N);
        reunidos.resize(N);

        for (int i = 0; i < N; i++) {
            dados[i] = i + 1;
        }
    }

    // Distribui apenas os blocos iguais.
    // Os elementos restantes continuam no vetor da raiz.
    MPI_Scatter(dados.data(), bloco, MPI_INT,
                local.data(), bloco, MPI_INT,
                0, MPI_COMM_WORLD);

    ostringstream mensagem;
    mensagem << "Rank " << rank << " | Nó: " << nome
             << " | Recebeu: ";

    for (int i = 0; i < bloco; i++) {
        mensagem << local[i] << ' ';
    }

    // Todos calculam o quadrado dos elementos de seu bloco.
    for (int i = 0; i < bloco; i++) {
        local[i] = local[i] * local[i];
    }

    mensagem << "| Calculou: ";

    for (int i = 0; i < bloco; i++) {
        mensagem << local[i] << ' ';
    }

    mensagem << '\n';
    cout << mensagem.str() << flush;

    // Reúne os blocos nas primeiras posições do vetor final.
    MPI_Gather(local.data(), bloco, MPI_INT,
               reunidos.data(), bloco, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calcula os elementos que não foram distribuídos.
        // Eles estão nas posições "distribuidos" até N - 1.
        if (sobra > 0) {
            cout << "Rank 0 | Nó: " << nome
                 << " | Calculando as sobras: ";

            for (int i = distribuidos; i < N; i++) {
                reunidos[i] = dados[i] * dados[i];

                cout << dados[i] << "² = "
                     << reunidos[i] << ' ';
            }

            cout << endl;
        }

        cout << "Vetor final: ";

        for (int i = 0; i < N; i++) {
            cout << reunidos[i] << ' ';
        }

        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
```
O rank 0 cria um vetor com os números de **1 a 14**. Com quatro processos, a divisão resulta em **três elementos por processo e dois elementos restantes**.

O `MPI_Scatter` distribui os primeiros 12 números em blocos iguais. Cada processo, inclusive a raiz, calcula o quadrado dos três valores recebidos:

| Processo | Valores recebidos | Quadrados calculados |
| --- | --- | --- |
| Rank 0 | 1, 2, 3 | 1, 4, 9 |
| Rank 1 | 4, 5, 6 | 16, 25, 36 |
| Rank 2 | 7, 8, 9 | 49, 64, 81 |
| Rank 3 | 10, 11, 12 | 100, 121, 144 |

Depois, `MPI_Gather` reúne esses blocos nas primeiras 12 posições do vetor final, na ordem dos ranks. Os números **13 e 14 não foram distribuídos**: o rank 0 calcula seus quadrados e os coloca nas duas últimas posições.

Assim, o nó raiz imprime:

```text
Vetor final: 1 4 9 16 25 36 49 64 81 100 121 144 169 196
```

### Compilação e execução

```bash
mpic++ -O2 scatter_gather.cpp -o scatter_gather


srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     --time=00:05:00 ./scatter_gather



Submissão com `sbatch`

`run.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=mpi-scatter-gather
#SBATCH --partition=merry_cpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=scatter-gather-%j.out

srun --mpi=pmix ./scatter_gather
```

```bash
sbatch run.slurm
```


## Para entender se você entendeu

1. Substitua o quadrado pelo dobro de cada elemento.
2. Use `double` e `MPI_DOUBLE` para trabalhar com valores decimais.
3. Faça cada processo calcular uma soma parcial e use Gather com um elemento por rank para reunir essas somas. Depois, a raiz deve somar o vetor recebido e comparar com a solução que usa Reduce.
4. Use o rank 1 como raiz de Scatter e Gather, em vez do rank 0, faça todos os ajustes necessários. 


