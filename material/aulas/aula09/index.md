# Programação Distribuída com MPI

No MPI existem duas formas de comunicação: a ponto-a-ponto e a coletiva.  
Na comunicação ponto-a-ponto, apenas dois processos participam: um envia (`MPI_Send`) e outro recebe (`MPI_Recv`).  


## Token em anel

No anel, um único “pacote” (token) dá uma volta completa: cada processo recebe o vetor, acrescenta sua informação e passa adiante. Assim, há N mensagens para N processos.

`anel.cpp` 
```cpp
#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // identifica o rank de cada processo
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // descobre quantos nós estão rodando

    const int chunk = 2048;                // cada processo envia seu vetor

    // Cada processo prepara seu vetor local 
    vector<int> local(chunk);
    for (int i = 0; i < chunk; i++) {
        // preenche o vetor com o dado
        local[i] = rank * 1000 + i;
    }

    // O root precisa de um buffer para juntar TODOS os blocos: size * chunk
    vector<int> result;
    if (rank == 0) result.resize(size * chunk);

    // Vamos medir o tempo da troca de dados
    double t0 = MPI_Wtime();

    if (rank == 0) {
        // O root primeiro coloca o próprio bloco no lugar correto
        //    (posição 0 .. chunk-1, já que rank==0)
        std::copy(local.begin(), local.end(), result.begin());

        // Recebe de cada rank (1..size-1) o respectivo bloco
        //    e grava diretamente na “faixa” correta do vetor final.
        for (int src = 1; src < size; ++src) {
            MPI_Recv(result.data() + src * chunk,   // destino no buffer final
                    chunk, MPI_INT,                 // quantos / tipo
                    src,                            // origem (rank esperado)
                    0, MPI_COMM_WORLD,              // tag/comunicador
                    MPI_STATUS_IGNORE);
        }
    } else {
        // Ranks != 0 enviam seu bloco para o root
        MPI_Send(local.data(), chunk, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();

    // Root imprime um resumo e alguns elementos para verificação
    if (rank == 0) {
        cout << "[Anel] N=" << size
            << " tempo=" << (t1 - t0) << " s" << endl;

        // Mostra só os 5 primeiros valores de cada bloco para não poluir
        for (int p = 0; p < size; p++) {
            cout << "Rank " << p << ": ";
            for (int i = 0; i < 5; i++)
                cout << result[p * chunk + i] << " ";
            cout << "...\n";
        }
    }

    MPI_Finalize();
    return 0;
}
```

#### Como compilar e rodar

```bash
mpic++ -O2 anel.cpp -o anel
```

Submetendo ao Slurm

```bash
#!/bin/bash
#SBATCH --job-name=anel               # nome do job
#SBATCH --output=anel.%j.txt           # saída em arquivo
#SBATCH --time=00:05:00               # tempo limite
#SBATCH --nodes=5                    # número de nós (Computadores)
#SBATCH --ntasks=5                    # número total de Ranks MPI
#SBATCH --cpus-per-task=1             # CPUs por processo
#SBATCH --partition=normal            # Fila do SLURM
#SBATCH --mem=1GB                    # memória solicitada

echo "=== Executando com 2 processos ==="
mpirun -np 2 ./anel
echo "=================================="

echo "=== Executando com 3 processos ==="
mpirun -np 3 ./anel
echo "=================================="

echo "=== Executando com 4 processos ==="
mpirun -np 4 ./anel
echo "=================================="

echo "=== Executando com 5 processos ==="
mpirun -np 5 ./anel
echo "=================================="

```


## Desafio Ping-Pong: Latência e Largura de Banda

No MPI, a comunicação pode ocorrer de forma ponto-a-ponto, onde dois processos trocam mensagens diretamente usando
 `MPI_Send`/`MPI_Recv`

Para avaliar a qualidade da comunicação em um sistema de HPC o ping-pong pode ser utilizado.
Esse experimento permite medir dois conceitos fundamentais:

* **Latência** → tempo necessário para realizar a comunicação entre dois processos.
* **Largura de banda** → quantidade de dados transferidos por unidade de tempo.


A partir do exemplo `anel.cpp`, implemente um programa chamado `pingpong.cpp`.

O programa deve:

1. Utilizar 2 processos MPI.
2. O processo 0 envia uma mensagem ao processo 1.
3. O processo 1 recebe a mensagem e envia uma resposta para o processo 0.
4. Esse ciclo deve ser repetido várias vezes.


Execute o ping-pong variando o tamanho da mensagem.

Por exemplo:

| Tamanho da mensagem | Repetições |
| ------------------- | ---------- |
| 8 Bytes             | 10000      |
| 64 Bytes            | 10000      |
| 512 Bytes           | 10000      |
| 4 KB                | 10000      |
| 32 KB               | 10000      |
| 256 KB              | 10000      |


A partir do tempo medido, calcule a latência média e largura_de_banda:

A latência pode ser aproximada por:

$$
latencia = \frac{tempo\ total}{2 \times repetições}
$$

O 2 é porque cada troca possui uma mensagem de envio e uma mensagem de resposta


$$
larguraBanda =
\frac{tamanho\ da\ mensagem}{tempo\ medio\ de\ transferencia}
$$

Apresente o resultado em **MB/s**.

Monte uma tabela com os resultados:

| Tamanho da mensagem | Tempo total | Latência | Largura de banda |
| ------------------- | ----------- | -------- | ---------------- |


## Perguntas Teóricas


1- Qual a diferença entre latência e largura de banda em sistemas de comunicação?

2- Por que mensagens muito pequenas são mais influenciadas pela latência?

3- O que acontece com a largura de banda quando o tamanho da mensagem aumenta?

4- Por que o ping-pong é adequado para medir desempenho de comunicação em sistemas de HPC?

# Esta atividade não precisa ser entregue. Bom fim de semana!
