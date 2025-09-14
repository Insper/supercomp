# MPI e as Operações Coletivas

No MPI existem duas formas de comunicação: a ponto-a-ponto e a coletiva.  
Na comunicação ponto-a-ponto, apenas dois processos participam: um envia (`MPI_Send`) e outro recebe (`MPI_Recv`).  

Já nas operações coletivas, todos os processos de um comunicador (`MPI_COMM_WORLD`) precisam participar da mesma chamada ao mesmo tempo.  

Quando dizemos que “um processo chama a coletiva”, significa que ele executa uma função como `MPI_Gather`, `MPI_Scatter` ou `MPI_Reduce`, junto com todos os outros processos do comunicador. Se algum processo não participar, os demais ficarão bloqueados, pois o MPI espera que todos estejam presentes para completar a operação.

Mesmo quando existe um root, um processo principal que organiza a comunicação, todos os outros processos ainda assim chamam a função. O root apenas desempenha um papel diferente, como receber todos os dados (`Gather`), distribuí-los (`Scatter`) ou armazenar o resultado final (`Reduce`).

## MPI_Gather

O `MPI_Gather` coleta dados de todos os processos e junta em um único buffer no root.  
Cada processo envia a mesma quantidade de dados. O root recebe os blocos organizados na ordem dos ranks.

As coletivas no MPI são operações globais que substituem sequências de envios e recebimentos individuais. Elas funcionam porque todos os processos participam da chamada, garantindo sincronização e consistência. O root organiza a operação, mas nunca atua sozinho: ele é apenas mais um processo dentro do grupo. 

# Token em anel

No anel, um único “pacote” (token) dá uma volta completa: cada processo recebe o vetor, acrescenta sua informação e passa adiante. Assim, há N mensagens para N processos.

`anel_gather.cpp`
```cpp
#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // ------------------------------------------------------------------
    // [Etapa 0] Identificação de ranks e processos
    // ------------------------------------------------------------------
    int rank = -1, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // meu rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // total de processos

    // ------------------------------------------------------------------
    // [Etapa 1] setup dos vetores e buffers
    // ------------------------------------------------------------------
    const int chunk = 2048;                 
    vector<int> local(chunk, -1);

    if (rank != 0) {
        // Preenche vetor local do worker de forma identificável
        for (int i = 0; i < chunk; ++i) local[i] = rank * 1000 + i;
    }

    // ------------------------------------------------------------------
    // [Etapa 2] Tabelas de envio por rank 
    //           counts[r] = quantos elementos o rank r envia
    //           displs[r] = deslocamento (em elementos) no recvbuf do ROOT
    // ------------------------------------------------------------------
    vector<int> counts(size, 0);
    vector<int> displs(size, 0);

    // ranks 1..size-1 enviam 'chunk'
    for (int r = 1; r < size; ++r) counts[r] = chunk;

    // displs compactado: r=1 ocupa [0..chunk-1], r=2 ocupa [chunk..2*chunk-1], etc.
    for (int r = 2; r < size; ++r) displs[r] = displs[r - 1] + counts[r - 1];

    // Tamanho total a receber no ROOT: soma(counts[r]) = (size-1)*chunk
    int total_recv = 0;
    for (int r = 0; r < size; ++r) total_recv += counts[r];

    // Buffer de recepção do ROOT
    vector<int> result;
    if (rank == 0) result.resize(total_recv);

    // ------------------------------------------------------------------
    // [Etapa 3] Parâmetros de envio por processo 
    // ------------------------------------------------------------------
    int   sendcount = 0;
    int*  sendbuf   = nullptr;
    int* recvbuf = nullptr;               


    if (rank =! 0) {
        // Workers enviam 'chunk' inteiros do seu vetor local
        sendcount = chunk;
        sendbuf   = local.data();
    }

    if (rank == 0) recvbuf = result.data();

    // ------------------------------------------------------------------
    // [Etapa 4] Coletiva 
    // ------------------------------------------------------------------
    double t0 = MPI_Wtime();

    MPI_Gatherv(
        /*sendbuf   */ sendbuf,
        /*sendcount */ sendcount,
        /*sendtype  */ MPI_INT,
        /*recvbuf   */ recvbuf,
        /*recvcounts*/ counts.data(),
        /*displs    */ displs.data(),
        /*recvtype  */ MPI_INT,
        /*root      */ 0,
        /*comm      */ MPI_COMM_WORLD
    );

    double t1 = MPI_Wtime();

    // ------------------------------------------------------------------
    // [Etapa 5] Saída: ROOT gerencia e imprime
    // ------------------------------------------------------------------
    if (rank == 0) {
        cout << "[GATHER] N=" << size
             << " (workers=" << (size - 1)
             << ") tempo=" << (t1 - t0) << " s\n";

        // Amostra: 5 primeiros elementos de cada worker
        for (int r = 1; r < size; ++r) {
            cout << "Rank " << r << ": ";
            int base = displs[r];
            for (int i = 0; i < 5; ++i) cout << result[base + i] << ' ';
            cout << "...\n";
        }
    }

    MPI_Finalize();
    return 0;
}


```

#### Como compilar e rodar

```bash
mpic++ -O2 anel_gather.cpp -o anel_gather
```

Submetendo ao Slurm

```bash
#!/bin/bash
#SBATCH --job-name=gather               # nome do job
#SBATCH --output=gather.%j.txt           # saída em arquivo
#SBATCH --time=00:05:00               # tempo limite
#SBATCH --nodes=5                     # número de nós (Computadores)
#SBATCH --ntasks=5                    # número total de Ranks MPI
#SBATCH --cpus-per-task=1             # CPUs por processo
#SBATCH --partition=normal            # Fila do SLURM
#SBATCH --mem=2GB                    # memória solicitada

echo "=== Executando com 2 processos ==="
time mpirun -np 2 ./anel_gather
echo ""

echo "=== Executando com 3 processos ==="
time mpirun -np 3 ./anel_gather
echo ""

echo "=== Executando com 4 processos ==="
time mpirun -np 4 ./anel_gather
echo ""

echo "=== Executando com 5 processos ==="
time mpirun -np 5 ./anel_gather
echo ""

```

Este código é equivalente ao anterior, só que agora implementando usando Send/Recv

`anel_p2p.cpp` 
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
        // preenche de forma didática
        local[i] = rank * 1000 + i;
    }

    // O root precisa de um buffer para juntar TODOS os blocos: size * chunk
    vector<int> result;
    if (rank == 0) result.resize(size * chunk);

    // Para comparar com o MPI_Gather, medimos somente a fase de troca de dados
    double t0 = MPI_Wtime();

    if (rank == 0) {
        // 1) O root primeiro coloca o próprio bloco no lugar correto
        //    (posição 0 .. chunk-1, já que rank==0)
        std::copy(local.begin(), local.end(), result.begin());

        // 2) Em seguida, recebe de cada rank (1..size-1) o respectivo bloco
        //    e grava diretamente na “faixa” correta do vetor final.
        for (int src = 1; src < size; ++src) {
            MPI_Recv(result.data() + src * chunk,   // destino no buffer final
                     chunk, MPI_INT,                 // quantos / tipo
                     src,                            // origem (rank esperado)
                     0, MPI_COMM_WORLD,              // tag/comunicador
                     MPI_STATUS_IGNORE);
        }
    } else {
        // Ranks != root enviam seu bloco para o root
        MPI_Send(local.data(), chunk, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();

    // Root imprime um resumo e alguns elementos para verificação
    if (rank == 0) {
        cout << "[P2P] N=" << size
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
mpic++ -O2 anel_p2p.cpp -o p2p
```

Submetendo ao Slurm

```bash
#!/bin/bash
#SBATCH --job-name=p2p               # nome do job
#SBATCH --output=p2p.%j.txt           # saída em arquivo
#SBATCH --time=00:05:00               # tempo limite
#SBATCH --nodes=5                    # número de nós (Computadores)
#SBATCH --ntasks=5                    # número total de Ranks MPI
#SBATCH --cpus-per-task=1             # CPUs por processo
#SBATCH --partition=normal            # Fila do SLURM
#SBATCH --mem=1GB                    # memória solicitada

echo "=== Executando com 2 processos ==="
mpirun -np 2 ./p2p
echo "=================================="

echo "=== Executando com 3 processos ==="
mpirun -np 3 ./p2p
echo "=================================="

echo "=== Executando com 4 processos ==="
mpirun -np 4 ./p2p
echo "=================================="

echo "=== Executando com 5 processos ==="
mpirun -np 5 ./p2p
echo "=================================="

```


