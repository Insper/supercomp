## Conceitos básicos de MPI

MPI (Message Passing Interface) é um padrão para programação paralela em memória compartilhada, o seja, em vários nós de computação. Cada processo é identificado por um **rank** (um número de 0 até `size-1`), todos os processos de um programa MPI formam um **comunicador** (por padrão `MPI_COMM_WORLD`).

## Comunicação ponto a ponto

### Envio e recebimento **bloqueantes**

```cpp
MPI_Send(buffer, count, MPI_CHAR, destino, tag, MPI_COMM_WORLD);
MPI_Recv(buffer, count, MPI_CHAR, origem, tag, MPI_COMM_WORLD, &status);
```

* `buffer`: ponteiro para os dados.
* `count`: número de elementos.
* `MPI_CHAR`, `MPI_INT`, `MPI_DOUBLE` etc.
* `destino` / `origem`: rank do processo de envio/recebimento.
* `tag`: identificador da mensagem (permite diferenciar mensagens).
* `status`: metadados sobre a recepção.

### Envio e recebimento **não-bloqueantes**

```cpp
MPI_Request req;
MPI_Isend(buffer, count, MPI_INT, destino, tag, MPI_COMM_WORLD, &req);
MPI_Wait(&req, MPI_STATUS_IGNORE);
```

```cpp
MPI_Request req;
MPI_Irecv(buffer, count, MPI_INT, origem, tag, MPI_COMM_WORLD, &req);
MPI_Wait(&req, MPI_STATUS_IGNORE);
```

* Permitem **sobrepor comunicação e computação**.
* Precisa sempre de `MPI_Wait` (ou `MPI_Test`).


## Comunicação coletiva

* **Broadcast**

```cpp
MPI_Bcast(buffer, count, MPI_INT, root, MPI_COMM_WORLD);
```

Envia `buffer` do processo `root` para todos.

* **Scatter**

```cpp
MPI_Scatter(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, root, MPI_COMM_WORLD);
```

Distribui partes de um vetor do root para todos.

* **Gather**

```cpp
MPI_Gather(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, root, MPI_COMM_WORLD);
```

Cada processo envia dados para o root.

* **Reduce**

```cpp
MPI_Reduce(&valor_local, &valor_total, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
```

Combina valores de todos os processos (soma, máximo, mínimo, etc.).

* **Barrier**

```cpp
MPI_Barrier(MPI_COMM_WORLD);
```

Todos os processos param até que todos cheguem aqui.

---

## Medição de tempo

```cpp
double t0 = MPI_Wtime();
// ... código ...
double t1 = MPI_Wtime();
if(rank==0) std::cout << "Tempo = " << (t1-t0) << " segundos\n";
```

---

## Estruturas úteis

* **MPI\_Status** → contém informações de mensagens recebidas (como rank origem).

```cpp
MPI_Status status;
MPI_Recv(buf, n, MPI_INT, origem, tag, MPI_COMM_WORLD, &status);
int origem_real; MPI_Get_count(&status, MPI_INT, &origem_real);
```

* **Tags** → permitem diferenciar mensagens diferentes em paralelo.


## Compilação e execução

Compilar:

```bash
mpic++ programa.cpp -o programa
```

Executar com SLURM (arquivo `job.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=mpi_job
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --partition=normal
srun ./programa arg1 arg2
```

Submit:

```bash
sbatch job.slurm
```


Documentação:
[https://www.physics.rutgers.edu/~haule/509/MPI_Guide_C++.pdf](https://www.physics.rutgers.edu/~haule/509/MPI_Guide_C++.pdf)