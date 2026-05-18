
Na aula anterior vimos a comunicação ponto-a-ponto, utilizando operações como `MPI_Send` e `MPI_Recv`, e analisamos como o custo de comunicação depende da **latência** e da **largura de banda** da rede por meio do experimento de *ping-pong*.

Nesta aula vamos dar mais um passo no uso do MPI. Em vez de trabalhar apenas com comunicação **ponto-a-ponto**, vamos ver alguns **padrões de comunicação** e também as **operações coletivas**, onde vários processos participam da troca de dados ao mesmo tempo.

Para isso, vamos implementar alguns exemplos e observar como as mensagens circulam entre os processos.


## **Ping-pong**

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
#SBATCH --output=saida%j.txt
#SBATCH --partition=express
#SBATCH --mem=1GB
#SBATCH --nodes=2
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --export=ALL

# Execute o seu binário com o MPI
mpirun -np $SLURM_NTASKS ./seu_binario

```

### Submeta o job com SLURM:
```bash
sbatch SeuSlurm.slurm
```

### Você deve ver algo como isso:
```bash
[liciascl@head-node mpi]$ cat saida.txt
Processo 2 está ocioso neste exercício.
Processo 3 está ocioso neste exercício.
Processo 0 enviou: Olá
Processo 0 recebeu: Oi
Processo 1 recebeu: Olá
Processo 1 enviou: Oi
Processo 4 está ocioso neste exercício.
```


## **Token em anel**

A ideia é perceber o custo de **coletar informações sequencialmente**.

teste o token em anel: cada rank adiciona uma informação nova ao vetor e passa adiante.
Execute em 2, 3 e 4 nós .
Compare com o mesmo problema usando `MPI_Gather`.


## Token em anel (comunicação sequencial)

A ideia é que cada processo **adicione seu rank em um vetor** e passe esse vetor para o próximo processo. O último processo devolve o vetor ao `rank 0`.

Isso cria uma comunicação **sequencial**, passando por todos os processos.

## Código

```cpp
#include <mpi.h>      // Biblioteca principal do MPI (Message Passing Interface)
#include <iostream>   // Biblioteca para entrada e saída (cout)
#include <vector>     // Biblioteca para usar std::vector

int main(int argc, char** argv) {

    // Inicializa o ambiente MPI
    // Todos os processos começam a execução aqui
    MPI_Init(&argc, &argv);

    int rank, size;

    // Descobre o identificador único (rank) do processo atual
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Descobre quantos processos estão participando do comunicador
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Vetor que representará o "token" que circula entre os processos
    // Cada posição será preenchida por um processo diferente
    std::vector<int> token(size);

    // Processo inicial do anel
    if(rank == 0){

        // O primeiro processo adiciona seu rank na primeira posição do vetor
        token[0] = rank;

        // Envia o vetor para o próximo processo (rank 1)
        MPI_Send(token.data(), size, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Aguarda o retorno do vetor vindo do último processo do anel
        MPI_Recv(token.data(), size, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Após percorrer todos os processos, o vetor estará completo
        std::cout << "Token final: ";

        // Imprime o conteúdo final do vetor
        for(int i=0;i<size;i++)
            std::cout << token[i] << " ";

        std::cout << std::endl;
    }

    else{

        // Recebe o vetor do processo anterior no anel
        MPI_Recv(token.data(), size, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Adiciona seu próprio rank no vetor
        token[rank] = rank;

        // Calcula qual será o próximo processo do anel
        // O operador % garante que o último processo envie de volta ao rank 0
        int next = (rank + 1) % size;

        // Envia o vetor atualizado para o próximo processo
        MPI_Send(token.data(), size, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    // Finaliza o ambiente MPI
    // Todos os processos encerram a execução aqui
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
#SBATCH --output=saida%j.txt
#SBATCH --partition=express
#SBATCH --mem=1GB
#SBATCH --nodes=2
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --export=ALL

# Execute o seu binário com o MPI
mpirun -np $SLURM_NTASKS ./seu_binario

```

### Submeta o job com SLURM:
```bash
sbatch SeuSlurm.slurm
```

### Versão usando MPI_Gather

Agora cada processo simplesmente envia seu valor para o processo `rank 0`, e o MPI faz a coleta automaticamente.

Essa comunicação é **coletiva**, e a biblioteca MPI normalmente usa algoritmos **otimizados em árvore**.

## Código

```cpp
#include <mpi.h>      
#include <iostream>   
#include <vector>     

int main(int argc, char** argv) {

    // Inicializa o ambiente MPI
    // Todos os processos começam a execução aqui
    MPI_Init(&argc, &argv);

    int rank, size;

    // Obtém o identificador único (rank) do processo atual
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Obtém o número total de processos no comunicador
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Cada processo cria um valor local.
    // Aqui usamos o próprio rank apenas como exemplo
    int valor = rank;

    // Vetor que armazenará os resultados coletados
    // Apenas o processo 0 precisa desse vetor
    std::vector<int> resultado;

    // Se este for o processo root (rank 0),
    // ele aloca espaço para armazenar um valor de cada processo
    if(rank == 0)
        resultado.resize(size);

    // Operação coletiva que coleta dados de todos os processos
    // Cada processo envia 1 inteiro (valor)
    // O processo root recebe todos os valores no vetor "resultado"
    MPI_Gather(&valor,          // endereço do dado local que será enviado
               1,               // quantidade de elementos enviados
               MPI_INT,         // tipo de dado enviado
               resultado.data(),// buffer onde o root armazenará os dados
               1,               // quantidade de elementos recebidos de cada processo
               MPI_INT,         // tipo de dado recebido
               0,               // rank do processo root (destino final)
               MPI_COMM_WORLD); // comunicador utilizado

    // Apenas o processo root imprime os resultados
    if(rank == 0){

        std::cout << "Valores coletados: ";

        // Imprime todos os valores recebidos
        for(int i=0;i<size;i++)
            std::cout << resultado[i] << " ";

        std::cout << std::endl;
    }

    // Finaliza o ambiente MPI
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
#SBATCH --output=saida%j.txt
#SBATCH --partition=express
#SBATCH --mem=1GB
#SBATCH --nodes=2
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --export=ALL

# Execute o seu binário com o MPI
mpirun -np $SLURM_NTASKS ./seu_binario

```

### Submeta o job com SLURM:
```bash
sbatch SeuSlurm.slurm
```


## Distribuição de dados com `MPI_Scatter`

Neste exemplo, um vetor grande é criado no **processo 0** e distribuído entre todos os processos.
Cada processo recebe uma parte do vetor e calcula a soma local.

## Ideia

1. `rank 0` cria um vetor grande.
2. O vetor é dividido entre os processos com `MPI_Scatter`.
3. Cada processo calcula uma soma parcial da sua parte.




```cpp
#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    // Descobre o rank do processo e o número total de processos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000; // tamanho total do vetor global

    std::vector<int> dados;

    // Apenas o processo 0 cria o vetor completo
    if(rank == 0){
        dados.resize(N);

        // Preenche o vetor com valores crescentes
        // Isso facilita visualizar os intervalos distribuídos
        for(int i=0;i<N;i++)
            dados[i] = i;
    }

    // Cada processo receberá uma fração do vetor
    int local_size = N / size;

    std::vector<int> local(local_size);

    // Distribui partes do vetor para todos os processos
    MPI_Scatter(dados.data(),
                local_size,
                MPI_INT,
                local.data(),
                local_size,
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    // Calcula soma local
    int soma_local = 0;

    for(int i=0;i<local_size;i++)
        soma_local += local[i];

    // Mostra qual intervalo foi recebido
    std::cout << "Processo " << rank
              << " recebeu intervalo ["
              << local.front() << ", "
              << local.back() << "]"
              << " | soma local = "
              << soma_local
              << std::endl;

    MPI_Finalize();
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
#SBATCH --output=saida%j.txt
#SBATCH --partition=express
#SBATCH --mem=1GB
#SBATCH --nodes=2
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --export=ALL

# Execute o seu binário com o MPI
mpirun -np $SLURM_NTASKS ./seu_binario

```

### Submeta o job com SLURM:
```bash
sbatch SeuSlurm.slurm
```

## Combinação de resultados com `MPI_Reduce`

Agora cada processo possui um valor (ou resultado parcial) e queremos combinar esses valores em um único resultado final.

Aqui usamos `MPI_Reduce` para calcular a soma total das somas locais.

1. Cada processo possui uma soma parcial.
2. `MPI_Reduce` combina todas as somas.
3. O resultado final aparece no **processo 0**.


```cpp
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {

    // Inicializa o ambiente MPI
    MPI_Init(&argc, &argv);

    int rank, size;

    // Descobre o identificador do processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Descobre quantos processos existem
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Cada processo cria um valor local
    int valor_local = rank * 222 + 100;

    // Mostra o valor local antes da redução
    std::cout << "Processo " << rank 
              << " possui valor local = "
              << valor_local << std::endl;

    int soma_total = 0;

    // Operação coletiva: soma todos os valores locais
    MPI_Reduce(&valor_local,
               &soma_total,
               1,
               MPI_INT,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    // Apenas o processo raiz recebe o resultado final
    if(rank == 0){

        std::cout << "\n---- Resultado da redução ----\n";

        std::cout << "Todos os valores foram somados no processo 0\n";

        std::cout << "Soma total = " << soma_total << std::endl;
    }

    MPI_Finalize();
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
#SBATCH --output=saida%j.txt
#SBATCH --partition=express
#SBATCH --mem=1GB
#SBATCH --nodes=2
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --export=ALL

# Execute o seu binário com o MPI
mpirun -np $SLURM_NTASKS ./seu_binario

```

### Submeta o job com SLURM:
```bash
sbatch SeuSlurm.slurm
```
