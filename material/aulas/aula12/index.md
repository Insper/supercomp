# Comunicação coletiva MPI: Broadcast e Reduce
Em um programa distribuído, muitas tarefas contam com a participação de vários processos ao mesmo tempo. Antes de iniciar um cálculo, por exemplo, todos podem precisar conhecer uma configuração comum. Ao terminar, pode ser necessário combinar os resultados parciais para obter uma resposta final. A comunicação coletiva permite realizar essas trocas como operações de grupo, facilitando a organização do trabalho entre os processos.

Vamos conhecer dois padrões. No um para muitos, um processo compartilha uma informação com os demais. No muitos para um, cada processo contribui com seu resultado local, e um processo escolhido recebe o resultado combinado.

No MPI, usamos **`MPI_Bcast`** para compartilhar os mesmos dados com todos e **`MPI_Reduce`** para combinar contribuições. O processo que fornece os dados no broadcast ou recebe o resultado no reduce é chamado de **raiz**.

Mesmo que os papéis sejam diferentes, todos os processos do comunicador precisam chamar a operação, inclusive a raiz. Assim, cada um participa da troca e o grupo consegue realizar a tarefa.


### Como os processos participam?

No **broadcast**, o processo raiz compartilha os mesmos dados com todos os demais processos. Na **redução**, cada processo fornece uma contribuição, inclusive a raiz. Essas contribuições são combinadas por uma operação, como soma ou máximo, e somente a raiz recebe o resultado final.

As funções **`MPI_Bcast` e `MPI_Reduce` são bloqueantes**: o processo que faz a chamada permanece dentro dela até que sua participação na operação esteja concluída. Só depois do retorno ele executa a próxima instrução.

No **`MPI_Bcast`**, isso significa que, ao retornar, o processo já pode utilizar os dados recebidos para calcular seu intervalo de trabalho.

No **`MPI_Reduce`**, após o retorno, cada processo pode reutilizar seu buffer de contribuição. Para o processos raiz, o retorno também garante que o resultado combinado está disponível para uso.

Esse bloqueio é local: um processo pode retornar enquanto outro ainda está dentro da chamada. Portanto, ser bloqueante não significa que todos terminam ao mesmo tempo, e essas funções não devem ser tratadas como uma barreira de sincronização.

### Broadcast: compartilhar uma informação com todos


No broadcast, todos chamam a mesma função, mas o buffer tem um papel diferente na raiz e nos demais processos.

| Parâmetro | Na raiz | Nos demais processos | Exemplos e alternativas |
| --- | --- | --- | --- |
| Buffer | Endereço dos dados que serão compartilhados | Endereço onde os dados serão armazenados | `&valor` para uma variável; `vetor` para um array; `vetor.data()` para um `std::vector` |
| Quantidade | Número de elementos compartilhados | Número de elementos que serão recebidos | `1`, `5` ou uma variável inteira. Conta elementos, e não necessariamente bytes |
| Tipo | Tipo dos elementos compartilhados | Tipo dos elementos recebidos | `MPI_INT`, `MPI_DOUBLE`, `MPI_FLOAT`, `MPI_CHAR` |
| Raiz (`root`) | Rank do próprio processo que fornece os dados | Rank do processo que fornece os dados | `0`, `1` ou outro rank válido no comunicador. Todos devem indicar a mesma raiz |
| Comunicador | Grupo participante da operação | O mesmo grupo participante | `MPI_COMM_WORLD` ou outro comunicador criado pelo programa |

### Como passar o buffer?

O símbolo **`&`** fornece o endereço de uma variável:

```cpp
MPI_Bcast(&valor, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

Em um `std::vector`, usamos `.data()`. Todos os processos precisam ter espaço para os elementos:

```cpp
std::vector<int> valores(5);

// A raiz deve preencher o vetor antes da chamada.
MPI_Bcast(valores.data(), 5, MPI_INT, 0, MPI_COMM_WORLD);
```

Os endereços dos buffers podem ser diferentes entre os processos, pois cada um possui sua própria memória. O que precisa corresponder é a descrição dos dados

### Tipos de dados mais usados

| Tipo em C++ | Tipo MPI correspondente |
| --- | --- |
| `int` | `MPI_INT` |
| `float` | `MPI_FLOAT` |
| `double` | `MPI_DOUBLE` |
| `char` | `MPI_CHAR` |
| `long` | `MPI_LONG` |
| `long long` | `MPI_LONG_LONG_INT` |
| `unsigned int` | `MPI_UNSIGNED` |


**Exemplo completo: uma configuração comum**

Salve como `broadcast.cpp`:

```cpp
#include <mpi.h>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total, tamanho;
    char nome[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Get_processor_name(nome, &tamanho);

    int quantidade = 0;

    // Somente a raiz define inicialmente a configuração.
    if (rank == 0)
        quantidade = 100;

    // Todos participam, incluindo o rank 0.
    MPI_Bcast(&quantidade, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cada processo mostra sua cópia do valor compartilhado.
    std::ostringstream mensagem;
    mensagem << "Rank " << rank << " | Nó: " << nome
             << " | Quantidade recebida: " << quantidade << '\n';
    std::cout << mensagem.str() << std::flush;

    MPI_Finalize();
    return 0;
}
```

Compile e execute:

```bash
mpic++  -O2 broadcast.cpp -o broadcast

srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     --time=00:05:00 ./broadcast
```

Os quatro ranks devem imprimir `100`. A ordem dos prints pode variar. 

### Reduce: transformar contribuições em um resultado

Depois de dividir uma computação, cada processo pode ter um resultado parcial. `MPI_Reduce` combina essas contribuições usando uma operação e coloca o resultado final no buffer do processo raiz.

Por exemplo, se quatro processos contribuírem com `1`, `2`, `3` e `4`, uma redução com `MPI_SUM` produz `10`.

```cpp
int local = rank + 1;
int resultado = 0;

// Todos contribuem com seu valor local.
MPI_Reduce(&local, &resultado, 1, MPI_INT,
           MPI_SUM, 0, MPI_COMM_WORLD);

// Apenas a raiz recebe o resultado global.
if (rank == 0)
    std::cout << "Soma = " << resultado << '\n';
```


```cpp
MPI_Reduce(
    &local,          // Endereço da contribuição deste processo.
    &resultado,      // Endereço do resultado, significativo na raiz.
    1,               // Quantidade de elementos por processo.
    MPI_INT,         // Tipo dos elementos.
    MPI_SUM,         // Operação que combina as contribuições.
    0,               // Rank da raiz que recebe o resultado.
    MPI_COMM_WORLD   // Comunicador participante.
);
```

| Parâmetro | Significado | Exemplo |
| --- | --- | --- |
| Buffer de contribuição | Dados fornecidos por este processo | `&local` |
| Buffer de resultado | Local onde a raiz recebe o resultado global | `&resultado` |
| Quantidade | Número de elementos contribuídos por processo | `1` |
| Tipo | Tipo dos elementos | `MPI_INT` |
| Operação | Como combinar as contribuições | `MPI_SUM` |
| Raiz | Rank que recebe o resultado global | `0` |
| Comunicador | Grupo participante | `MPI_COMM_WORLD` |

No exemplo, cada processo declara `resultado` para simplificar o código. Contudo, o buffer de resultado só é significativo na raiz: os demais processos não recebem o resultado global dessa chamada.

A raiz também contribui com seu valor de `local`.

Na coleta da aula anterior, o rank 0 guardava os blocos recebidos em faixas diferentes de um vetor. Uma redução tem outro objetivo: **combinar contribuições**.

Se cada processo contribuir com um vetor, a operação combina posições correspondentes. Por exemplo:

| Processo | Posição 0 | Posição 1 |
| --- | ---: | ---: |
| Rank 0 | 1 | 10 |
| Rank 1 | 2 | 20 |
| Resultado de `MPI_SUM` na raiz | 3 | 30 |

Nesse caso, `count = 2`. O resultado tem dois elementos. Para obter um único total dos elementos de todos os vetores, cada processo pode primeiro somar seu vetor localmente e depois reduzir essa soma usando `count = 1`.

**Exemplo completo: somar uma contribuição por rank**

Salve como `reduce.cpp`:

```cpp
#include <mpi.h>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total, tamanho;
    char nome[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Get_processor_name(nome, &tamanho);

    int local = rank + 1;
    int resultado = 0;

    std::ostringstream mensagem;
    mensagem << "Rank " << rank << " | Nó: " << nome
             << " | Contribuição: " << local << '\n';
    std::cout << mensagem.str() << std::flush;

    // Cada rank contribui. O resultado global fica no rank 0.
    MPI_Reduce(&local, &resultado, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        std::cout << "Soma das contribuições = " << resultado << '\n';

    MPI_Finalize();
    return 0;
}
```

```bash
mpic++ -std=c++11 -O2 reduce.cpp -o reduce

srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=4 --ntasks=4 --ntasks-per-node=2 \
     --time=00:05:00 ./reduce
```

Com quatro processos, o resultado é `10`, este exemplo muda as contribuições quando muda a quantidade de ranks. Na aplicação seguinte, manteremos o mesmo problema e alteraremos apenas sua distribuição.

### Tudo junto: Broadcast, computação local e Reduce

Vamos calcular a soma dos quadrados dos números de 1 a `N`.

O rank 0 define `N` e compartilha esse valor. Cada processo calcula os limites de seu intervalo, executa a computação e contribui com uma soma parcial. Ao final, o rank 0 recebe a soma global.

| Etapa | Operação | Quem participa? |
| --- | --- | --- |
| Compartilhar `N` | `MPI_Bcast` | Todos os processos |
| Determinar o intervalo e calcular | Computação local | Todos, incluindo a raiz |
| Combinar as somas parciais | `MPI_Reduce` com `MPI_SUM` | Todos os processos |
| Mostrar o resultado global | Impressão | Apenas o raiz |

Aqui o rank 0 também calcula uma parte. **Ser a raiz de uma comunicação coletiva não impede o processo de participar da computação.**



Salve como `coletivas.cpp`:

```cpp
#include <mpi.h>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total, tamanho;
    char nome[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Get_processor_name(nome, &tamanho);

    int N = 0;

    // A raiz define o tamanho do problema.
    if (rank == 0)
        N = 100;

    // Todos recebem o mesmo tamanho de problema.
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cada processo determina sua própria parcela do trabalho.
    long long inicio = (1LL * rank * N) / total + 1;
    long long fim = (1LL * (rank + 1) * N) / total;

    // Calcula a soma dos quadrados apenas do intervalo local.
    long long parcial = 0;
    for (long long numero = inicio; numero <= fim; ++numero)
        parcial += numero * numero;

    // Identifica o processo, o nó e a tarefa executada.
    std::ostringstream mensagem;
    mensagem << "Rank " << rank << " | Nó: " << nome;

    if (inicio <= fim)
        mensagem << " | Intervalo: " << inicio << " a " << fim;
    else
        mensagem << " | Intervalo vazio";

    mensagem << " | Soma parcial: " << parcial << '\n';
    std::cout << mensagem.str() << std::flush;

    // Combina todas as somas parciais no rank 0.
    long long resultado = 0;
    MPI_Reduce(&parcial, &resultado, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    // Apenas a raiz utiliza o resultado global.
    if (rank == 0)
        std::cout << "Soma dos quadrados de 1 a " << N
                  << " = " << resultado << '\n';

    MPI_Finalize();
    return 0;
}
```

Observe que apenas a definição inicial de `N` e a impressão final ficam dentro de `if (rank == 0)`. As chamadas de broadcast e redução são executadas por todos.

O broadcast compartilha `N`, mas não envia intervalos individuais. A distribuição das tarefas resulta do cálculo dos limites usando `rank`, `N` e `total`.

Compilação e execução

```bash
mpic++ -O2 coletivas.cpp -o coletivas


srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     --time=00:05:00 ./coletivas
```

Repita com dois processos:

```bash
srun --partition=merry_cpu --mpi=pmix --mem=1G \
     --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --time=00:05:00 ./coletivas
```



## Para entender se você entendeu


Implemente um programa que calcule a soma dos quadrados de 1 a `N`.

- O rank 0 deve definir `N` e compartilhar os dados com `MPI_Bcast`.
- Cada processo deve calcular uma parcela, incluindo o rank 0.
- Cada processo deve imprimir rank, nó, intervalo e soma parcial.
- Use `MPI_Reduce` para obter o resultado no rank 0.
- Execute com dois e quatro processos, para `N = 3000` e `N = 6006`.



??? info "gabarito"
    ```cpp
    #include <mpi.h>
    #include <iostream>
    #include <vector>
    #include <sstream>
    #include <iomanip>

    using namespace std;

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);

        int rank, total, tamanho;
        char nome[MPI_MAX_PROCESSOR_NAME];

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &total);
        MPI_Get_processor_name(nome, &tamanho);

        int N = 0;
        vector<int> dados;

        if (rank == 0) {
            N = 6006;
            dados.resize(N);

            // Prepara os números de 1 a N.
            for (int i = 0; i < N; i++) {
                dados[i] = i + 1;
            }
        }

        // Primeiro, todos recebem o tamanho do vetor.
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Os demais processos reservam espaço para os dados.
        if (rank != 0) {
            dados.resize(N);
        }

        // Compartilha o vetor completo com todos.
        MPI_Bcast(dados.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

        // Define quantos elementos cada processo irá calcular.
        int por_processo = N / total;
        int sobra = N % total;

        int quantidade_local = por_processo;

        if (rank < sobra) {
            quantidade_local++;
        }

        // Calcula a posição inicial da fatia no vetor.
        int inicio = rank * por_processo;

        if (rank < sobra) {
            inicio += rank;
        } else {
            inicio += sobra;
        }

        // "fim" é a primeira posição depois da fatia.
        int fim = inicio + quantidade_local;

        // Calcula somente sobre os elementos da fatia.
        double parcial = 0.0;

        for (int i = inicio; i < fim; i++) {
            double valor = dados[i];
            parcial += valor * valor;
        }

        // Identifica a parcela calculada por cada processo.
        ostringstream mensagem;
        mensagem << fixed << setprecision(0);
        mensagem << "Rank " << rank << " | Nó: " << nome;

        if (quantidade_local > 0) {
            mensagem << " | Valores: " << dados[inicio]
                    << " a " << dados[fim - 1];
        } else {
            mensagem << " | Fatia vazia";
        }

        mensagem << " | Soma parcial: " << parcial << '\n';
        cout << mensagem.str() << flush;

        // Reúne as somas parciais no rank 0.
        double resultado = 0.0;

        MPI_Reduce(&parcial, &resultado, 1, MPI_DOUBLE,
                MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << fixed << setprecision(0)
                << "Soma total = " << resultado << endl;
        }

        MPI_Finalize();
        return 0;
    }


    ```


### Alterações para explorar

1. Faça as alterações necessárias para calcular o **maior quadrado** dos dados.
2. Faça as alterações necessárias para calcular a **média dos quadrados**
3. Faça as alterações necessárias para usar o rank 1 como raiz na computação.


### Perguntas para discutir

1. Por que a raiz também precisa chamar as coletivas?
2. Broadcast fornece o mesmo dado ou parcelas diferentes a cada processo?
3. Quem recebe o resultado global de `MPI_Reduce`?
4. Qual é a diferença entre guardar todos os blocos e reduzir somas parciais?
5. Por que mudar o número de processos não muda a soma final quando `N` é mantido?
6. O que acontece se um processo chamar Reduce enquanto os demais chamam Bcast na mesma posição da sequência?

