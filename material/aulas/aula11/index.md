# Comunicação bloqueante e não bloqueante em MPI
Em um programa distribuído, vários processos trabalham juntos para resolver um problema. Eles podem executar no mesmo nó ou em nós diferentes de um cluster. Cada processo possui seu próprio espaço de memória: alterar uma variável em um processo não altera automaticamente a variável de outro.

Imagine que quatro processos calculam partes de um resultado. Para obter a resposta final, precisamos reunir essas partes. O MPI permite fazer isso por meio da troca de mensagens, que também pode ser usada para distribuir dados e coordenar o trabalho.

Na **comunicação ponto a ponto**, uma mensagem é enviada por um processo e recebida por outro. Nesta aula, estudaremos quatro chamadas:

| Forma de comunicação | Envio | Recebimento |
| --- | --- | --- |
| Bloqueante | `MPI_Send` | `MPI_Recv` |
| Não bloqueante | `MPI_Isend` | `MPI_Irecv` |

**Rank identifica um processo, não um nó.** Dois processos no mesmo nó têm ranks diferentes dentro de `MPI_COMM_WORLD`. Por exemplo, quatro processos distribuídos em dois nós continuam sendo quatro ranks.

### Como uma mensagem é descrita?

Podemos pensar na mensagem como uma correspondência: o **conteúdo** são os dados transmitidos; o **envelope** identifica a origem, o destino, a tag e o comunicador.

No envio abaixo, o processo transmite um inteiro armazenado em `valor` para o rank 1:

```cpp
MPI_Send(&valor, 1, MPI_INT, 1, 10, MPI_COMM_WORLD);
```

O recebimento correspondente, executado pelo rank 1, é:

```cpp
MPI_Recv(&recebido, 1, MPI_INT, 0, 10,
         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

### Parâmetros e possibilidades

| Parâmetro | No envio | No recebimento | Exemplos e alternativas |
| --- | --- | --- | --- |
| Buffer | Endereço dos dados a enviar | Endereço onde armazenar os dados | `&valor` para uma variável; `vetor` para um array; `vetor.data()` para um `std::vector` |
| Quantidade | Número de elementos enviados | Capacidade do buffer em elementos | `1`, `5` ou uma variável inteira. Não é a quantidade de bytes, exceto quando cada elemento tem um byte |
| Tipo | Tipo dos elementos enviados | Tipo dos elementos esperados | `MPI_INT`, `MPI_DOUBLE`, `MPI_FLOAT`, `MPI_CHAR` |
| Rank | Destino da mensagem | Origem esperada | Um rank válido no comunicador. No recebimento, `MPI_ANY_SOURCE` aceita qualquer origem |
| Tag | Etiqueta da mensagem enviada | Etiqueta da mensagem esperada | `10`, `20` ou outra tag válida. No recebimento, `MPI_ANY_TAG` aceita qualquer tag |
| Comunicador | Contexto em que ocorre o envio | Contexto em que ocorre o recebimento | `MPI_COMM_WORLD` ou um comunicador criado pelo programa |
| Status | Não é parâmetro de `MPI_Send` | Informações sobre o recebimento | `MPI_STATUS_IGNORE` ou `&status`, após declarar `MPI_Status status;` |

O símbolo `&` fornece o endereço de uma variável. Um array, como `double valores[5]`, pode ser passado usando apenas `valores`, que fornece acesso ao primeiro elemento. Em um `std::vector`, use `.data()` e garanta que o vetor tenha o tamanho necessário.

A tag ajuda a distinguir mensagens, mas **não é o conteúdo transmitido**. A origem é identificada automaticamente pelo MPI no envio. No recebimento, origem, tag e comunicador determinam qual mensagem pode ser aceita; quantidade e tipo descrevem como os dados serão armazenados.

| Tipo em C++ | Tipo MPI correspondente |
| --- | --- |
| `int` | `MPI_INT` |
| `float` | `MPI_FLOAT` |
| `double` | `MPI_DOUBLE` |
| `char` | `MPI_CHAR` |
| `long` | `MPI_LONG` |
| `long long` | `MPI_LONG_LONG_INT` |
| `unsigned int` | `MPI_UNSIGNED` |

O buffer de recebimento deve comportar a mensagem. Ele pode ter capacidade maior que a quantidade enviada, mas uma mensagem maior que sua capacidade causa erro de truncamento. Nos exemplos desta aula, utilize o mesmo tipo MPI no envio e no recebimento. Tags devem ser não negativas e não exceder o limite `MPI_TAG_UB`. Uma quantidade zero produz uma mensagem sem conteúdo, que ainda pode servir como sinal.

### Comunicação bloqueante: receber antes de calcular

Quando o próximo cálculo depende da mensagem, a comunicação bloqueante oferece um fluxo simples: **receber o dado, calcular e mostrar o resultado**.

- `MPI_Send` retorna quando o buffer de envio pode ser alterado ou reutilizado. Isso não garante que o destinatário tenha concluído o recebimento: o MPI pode ter copiado os dados para um buffer interno.
- `MPI_Recv` retorna quando o recebimento terminou e os dados estão disponíveis no buffer.


`bloqueante.cpp`:

```cpp
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Inicializa o MPI em cada processo.
    MPI_Init(&argc, &argv);

    int rank, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);

    // O exemplo exige exatamente dois processos.
    if (total != 2) {
        if (rank == 0)
            std::cerr << "Execute com exatamente 2 processos.\n";
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        int valor = 7;

        // Envia um inteiro ao rank 1, com a tag 10.
        MPI_Send(&valor, 1, MPI_INT, 1, 10, MPI_COMM_WORLD);
    } else {
        int recebido;

        // Espera o valor necessário para realizar o cálculo.
        MPI_Recv(&recebido, 1, MPI_INT, 0, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int quadrado = recebido * recebido;
        std::cout << "Rank 1: quadrado = " << quadrado << '\n';
    }

    MPI_Finalize();
    return 0;
}
```


```bash
mpic++ -O2 bloqueante.cpp -o bloq
```


```bash
srun --partition=merry_cpu --mpi=pmix --mem=1G --nodes=2 \
     --ntasks=2 --ntasks-per-node=1 ./bloq
```


O resultado esperado no rank 1 é `49`. Todos executam o mesmo programa, mas o `if` define a função de cada rank.

### Cuidado com a ordem das chamadas

Bloqueante não significa que todos os processos param juntos. Cada processo aguarda sua própria operação. Entretanto, uma ordem inadequada pode causar **deadlock**, uma situação em que os processos ficam esperando uns pelos outros sem conseguir avançar.

Por exemplo, se dois processos chamarem `MPI_Send` um para o outro antes de chamar `MPI_Recv`, ambos poderão ficar esperando que o outro inicie o recebimento. Mensagens pequenas podem funcionar graças a buffers internos, mas isso não garante que o programa funcionará para mensagens maiores.

Para uma troca de ida e volta, uma ordem segura é: o rank 0 envia e depois recebe; o rank 1 recebe e depois envia. Essa será a organização do desafio ping-pong.

### Comunicação não bloqueante: calcular enquanto a operação está pendente

Agora suponha que o rank 1 também precise somar um vetor local. Essa soma não depende do valor enviado pelo rank 0. Podemos iniciar o recebimento, fazer a soma e só então aguardar a mensagem.

`MPI_Isend` e `MPI_Irecv` iniciam as operações e retornam um **pedido**, do tipo `MPI_Request`. Esse pedido permite acompanhar a operação. O retorno da chamada inicial não garante sua conclusão.

As chamadas usam os mesmos parâmetros básicos de envio e recebimento, com um endereço de pedido ao final. `MPI_Irecv` recebe `&pedido` nessa posição, em vez do status usado em `MPI_Recv`:

```cpp
MPI_Isend(&valor, 1, MPI_INT, 1, 10, MPI_COMM_WORLD, &pedido);
MPI_Irecv(&recebido, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &pedido);
```

Essas duas linhas ilustram as assinaturas; cada operação pendente deve ter seu próprio pedido. Nos exemplos abaixo, elas são executadas por processos diferentes.

`nao_bloqueante.cpp`:

```cpp
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);

    if (total != 2) {
        if (rank == 0)
            std::cerr << "Execute com exatamente 2 processos.\n";
        MPI_Finalize();
        return 1;
    }

    MPI_Request pedido;

    if (rank == 0) {
        int valor = 7;

        // Inicia o envio e guarda seu identificador em "pedido".
        MPI_Isend(&valor, 1, MPI_INT, 1, 10,
                  MPI_COMM_WORLD, &pedido);

        // Mantém "valor" válido até a conclusão do envio.
        MPI_Wait(&pedido, MPI_STATUS_IGNORE);
    } else {
        int recebido;

        // Inicia o recebimento sem exigir que ele termine agora.
        MPI_Irecv(&recebido, 1, MPI_INT, 0, 10,
                  MPI_COMM_WORLD, &pedido);

        // Trabalho independente: não acessa o buffer "recebido".
        int dados[] = {10, 20, 30, 40};
        int soma = 0;
        for (int numero : dados)
            soma += numero;

        // Confirma a conclusão antes de utilizar o dado recebido.
        MPI_Wait(&pedido, MPI_STATUS_IGNORE);

        int quadrado = recebido * recebido;
        std::cout << "Rank 1: soma local = " << soma << '\n';
        std::cout << "Rank 1: quadrado = " << quadrado << '\n';
    }

    MPI_Finalize();
    return 0;
}
```


```bash
mpic++ -O2 nao_bloqueante.cpp -o nobloq
```


```bash
srun --partition=merry_cpu --mpi=pmix --mem=1G --nodes=2 \
     --ntasks=2 --ntasks-per-node=1 ./nobloq
```


O rank 1 imprime soma `100` e quadrado `49`. A oportunidade de sobreposição está entre `MPI_Irecv` e `MPI_Wait`. No rank 0, a espera imediata foi usada para manter o exemplo simples; ali não há trabalho sobreposto.

### Quando podemos usar os buffers?

Enquanto uma operação não bloqueante estiver pendente:

| Buffer | Regra |
| --- | --- |
| Envio | Não altere o conteúdo nem libere a memória |
| Recebimento | Não leia, altere ou libere a memória |


Após a conclusão, o buffer correspondente pode ser utilizado normalmente.

`MPI_Wait` aguarda a conclusão. Outra possibilidade é `MPI_Test`, que retorna uma flag indicando se a operação terminou:

```cpp
int terminou = 0;
MPI_Test(&pedido, &terminou, MPI_STATUS_IGNORE);

if (terminou) {
    // A operação terminou: o buffer pode ser utilizado.
} else {
    // A operação continua pendente: mantenha os cuidados com o buffer.
}
```

### Qual forma escolher?

| Situação | Escolha e motivo |
| --- | --- |
| O próximo cálculo depende imediatamente da mensagem | Bloqueante: organiza um fluxo direto de receber e calcular |
| Existe trabalho independente dos dados transferidos | Não bloqueante: permite tentar sobrepor comunicação e cálculo |
| A chamada não bloqueante é seguida imediatamente por uma espera | Pode ser correta, mas não aproveita esse intervalo para computação |

O vetor pequeno apenas ilustra a organização. Comunicação não bloqueante não é automaticamente mais rápida: o ganho depende da quantidade de trabalho independente e do progresso da comunicação na implementação MPI. Meça antes de concluir que houve melhoria.



### Para entender se você entendeu: coleta centralizada de blocos

Agora cada processo cria um bloco de inteiros. Os nós enviam seus blocos ao rank 0, que monta um vetor com todos os dados. Há `total - 1` mensagens: uma de cada nó para o coordenador, que é o rank 0.

`coleta.cpp`:

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

    // Os valores permitem identificar de qual rank veio cada bloco.
    for (int i = 0; i < bloco; ++i)
        local[i] = rank * 10000 + i;

    if (rank == 0) {
        std::vector<int> reunidos(total * bloco);

        // O bloco do coordenador já está disponível localmente.
        std::copy(local.begin(), local.end(), reunidos.begin());

        for (int origem = 1; origem < total; ++origem) {
            // Cada origem ocupa uma faixa diferente do vetor final.
            MPI_Recv(reunidos.data() + origem * bloco,
                     bloco, MPI_INT, origem, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int p = 0; p < total; ++p) {
            std::cout << "Bloco do rank " << p << ": ";
            for (int i = 0; i < 5; ++i)
                std::cout << reunidos[p * bloco + i] << ' ';
            std::cout << "...\n";
        }
    } else {
        // Cada trabalhador envia seu bloco completo ao rank 0.
        MPI_Send(local.data(), bloco, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
```

O primeiro bloco começa em `0`; o segundo, em `10000`; o terceiro, se houver, em `20000`. O deslocamento `origem * bloco` determina onde guardar cada bloco. O recebimento é feito por ordem de rank, mesmo que outro trabalhador já esteja pronto para enviar.

**Perguntas para discutir:**

1. Por que o rank 0 não precisa enviar uma mensagem para si mesmo?
2. Por que apenas ele precisa do vetor `reunidos`?
3. O que acontece se o rank 1 demorar, mas o rank 2 já estiver pronto?
4. Como poderíamos iniciar todos os recebimentos com `MPI_Irecv`? Considere um pedido por nó, faixas separadas no vetor e a conclusão de todos os pedidos antes de imprimir.
