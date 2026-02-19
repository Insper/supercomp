
## Objetivo
Explorar técnicas de otimização de código sequencial em C++ a partir da análise de desempenho.
O foco será:

* Compreender a relação entre **hierarquia de memória (L1, L2, L3)** e desempenho.
* Aplicar **tiling (fateamento em blocos)** para melhorar o aproveitamento da memória cache.
* Usar a classe vector do C++ para manipular os dados


## Vector em C++ (std::vector)

O `vector` é uma classe da biblioteca padrão do C++ (a famosa STL) e, sinceramente, é uma das estruturas mais úteis para a nossa vida. O grande diferencial do `vector`, além da variedade de métodos que já vêm prontos para facilitar a manipulação e as operações nos dados, é que ele oferece **redimensionamento dinâmico** e **gerenciamento eficiente de memória**.

Na prática, isso significa que você não precisa se preocupar com detalhes como alocação, realocação ou liberação de memória, o próprio `vector` cuida disso pra gente. Claro, se você quiser muito ter total controle sobre tudo, você é livre, inclusive, usando vector, fica até mais facil realizar essas operações. Mas, na maioria dos casos, dá para simplesmente usar os métodos e não se preocupar com isso.


## O que é um Vector em C++?

Um **vector em C++** é um **array dinâmico** que se redimensiona automaticamente quando elementos são adicionados ou removidos. Diferente dos arrays tradicionais (de tamanho fixo), o `std::vector` pode crescer ou diminuir em tempo de execução.

O armazenamento interno é gerenciado automaticamente pelo próprio contêiner.

Os elementos de um vector são armazenados em **posições contíguas de memória**, permitindo:

* Acesso eficiente via operador de índice `[]`
* Uso de iteradores
* Passagem de ponteiro para funções que esperam arrays

Você pode **pré-alocar** um espaço específico para os seus dados e ir preenchendo ao longo do código, se já souber aproximadamente quantos elementos vai utilizar. Mas também pode deixar o contêiner se virar sozinho: sempre que precisar de mais espaço, o `vector` se redimensiona automaticamente, realocando memória de forma eficiente para acomodar os novos dados.

Embora utilize um pouco mais de memória do que um array de tamanho fixo, o `std::vector` costuma ser mais eficiente no acesso aos elementos quando comparado a outros contêineres sequenciais, como o `std::deque` e o `std::list`, principalmente porque seus elementos ficam armazenados de forma contígua na memória.


#  Como declarar um `vector` em C++?

Primeiro, você precisa incluir a biblioteca:

```cpp
#include <vector>
```

Depois disso, pode declarar um vector assim:

```cpp
std::vector<int> v;
```

Isso cria um vector de números inteiros vazio.

* `vector` → é o tipo da estrutura.
* `<int>` → é o tipo de dado que será armazenado.
* `std::` → indica que o vector pertence à biblioteca std.

Se você usar:

```cpp
using namespace std;
```

Pode escrever apenas:

```cpp
vector<int> v;
```

Você já pode criar o vector com valores iniciais:

```cpp
std::vector<int> v = {1, 2, 3, 4};
```

ou

```cpp
std::vector<int> v{1, 2, 3, 4};
```

O vector já nasce com esses quatro elementos.


Você também pode criar um vector com tamanho definido:

```cpp
std::vector<int> v(3, 4);
```

Isso significa:

* O vector terá **3 posições**
* Todas começam com o valor **4**

Equivalente a:

```
{4, 4, 4}
```

Você pode:

* Criar vazio → `vector<int> v;`
* Criar com valores → `{1,2,3}`
* Criar com tamanho fixo inicial → `(quantidade, valor)`

### Alocação dos dados na memória

Quando usamos pré-alocação em um `std::vector`, como no caso do `reserve()`, a principal vantagem está em evitar realocações desnecessárias de memória durante a execução do programa.

Por padrão, quando você vai inserindo elementos com `push_back()`, o vector começa com uma capacidade pequena. Quando essa capacidade é atingida, ele precisa alocar um novo bloco de memória maior, copiar todos os elementos antigos para esse novo espaço e depois liberar a memória anterior. Esse processo pode acontecer várias vezes e tem custo computacional.

Ao usar `dados.reserve(quantidade);`, você já informa ao programa quantos elementos pretende usar. Assim, o espaço é reservado uma única vez, e os `push_back()` seguintes apenas colocam os valores nas posições já disponíveis, sem precisar realocar memória. Isso melhora o desempenho porque reduz cópias, reduz chamadas ao alocador de memória e evita interrupções frequentes no fluxo do programa.


Vamos observar isto acontecendo na prática:

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> dados;

    dados.reserve(10000);  // pré-aloca espaço para 10.000 elementos

    for (int i = 0; i < 333; ++i) { // preenche de 0 a 333
        dados.push_back(i);         // aloca os dados sempre ao final
    }

    std::cout << "Tamanho: " << dados.size() << std::endl;      // não usei o namespace, por isso,
                                                                // tenho que deixar explicito o sdt 
                                                                // quando for usar as classes da biblioteca
    std::cout << "Capacidade: " << dados.capacity() << std::endl;

    return 0;
}

```
Saída esperada:

```bash
Tamanho: 333
Capacidade: 10000
```


Neste código, não vamos pré-alocar os dados, vamos observar o que acontece:

```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {

    vector<int> dados;

    cout << "=== ESTADO INICIAL ===" << endl;
    cout << "Tamanho: " << dados.size() << endl;
    cout << "Capacidade: " << dados.capacity() << endl;

    size_t capacidade_anterior = dados.capacity();

    cout << "\n=== INSERINDO ELEMENTOS ===\n" << endl;
    int dado = 1;
    for (int i = 0; i < 23; ++i) {
        dado = i * 5; 
        cout << "Inserindo dado: " << dado << endl;
        dados.push_back(dado);

        if (dados.capacity() != capacidade_anterior) {

            cout << ">> Capacidade mudou para: "
                 << dados.capacity() << endl << endl;

            capacidade_anterior = dados.capacity();
        }
    }

    cout << "\n=== ESTADO FINAL ===" << endl;
    cout << "Tamanho final: " << dados.size() << endl;
    cout << "Capacidade final: " << dados.capacity() << endl;

    cout << "\n=== ALOCACAO DOS DADOS ===" << endl;

    cout << "| ";
    for (size_t i = 0; i < dados.size(); ++i) {
        cout << dados[i] << " | ";
    }
    cout << endl;

    return 0;
}

```

Quando trabalhamos com `std::vector`, é importante entender a diferença entre **tamanho (`size`)** e **capacidade (`capacity`)**. O tamanho representa quantos elementos estão realmente armazenados no vector, enquanto a capacidade indica quanto espaço já foi reservado na memória.

No primeiro exemplo, fazemos:

```cpp
std::vector<int> dados;
dados.reserve(10000);
```

Aqui o vector começa vazio, mas ao chamar `reserve(10000)` estamos dizendo ao programa para já separar espaço suficiente para armazenar até 10.000 elementos. Depois disso, o laço insere apenas 333 valores com `push_back()`.

Por isso a saída é:

```
Tamanho: 333
Capacidade: 10000
```

O tamanho é 333 porque só inserimos 333 elementos. A capacidade é 10000 porque reservamos esse espaço antecipadamente. Como havia memória suficiente disponível desde o início, o vector não precisou realocar memória durante as inserções. Isso evita cópias internas de dados, reduz chamadas ao alocador de memória e melhora o desempenho.

Já no segundo exemplo, não usamos `reserve()`. O código começa com:

```cpp
vector<int> dados;
```

Agora o vector cresce dinamicamente conforme os elementos são inseridos. Sempre que a capacidade atual se esgota, o vector precisa:

1. Alocar um novo bloco de memória maior.
2. Copiar todos os elementos antigos para esse novo bloco.
3. Liberar o bloco anterior.

O trecho:

```cpp
if (dados.capacity() != capacidade_anterior)
```

mostra exatamente quando essa realocação acontece. Ao rodar o programa, você verá mensagens indicando que a capacidade mudou várias vezes. Isso acontece porque o vector geralmente dobra sua capacidade quando precisa crescer.

Essa diferença é justamente onde entra a vantagem da pré-alocação. Quando você sabe aproximadamente quantos elementos irá inserir, usar `reserve()` evita essas múltiplas realocações. Isso melhora o desempenho porque reduz cópias de memória e torna a execução mais estável.

Além disso, como o `vector` armazena seus elementos em posições contíguas de memória, ele favorece a localidade espacial, permitindo melhor aproveitamento do cache do processador. Ao evitar realocações frequentes, também ajudamos a manter esse padrão de acesso eficiente.

No segundo código aparece também:

```cpp
using namespace std;
```

Isso apenas evita que precisemos escrever `std::vector`, `std::cout` e `std::endl`. Não muda o funcionamento do programa, apenas simplifica a escrita.

O primeiro exemplo mostra como a pré-alocação garante capacidade suficiente desde o início, evitando realocações. O segundo exemplo demonstra o comportamento padrão de crescimento dinâmico do vector, evidenciando quando e como a capacidade aumenta automaticamente.



##  Memórias cache

Vamos tomar como base o hardware do monstrão, ele tem um processador **Intel Xeon Gold 5215**, que possui:

* **L1d cache**: 32 KiB por núcleo
* **L2 cache**: 1 MiB por núcleo
* **L3 cache**: 13.75 MiB por socket



Na multiplicação de matrizes, o maior gargalo costuma se o acesso a memória. Para otimizar o desempenho de um algoritmo como esse,  dividimos a matriz em blocos que cabem na memória cache, porque ela é a que está mais proxima da CPU. No nosso caso, cada submatriz de tamanho `B×B` precisa caber na cache junto com mais dois blocos (A, B e C). A fórmula para calcular o tamanho máximo do bloco é:

$$
B \leq \sqrt{\frac{\text{Capacidade da Cache}}{24}}
$$

(onde 24 = 3 matrizes × 8 bytes por double).

Analise o código `matmul_seq.cpp`:

```cpp 
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <algorithm>


#define TAM_MATRIZ 1000
/*
 ============================================================
   OBJETIVO
   -----------------------------------------------------------
   Este programa faz a multiplicação de matrizes aninhadas
   de forma SEQUENCIAL e mede o tempo de execução.

   Ele pode rodar em dois modos:
   - Versão INGENUA (sem otimizações)
   - Versão com TILING (fateamento em blocos), onde o tamanho
     do bloco B é passado como parâmetro na linha de comando.

   O objetivo é observar como o tamanho do bloco B influencia:
   - O tempo de execução
   - O uso de cache

 ============================================================
*/

/* Definicoes para melhorar a legibilidade*/

using Matriz = std::vector<std::vector<double>>;

inline Matriz criaMatriz(int size, double value){
    return Matriz(size, std::vector<double>(size, value));
}

/**
 * @brief Versão ingênua da multiplicação de matrizes.
 * 
 * Implementa a multiplicação com três loops aninhados (i, j, k) sem uso de tiling.
 * O acesso às matrizes é feito de forma direta, sem otimizações de cache.
 */
inline void versaoIngenua(){

    // Cria três matrizes NxN em memória, preenchidas com valores fixos
    // - A inicializada com 1.0
    // - Bmat inicializada com 2.0
    // - C inicializada com 0.0 (resultado)

    Matriz A    = criaMatriz(TAM_MATRIZ, 1.0);
    Matriz Bmat = criaMatriz(TAM_MATRIZ, 2.0);
    Matriz C    = criaMatriz(TAM_MATRIZ, 0.0);

    for (int i = 0; i < TAM_MATRIZ; i++) {
        for (int j = 0; j < TAM_MATRIZ; j++) {
            for (int k = 0; k < TAM_MATRIZ; k++) {
                C[i][j] += A[i][k] * Bmat[k][j];
            }
        }
    }
}

/**
 * @brief Multiplicação de matrizes utilizando a técnica de tiling (blocking).
 * 
 * Realiza a multiplicação de matrizes dividindo as matrizes em blocos (tiles) de tamanho `tamBloco`.
 * Otimiza o uso da cache ao trabalhar com submatrizes menores que cabem na hierarquia de memória.
 * 
 * @param tamBloco Tamanho do bloco (tile) usado para dividir as matrizes na multiplicação.
 */
inline void versaoTiling(int tamBloco){

    // Cria três matrizes NxN em memória, preenchidas com valores fixos
    // - A inicializada com 1.0
    // - Bmat inicializada com 2.0
    // - C inicializada com 0.0 (resultado)

    Matriz A    = criaMatriz(TAM_MATRIZ, 1.0);
    Matriz Bmat = criaMatriz(TAM_MATRIZ, 2.0);
    Matriz C    = criaMatriz(TAM_MATRIZ, 0.0);

    for (int ii = 0; ii < TAM_MATRIZ; ii += tamBloco) {        // blocos de linhas
        for (int jj = 0; jj < TAM_MATRIZ; jj += tamBloco) {    // blocos de colunas
            for (int kk = 0; kk < TAM_MATRIZ; kk += tamBloco) {// blocos intermediários
                // Multiplicação de submatrizes tamBloco x tamBloco
                // Ordem j -> i -> k
                for (int j = jj; j < std::min(jj + tamBloco, TAM_MATRIZ); j++) {
                    for (int i = ii; i < std::min(ii + tamBloco, TAM_MATRIZ); i++) {
                        double sum = C[i][j];
                        for (int k = kk; k < std::min(kk + tamBloco, TAM_MATRIZ); k++) {
                            sum += A[i][k] * Bmat[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}



int main(int argc, char* argv[]) {
    int tamBloco = 0; // Tamanho do bloco. Se for 0 → versão ingênua.

    // Lê o tamanho do bloco da linha de comando
    // Exemplo: ./matmul_seq 200  → roda com blocos 200×200
    if (argc > 1) {
        // Atualiza o valor de tamBloco de acordo com o parâmetro de entrada
        tamBloco = std::atoi(argv[1]);
    }

    // Marca o início da medição de tempo
    auto start = std::chrono::high_resolution_clock::now();

    if (tamBloco <= 0) {
        versaoIngenua();
    } 
    else {
        versaoTiling(tamBloco);
    }

    // Marca o fim da medição
    auto end = std::chrono::high_resolution_clock::now();

    // Calcula e imprime o tempo total em milissegundos
    std::cout << "Execução ("
              << (tamBloco <= 0 ? "ingênua" : "tiling tamBloco=" + std::to_string(tamBloco))
              << "): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    return 0;
}
```
## Missões:

### 1. Compilação

Compile o código no terminal do head-node `matmul_seq.cpp`:

```bash
g++ -O2  matmul_seq.cpp -o matmul_seq
```

### 2. Execução

Crie o lançador do SLURM como em `tiling.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=monstrao_tiling
#SBATCH --output=monstrao_tiling%j.out
#SBATCH --error=monstrao_tiling%j.err
#SBATCH --partition=monstrao
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=2G

echo "=============== FILA MONSTRAO=============="

echo "=== Execução versão ingênua ==="
time ./matmul_seq 0

echo "=== Execução com blocos L1 (~36x36) ==="
time ./matmul_seq 36

echo "=== Execução com blocos L2 (~200x200) ==="
time ./matmul_seq 200

echo "=== Execução com blocos L3 (~768x768) ==="
time ./matmul_seq 768
```

Execute com:

```
sbatch tiling.slurm
```


## Explorando Ordenação de Loops e Flags de Otimização em Diferentes Filas

Você já visualizou o efeito do **tiling**. Agora, o objetivo é entender como **a organização dos loops** e **as otimizações do compilador** influenciam o desempenho do mesmo código.


### 1. Alterando a Ordem dos Loops

Modifique o código `matmul_seq.cpp` para usar a ordem **i → k → j** no lugar da ordem original **j → i → k**.
Essa mudança melhora a localidade espacial dos acessos à matriz B, e também beneficia os acessos às matrizes A e C.


### 2. Testar Diferentes Flags de Otimização

Encontre a flag de Otimização com o melhor resultado para esse algoritmo (O2, O3, Ofast)

### 3. Rodar em Diferentes Filas do Cluster

Após identificar as melhores combinações de loop e flags de otimização no **monstrao**, identifique quais são os tamanhos das memórias L1, L2 e L3 na fila GPU e repita os testes.


### Perguntas para entender se você entendeu:

1. A troca de ordem dos loops melhorou ou piorou o tempo de execução? Por quê?
2. Houveram diferenças entre os nós **monstrao** e **gpu**? Quais?
3. Qual o **tamanho de bloco** que apresentou o melhor equilíbrio entre tempo de execução e aproveitamento de cache em cada fila?

## **Esta atividade não tem entrega, Bom final de semana!**

Se quiser estudar um pouco mais sobre vector, veja as [indicações aqui](../../teoria/aula03/index.md) 
