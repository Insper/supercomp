# Classes e objetos

Uma classe em C++ é uma estrutura que define um conjunto de atributos (dados) e métodos (funções) que operam sobre esses dados. É uma forma de agrupar dados e comportamentos relacionados, facilitando a modularidade e a reutilização do código. Objetos são instâncias de classes. Eles representam entidades específicas que possuem os atributos e métodos definidos pela classe.

### Exemplo Prático: Problema da Mochila

No exemplo do problema da mochila, criamos uma classe `Knapsack` para encapsular a lógica do problema. Vamos detalhar cada parte do código para entender como classes e objetos são utilizados.

### Declaração da Classe

A classe `Knapsack` é definida com atributos e métodos necessários para resolver o problema da mochila.

```cpp
#include <iostream>

// Declaração da classe Knapsack
class Knapsack {
public:
    Knapsack(int capacidade, int numItens);       // Construtor que inicializa a mochila
    ~Knapsack();                                  // Destrutor que libera a memória alocada
    void adicionaItem(int peso, int valor);       // Método para adicionar um item
    int resolve();                                // Método para resolver o problema da mochila
    void imprimeItens();                          // Método para imprimir os itens adicionados

private:
    int capacidade;                               // Capacidade máxima da mochila
    int numItens;                                 // Número total de itens
    int* pesos;                                   // Vetor dinâmico para armazenar os pesos dos itens
    int* valores;                                 // Vetor dinâmico para armazenar os valores dos itens
    int contadorItens;                            // Contador de itens adicionados
};

```

- **Atributos:**
    - `capacidade`: Capacidade máxima da mochila.
    - `numItens`: Número total de itens.
    - `pesos`: Ponteiro para um vetor que armazena os pesos dos itens.
    - `valores`: Ponteiro para um vetor que armazena os valores dos itens.
    - `contadorItens`: Contador para acompanhar quantos itens foram adicionados.
- **Métodos:**
    - `Knapsack(int capacidade, int numItens)`: Construtor que inicializa os atributos e aloca memória para os vetores.
    - `~Knapsack()`: Destrutor que libera a memória alocada para os vetores.
    - `void adicionaItem(int peso, int valor)`: Método para adicionar um item à mochila.
    - `int resolve()`: Método para resolver o problema da mochila usando programação dinâmica.
    - `void imprimeItens()`: Método para imprimir os itens adicionados à mochila.

### Implementação do Construtor e Destrutor

O construtor inicializa os atributos e aloca memória para os vetores de pesos e valores. O destrutor libera essa memória.

```cpp
// Implementação do construtor
Knapsack::Knapsack(int capacidade, int numItens)
    : capacidade(capacidade), numItens(numItens), contadorItens(0) {
    pesos = new int[numItens];                    // Aloca memória para os pesos dos itens
    valores = new int[numItens];                  // Aloca memória para os valores dos itens
}

// Implementação do destrutor
Knapsack::~Knapsack() {
    delete[] pesos;                               // Libera a memória alocada para os pesos
    delete[] valores;                             // Libera a memória alocada para os valores
}

```

### Método para Adicionar Itens

O método `adicionaItem` permite adicionar itens à mochila, atualizando os vetores de pesos e valores.

```cpp
// Implementação do método para adicionar um item
void Knapsack::adicionaItem(int peso, int valor) {
    if (contadorItens < numItens) {
        pesos[contadorItens] = peso;              // Adiciona o peso do item
        valores[contadorItens] = valor;           // Adiciona o valor do item
        contadorItens++;                          // Incrementa o contador de itens
    } else {
        std::cerr << "Número máximo de itens excedido!" << std::endl;
    }
}

```

### Método para Resolver o Problema da Mochila

Programação Dinâmica (Dynamic Programming, DP) é uma técnica de otimização usada para resolver problemas complexos dividindo-os em subproblemas mais simples. Ela é especialmente eficaz para problemas que podem ser divididos em subproblemas menores, onde os resultados de subproblemas anteriores podem ser reutilizados para resolver subproblemas maiores.

No problema da mochila, a programação dinâmica é usada para encontrar a combinação de itens que maximiza o valor total sem exceder a capacidade da mochila. Construindo uma tabela que armazena os valores máximos possíveis para diferentes capacidades e conjuntos de itens.

A tabela de DP (K) é construída de forma que cada entrada K[i][w] representa o valor máximo que pode ser obtido usando os primeiros i itens com uma capacidade máxima de w.

### Método resolve

1. **Inicialização da Tabela de DP:**
    
    ```cpp
    int** K = new int*[numItens + 1];             // Aloca memória para a tabela de DP
    for (int i = 0; i <= numItens; ++i) {
        K[i] = new int[capacidade + 1];           // Aloca memória para cada linha da tabela de DP
    }
    
    ```
    
    - Alocamos uma tabela `K` com `numItens + 1` linhas e `capacidade + 1` colunas. Cada entrada `K[i][w]` armazenará o valor máximo possível para a submochila com capacidade `w` usando os primeiros `i` itens.
2. **Preenchimento da Tabela de DP:**
    
    ```cpp
    for (int i = 0; i <= numItens; ++i) {
        for (int w = 0; w <= capacidade; ++w) {
            if (i == 0 || w == 0) {
                K[i][w] = 0;                      // Caso base: capacidade 0 ou nenhum item
            } else if (pesos[i - 1] <= w) {
                // Escolhe o máximo entre incluir ou não o item atual
                K[i][w] = std::max(valores[i - 1] + K[i - 1][w - pesos[i - 1]], K[i - 1][w]);
            } else {
                K[i][w] = K[i - 1][w];            // Não inclui o item atual
            }
        }
    }
    
    ```
    
    - Usamos um loop duplo para preencher a tabela.
    - **Caso Base:** Se não há itens (`i == 0`) ou a capacidade é zero (`w == 0`), o valor máximo é 0.
    - **Decisão:** Para cada item, verificamos se ele pode ser incluído na submochila (`pesos[i - 1] <= w`). Se puder, escolhemos o máximo entre incluir o item (somando seu valor com o valor da submochila restante) e não incluí-lo.
    - **Exclusão do Item:** Se o item não puder ser incluído, simplesmente copiamos o valor da submochila sem ele.
3. **Resultado Final:**
    
    ```cpp
    int resultado = K[numItens][capacidade];      // Resultado final da DP
    
    ```
    
    - O valor máximo possível para a mochila completa é encontrado em `K[numItens][capacidade]`.
4. **Liberação da Memória:**
    
    ```cpp
    for (int i = 0; i <= numItens; ++i) {
        delete[] K[i];                            // Libera memória para cada linha
    }
    delete[] K;                                   // Libera memória para o vetor de ponteiros
    
    ```
    
    - Após o cálculo, liberamos a memória alocada dinamicamente para a tabela `K`.

Neste contexto, a programação dinâmica (DP) é usada para resolver o problema da mochila de maneira eficiente, evitando recomputações de subproblemas ao armazenar os resultados intermediários em uma tabela. A classe `Knapsack` encapsula a lógica do problema, tornando o código modular e fácil de manter. A alocação dinâmica de memória permite a flexibilidade de lidar com diferentes tamanhos de problemas sem desperdiçar memória.

### Método para Imprimir Itens

O método `imprimeItens` imprime os itens adicionados à mochila.

```cpp
// Implementação do método para imprimir os itens adicionados
void Knapsack::imprimeItens() {
    std::cout << "Itens adicionados (peso, valor):" << std::endl;
    for (int i = 0; i < contadorItens; ++i) {
        std::cout << "(" << pesos[i] << ", " << valores[i] << ")" << std::endl;
    }
}

```

### Função `main`

A função `main` cria um objeto `Knapsack`, adiciona itens à mochila, imprime os itens adicionados e resolve o problema da mochila.

```cpp
int main() {
    int capacidade = 50;                           // Capacidade da mochila
    int numItens = 3;                              // Número de itens disponíveis

    Knapsack mochila(capacidade, numItens);

    mochila.adicionaItem(10, 60);                  // Adiciona item (peso, valor)
    mochila.adicionaItem(20, 100);                 // Adiciona item (peso, valor)
    mochila.adicionaItem(30, 120);                 // Adiciona item (peso, valor)

    mochila.imprimeItens();                        // Imprime os itens adicionados

    int valorMaximo = mochila.resolve();           // Resolve o problema da mochila

    std::cout << "Valor máximo que pode ser levado: " << valorMaximo << std::endl;

    return 0;
}

```

Neste exemplo, a classe `Knapsack` encapsula todos os dados e métodos necessários para resolver o problema da mochila. Usamos alocação dinâmica de memória para gerenciar os vetores de pesos, valores dos itens, e a tabela de programação dinâmica utilizada na solução do problema.