# Loops  e Laços

No contexto de HPC, onde a eficiência e a performance são cruciais, as estruturas de controle (loops e laços) desempenham papéis vitais:

1. **Otimização de Algoritmos**
    - As estruturas de controle permitem que algoritmos sejam implementados de forma eficiente. Condicionais e loops bem utilizados podem reduzir o número de operações e evitar cálculos desnecessários, otimizando o tempo de execução.
2. **Paralelismo**
    - Em HPC, o paralelismo é frequentemente utilizado para acelerar a execução dos programas. Estruturas de controle são essenciais para dividir tarefas entre diferentes threads ou processos.
    - O uso adequado de condicionais pode garantir que as tarefas sejam distribuídas eficientemente entre os recursos computacionais, evitando sobrecarga em um único nó de processamento.
3. **Balanceamento de Carga**
    - Estruturas de controle podem ajudar no balanceamento de carga, distribuindo o trabalho de maneira uniforme entre os processadores. Por exemplo, condicionais podem ser usados para verificar a carga de trabalho em diferentes nós e ajustar dinamicamente a distribuição das tarefas.
    - Isso é crucial para evitar situações onde alguns processadores ficam ociosos enquanto outros estão sobrecarregados, maximizando a utilização de recursos e melhorando a performance geral.
4. **Gerenciamento de Recursos**
    - Condicionais e loops podem ser usados para gerenciar recursos, como alocação de memória e acesso a dispositivos de I/O. Em ambientes HPC, onde grandes volumes de dados são manipulados, o gerenciamento eficiente de memória é fundamental. Estruturas de controle podem ajudar a evitar desperdício de memória e garantir que os recursos sejam utilizados de maneira eficiente.

### Estruturas de controle condicionais (`if`, `else if`, `else` ):

Exemplo: Encontrar o valor máximo em uma matriz

```cpp
#include <iostream>
#include <vector>

int main() {
    // Inicializa uma matriz 3x3 com valores inteiros
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Define a variável max_value com o primeiro valor da matriz
    int max_value = matrix[0][0];

    // Percorre cada linha da matriz
    for (size_t i = 0; i < matrix.size(); ++i) {
        // Percorre cada coluna da matriz
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            // Se o valor atual da matriz for maior que max_value, atualiza max_value
            if (matrix[i][j] > max_value) {
                max_value = matrix[i][j];
            }
        }
    }

    // Imprime o valor máximo encontrado na matriz
    std::cout << "O valor máximo na matriz é: " << max_value << std::endl;

    return 0; // Termina o programa
}

```

### Estrutura de controle condicional `switch` :

Exemplo: Imprimir a posição de um número específico na matriz

```cpp
#include <iostream>
#include <vector>

int main() {
    // Inicializa uma matriz 3x3 com valores inteiros
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Define o valor alvo a ser encontrado
    int target = 5;

    // Percorre cada linha da matriz
    for (size_t i = 0; i < matrix.size(); ++i) {
        // Percorre cada coluna da matriz
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            // Verifica o valor atual da matriz usando switch
            switch(matrix[i][j]) {
                case 5:
                    // Se o valor for 5, imprime a posição e sai do switch
                    std::cout << "Número " << target << " encontrado em: (" << i << ", " << j << ")" << std::endl;
                    break;
                default:
                    // Caso padrão do switch, não faz nada
                    break;
            }
        }
    }

    return 0; // Termina o programa
}

```

### Estrutura de controle loop `for` :

Exemplo Somar todos os elementos de uma matriz :

```cpp
#include <iostream>
#include <vector>

int main() {
    // Inicializa uma matriz 3x3 com valores inteiros
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Inicializa a variável sum com 0
    int sum = 0;

    // Percorre cada linha da matriz
    for (size_t i = 0; i < matrix.size(); ++i) {
        // Percorre cada coluna da matriz
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            // Adiciona o valor atual da matriz à sum
            sum += matrix[i][j];
        }
    }

    // Imprime a soma de todos os elementos na matriz
    std::cout << "A soma de todos os elementos na matriz é: " << sum << std::endl;

    return 0; // Termina o programa
}

```

### Estrutura de controle loop `while` :

Encontrar um número específico na matriz 

```cpp
#include <iostream>
#include <vector>

int main() {
    // Inicializa uma matriz 3x3 com valores inteiros
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Define o valor alvo a ser encontrado
    int target = 5;
    bool found = false; // Flag para indicar se o valor foi encontrado
    size_t i = 0; // Índice para as linhas

    // Loop externo para percorrer as linhas
    while (i < matrix.size() && !found) {
        size_t j = 0; // Índice para as colunas
        // Loop interno para percorrer as colunas
        while (j < matrix[i].size() && !found) {
            // Se o valor atual da matriz for igual ao alvo, imprime a posição
            if (matrix[i][j] == target) {
                std::cout << "Número " << target << " encontrado em: (" << i << ", " << j << ")" << std::endl;
                found = true; // Atualiza a flag
            }
            ++j; // Incrementa o índice das colunas
        }
        ++i; // Incrementa o índice das linhas
    }

    // Se o valor não foi encontrado, imprime uma mensagem
    if (!found) {
        std::cout << "Número " << target << " não encontrado na matriz." << std::endl;
    }

    return 0; // Termina o programa
}

```

### Estrutura de controle loop `do-while` :

Verificar se todos os elementos da matriz são positivos 

```cpp
#include <iostream>
#include <vector>

int main() {
    // Inicializa uma matriz 3x3 com valores inteiros
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    bool all_positive = true; // Flag para indicar se todos os elementos são positivos
    size_t i = 0; // Índice para as linhas

    // Loop externo do-while para percorrer as linhas
    do {
        size_t j = 0; // Índice para as colunas
        // Loop interno do-while para percorrer as colunas
        do {
            // Se o valor atual da matriz for menor ou igual a 0, atualiza a flag e sai do loop
            if (matrix[i][j] <= 0) {
                all_positive = false;
                break;
            }
            ++j; // Incrementa o índice das colunas
        } while (j < matrix[i].size());
        ++i; // Incrementa o índice das linhas
    } while (i < matrix.size() && all_positive);

    // Imprime o resultado da verificação
    if (all_positive) {
        std::cout << "Todos os elementos da matriz são positivos." << std::endl;
    } else {
        std::cout << "Nem todos os elementos da matriz são positivos." << std::endl;
    }

    return 0; // Termina o programa
}

```

### Controladores de loop `break` :

Interromper a busca ao encontrar um número específico (`break`)

```cpp
#include <iostream>
#include <vector>

int main() {
    // Inicializa uma matriz 3x3 com valores inteiros
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int target = 5; // Define o valor alvo a ser encontrado
    bool found = false; // Flag para indicar se o valor foi encontrado

    // Loop externo para percorrer as linhas
    for (size_t i = 0; i < matrix.size(); ++i) {
        // Loop interno para percorrer as colunas
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            // Se o valor atual da matriz for igual ao alvo, imprime a posição
            if (matrix[i][j] == target) {
                std::cout << "Número " << target << " encontrado em: (" << i << ", " << j << ")" << std::endl;
                found = true; // Atualiza a flag
                break; // Interrompe o loop interno
            }
        }
        if (found) {
            break; // Interrompe o loop externo
        }
    }

    // Se o valor não foi encontrado, imprime uma mensagem
    if (!found) {
        std::cout << "Número " << target << " não encontrado na matriz." << std::endl;
    }

    return 0; // Termina o programa
}

```

### Controladores de loop continue :

Ignorar números negativos ao somar elementos de uma matriz (`continue`)

```cpp
#include <iostream>
#include <vector>

int main() {
    // Inicializa uma matriz 3x3 com valores inteiros, incluindo números negativos
    std::vector<std::vector<int>> matrix = {
        {1, -2, 3},
        {-4, 5, -6},
        {7, -8, 9}
    };

    int sum = 0; // Inicializa a variável sum com 0

    // Loop externo para percorrer as linhas
    for (size_t i = 0; i < matrix.size(); ++i) {
        // Loop interno para percorrer as colunas
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            // Se o valor atual da matriz for negativo, ignora-o e continua para a próxima iteração
            if (matrix[i][j] < 0) {
                continue; // Ignora números negativos
            }
            // Adiciona o valor atual da matriz à soma
            sum += matrix[i][j];
        }
    }

    // Imprime a soma de todos os elementos positivos na matriz
    std::cout << "A soma de todos os elementos positivos na matriz é: " << sum << std::endl;

    return 0; // Termina o programa
}

```