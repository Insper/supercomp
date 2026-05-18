# Passagem de Parâmetros

Parâmetros podem ser passados por valor, por referência ou por ponteiro. No contexto de HPC, passar parâmetros por referência ou ponteiro é geralmente preferível para evitar cópias desnecessárias de dados, que podem ser custosas em termos de tempo e memória.

### Passagem de Parâmetros por Valor

Passar por valor significa que uma cópia do argumento é passada para a função. Qualquer modificação feita ao parâmetro dentro da função não afeta o argumento original.

```cpp
// Função que recebe um parâmetro por valor
void exemploValor(int x) {
    x = 10; // Modificação local, não afeta o argumento original
}

```

### Passagem por Referência e Passagem por Ponteiro

Passagem por referência e passagem por ponteiro são duas formas de passar argumentos para funções em C++, permitindo que a função modifique o argumento original. Apesar de terem propósitos similares, elas diferem em sintaxe e uso. Vamos explorar essas diferenças detalhadamente.

### Passagem por Referência

Passar um argumento por referência significa que a função recebe uma referência ao argumento original, permitindo modificar diretamente o valor do argumento. A sintaxe usa o operador `&` no parâmetro da função.

### Sintaxe e Exemplo

```cpp
#include <iostream>

// Função que recebe um parâmetro por referência
void alteraPorReferencia(int& x) {
    x = 10; // Modificação afeta o argumento original
}

int main() {
    int valor = 5;
    std::cout << "Antes da função: " << valor << std::endl;
    alteraPorReferencia(valor);
    std::cout << "Depois da função: " << valor << std::endl;
    return 0;
}

```

### Características

- **Sintaxe Limpa:** A sintaxe é mais clara e fácil de ler, pois não envolve o uso explícito de ponteiros.
- **Segurança:** Reduz o risco de manipulação incorreta de ponteiros (como desreferenciamento de ponteiros nulos).
- **Não Nulo:** Referências devem ser inicializadas e não podem ser nulas.

### Passagem por Ponteiro

Passar um argumento por ponteiro significa que a função recebe o endereço do argumento original. A sintaxe usa o operador `*` no parâmetro da função e o operador `&` ao passar o argumento.

### Sintaxe e Exemplo

```cpp
#include <iostream>

// Função que recebe um parâmetro por ponteiro
void alteraPorPonteiro(int* x) {
    *x = 10; // Modificação afeta o argumento original
}

int main() {
    int valor = 5;
    std::cout << "Antes da função: " << valor << std::endl;
    alteraPorPonteiro(&valor);
    std::cout << "Depois da função: " << valor << std::endl;
    return 0;
}

```

### Características

- **Flexibilidade:** Permite a passagem de valores nulos (ponteiros nulos).
- **Controle Explícito:** Fornece controle explícito sobre a memória, podendo ser útil em contextos onde manipulação direta de endereços é necessária.
- **Complexidade:** A sintaxe pode ser mais complexa e propensa a erros, como desreferenciamento de ponteiros nulos ou incorretos.

### Quando Usar Cada Um

- **Passagem por Referência:** Use quando você precisa modificar o argumento original e quer uma sintaxe mais limpa e segura. Ideal para a maioria dos casos onde a referência não precisa ser nula.
- **Passagem por Ponteiro:** Use quando há a necessidade de manipular diretamente endereços de memória ou quando o valor passado pode ser opcional (nulo).

### Exemplo Comparativo

Vamos comparar um exemplo onde modificamos um valor usando ambas as abordagens.

### Passagem por Referência

```cpp
#include <iostream>

void incrementaReferencia(int& x) {
    x++; // Incrementa o valor
}

int main() {
    int valor = 5;
    std::cout << "Antes: " << valor << std::endl;
    incrementaReferencia(valor);
    std::cout << "Depois: " << valor << std::endl;
    return 0;
}

```

### Passagem por Ponteiro

```cpp
#include <iostream>

void incrementaPonteiro(int* x) {
    if (x) { // Verifica se o ponteiro não é nulo
        (*x)++; // Incrementa o valor
    }
}

int main() {
    int valor = 5;
    std::cout << "Antes: " << valor << std::endl;
    incrementaPonteiro(&valor);
    std::cout << "Depois: " << valor << std::endl;
    return 0;
}

```

Ambos os exemplos acima modificam o valor original de `valor`, mas a abordagem de referência é mais limpa, enquanto a abordagem de ponteiro oferece maior flexibilidade em termos de manipulação de endereços e valores nulos.