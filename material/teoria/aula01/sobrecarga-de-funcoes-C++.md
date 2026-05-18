# Sobrecarga de Funções em C++

Sobrecarga de funções é um recurso da linguagem C++ que permite definir múltiplas funções com o mesmo nome, mas com diferentes listas de parâmetros. Isso significa que você pode ter várias funções que realizam tarefas semelhantes, mas aceitam diferentes tipos ou números de argumentos.

No contexto de HPC, a sobrecarga de funções pode ser usada para otimizar operações em diferentes tipos de dados (por exemplo, operações em inteiros, floats e doubles) sem duplicar o código. Isso pode ajudar a escrever código mais limpo e eficiente.


### Regras para Sobrecarga de Funções

Para que duas ou mais funções sejam sobrecarregadas corretamente, elas devem ser diferentes em pelo menos um dos seguintes aspectos:

1. **Número de parâmetros:** As funções devem ter um número diferente de parâmetros.
2. **Tipo de parâmetros:** As funções devem ter tipos diferentes de parâmetros.

A sobrecarga de funções não pode ser feita apenas com base no tipo de retorno das funções.

### Exemplo Básico de Sobrecarga de Funções

```cpp
#include <iostream>
#include <string>

// Função que imprime um inteiro
void imprimir(int valor) {
    std::cout << "Inteiro: " << valor << std::endl;
}

// Função que imprime um float
void imprimir(float valor) {
    std::cout << "Float: " << valor << std::endl;
}

// Função que imprime uma string
void imprimir(std::string valor) {
    std::cout << "String: " << valor << std::endl;
}

int main() {
    imprimir(10);          // Chama a função que imprime um inteiro
    imprimir(3.14f);       // Chama a função que imprime um float
    imprimir("Hello");     // Chama a função que imprime uma string

    return 0;
}

```

Neste exemplo, a função `imprimir` é sobrecarregada para aceitar diferentes tipos de argumentos: `int`, `float` e `std::string`

Vamos considerar um exemplo de sobrecarga de funções para calcular o produto escalar de vetores de diferentes tipos.

```cpp
#include <iostream>
#include <vector>

// Função para calcular o produto escalar de vetores de inteiros
int produtoEscalar(const std::vector<int>& v1, const std::vector<int>& v2) {
    int produto = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        produto += v1[i] * v2[i];
    }
    return produto;
}

// Função para calcular o produto escalar de vetores de floats
float produtoEscalar(const std::vector<float>& v1, const std::vector<float>& v2) {
    float produto = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        produto += v1[i] * v2[i];
    }
    return produto;
}

// Função para calcular o produto escalar de vetores de doubles
double produtoEscalar(const std::vector<double>& v1, const std::vector<double>& v2) {
    double produto = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        produto += v1[i] * v2[i];
    }
    return produto;
}

int main() {
    std::vector<int> v1_int = {1, 2, 3};
    std::vector<int> v2_int = {4, 5, 6};

    std::vector<float> v1_float = {1.0f, 2.0f, 3.0f};
    std::vector<float> v2_float = {4.0f, 5.0f, 6.0f};

    std::vector<double> v1_double = {1.0, 2.0, 3.0};
    std::vector<double> v2_double = {4.0, 5.0, 6.0};

    std::cout << "Produto Escalar (int): " << produtoEscalar(v1_int, v2_int) << std::endl;
    std::cout << "Produto Escalar (float): " << produtoEscalar(v1_float, v2_float) << std::endl;
    std::cout << "Produto Escalar (double): " << produtoEscalar(v1_double, v2_double) << std::endl;

    return 0;
}

```

### Sobrecarga de Funções e Templates

Outra abordagem para lidar com operações semelhantes em diferentes tipos de dados é o uso de templates de funções. Templates podem ser usados para evitar a necessidade de sobrecarregar funções manualmente para cada tipo de dado.

### Exemplo de Template de Função

```cpp
#include <iostream>
#include <vector>

// Template de função para calcular o produto escalar de vetores
template <typename T>
T produtoEscalar(const std::vector<T>& v1, const std::vector<T>& v2) {
    T produto = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        produto += v1[i] * v2[i];
    }
    return produto;
}

int main() {
    std::vector<int> v1_int = {1, 2, 3};
    std::vector<int> v2_int = {4, 5, 6};

    std::vector<float> v1_float = {1.0f, 2.0f, 3.0f};
    std::vector<float> v2_float = {4.0f, 5.0f, 6.0f};

    std::vector<double> v1_double = {1.0, 2.0, 3.0};
    std::vector<double> v2_double = {4.0, 5.0, 6.0};

    std::cout << "Produto Escalar (int): " << produtoEscalar(v1_int, v2_int) << std::endl;
    std::cout << "Produto Escalar (float): " << produtoEscalar(v1_float, v2_float) << std::endl;
    std::cout << "Produto Escalar (double): " << produtoEscalar(v1_double, v2_double) << std::endl;

    return 0;
}

```

A sobrecarga de funções em C++ permite a definição de múltiplas funções com o mesmo nome, mas com diferentes listas de parâmetros. Isso melhora a legibilidade, manutenção e facilidade de uso do código. No contexto de HPC, a sobrecarga de funções pode ser usada para otimizar operações em diferentes tipos de dados, evitando duplicação de código e melhorando a eficiência. Alternativamente, templates de funções podem ser usados para alcançar resultados similares com menos código.
