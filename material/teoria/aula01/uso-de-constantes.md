# Const Correctness em HPC

Const Correctness (uso de constantes) é um conceito em C++ que garante que os dados não sejam modificados quando não deveriam ser. Usar `const` corretamente pode melhorar a legibilidade do código, evitar erros e permitir otimizações pelo compilador. 

### Usos Comuns de `const`

### 1. Variáveis Locais

Declarar variáveis locais como `const` se você não pretende modificar o valor delas.

```cpp
const int valor = 10;

```

### 2. Parâmetros de Função

Usar `const` em parâmetros de função para garantir que os argumentos não sejam modificados.

```cpp
void imprimeValor(const int valor) {
    std::cout << "Valor: " << valor << std::endl;
}

```

### 3. Ponteiros e Referências

Declarar ponteiros e referências como `const` para garantir que os dados apontados ou referenciados não sejam alterados.

```cpp
void imprimeValor(const int* ptr) {
    std::cout << "Valor: " << *ptr << std::endl;
}

void imprimeValor(const int& ref) {
    std::cout << "Valor: " << ref << std::endl;
}

```

### 4. Métodos de Classe

Declarar métodos como `const` para garantir que eles não modifiquem o estado do objeto.

```cpp
class Ponto {
public:
    Ponto(int x, int y) : x(x), y(y) {}

    int getX() const { return x; }
    int getY() const { return y; }

private:
    int x, y;
};

```

### Exemplo: Multiplicação de Matrizes com Const Correctness

Vamos considerar um exemplo de multiplicação de matrizes onde aplicamos const correctness para garantir que os dados de entrada não sejam modificados.

```cpp
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Função para multiplicar duas matrizes com const correctness
void multiplicaMatriz(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int N = A.size();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int N = 100; // Tamanho da matriz
    vector<vector<int>> A(N, vector<int>(N, 1));
    vector<vector<int>> B(N, vector<int>(N, 1));
    vector<vector<int>> C(N, vector<int>(N, 0));

    auto inicio = high_resolution_clock::now();

    // Chama a função de multiplicação de matrizes
    multiplicaMatriz(A, B, C);

    auto fim = high_resolution_clock::now();
    auto duracao = duration_cast<milliseconds>(fim - inicio).count();

    cout << "Tempo de multiplicação de matrizes: " << duracao << "ms" << endl;

    return 0;
}

```

### Const Correctness em Classes

No contexto de HPC, classes frequentemente encapsulam dados e operações. Garantir const correctness em métodos de classe é essencial.

### Exemplo: Classe de Matriz com Métodos Const

```cpp
class Matriz {
public:
    Matriz(int N) : data(N, vector<int>(N, 0)) {}

    const vector<int>& operator[](int index) const {
        return data[index];
    }

    vector<int>& operator[](int index) {
        return data[index];
    }

    void imprime() const {
        for (const auto& linha : data) {
            for (int valor : linha) {
                cout << valor << " ";
            }
            cout << endl;
        }
    }

private:
    vector<vector<int>> data;
};

int main() {
    Matriz matriz(3);

    matriz[0][0] = 1;
    matriz[1][1] = 2;
    matriz[2][2] = 3;

    cout << "Matriz:" << endl;
    matriz.imprime();

    return 0;
}

```

Aplicar const correctness pode prevenir erros e permitir otimizações adicionais pelo compilador. Usar `const` de maneira apropriada em variáveis, parâmetros de função, ponteiros, referências e métodos de classe é uma prática recomendada para escrever código robusto e eficiente.