# Manipulação de Vetores

Manipulação básica de vetores em C++ envolve operações comuns como inicialização, acesso a elementos, modificação, iteração, inserção, remoção, e cópia de vetores. Esses conceitos são fundamentais, pois constituem a base para a manipulação de dados em grande escala.

### Inicialização de Vetores → Declaração e Inicialização

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec1;                // Declara um vetor vazio de inteiros
    std::vector<int> vec2(10);            // Declara um vetor de 10 inteiros inicializados com zero
    std::vector<int> vec3(10, 5);         // Declara um vetor de 10 inteiros, todos inicializados com 5

    // Exemplo de inicialização de vetor com valores específicos
    std::vector<int> vec4 = {1, 2, 3, 4, 5};

    // Imprime os elementos do vetor vec4
    for (int val : vec4) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

```

### Acesso e Modificação de Elementos

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Acessa elementos usando o operador []
    std::cout << "Primeiro elemento: " << vec[0] << std::endl;
    std::cout << "Terceiro elemento: " << vec[2] << std::endl;

    // Acessa elementos usando o método at()
    std::cout << "Segundo elemento: " << vec.at(1) << std::endl;

    return 0;
}

```

### Modificação de Elementos

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Modifica elementos usando o operador []
    vec[0] = 10;
    vec[2] = 30;

    // Modifica elementos usando o método at()
    vec.at(1) = 20;

    // Imprime os elementos modificados
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

```

### Iteração Usando Loop

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Itera sobre os elementos usando um loop tradicional
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

```

### Inserção de Elementos

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Insere elementos no final do vetor
    vec.push_back(6);
    vec.push_back(7);

    // Insere um elemento na posição específica
    vec.insert(vec.begin() + 2, 10); // Insere o valor 10 na terceira posição

    // Imprime os elementos após a inserção
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

```

### Remoção de Elementos

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Remove o último elemento
    vec.pop_back();

    // Remove um elemento na posição específica
    vec.erase(vec.begin() + 1); // Remove o segundo elemento

    // Imprime os elementos após a remoção
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

```

### Copiando Vetores

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec1 = {1, 2, 3, 4, 5};

    // Cria uma cópia de vec1
    std::vector<int> vec2 = vec1;

    // Modifica a cópia
    vec2[0] = 10;

    // Imprime os elementos dos dois vetores
    std::cout << "vec1: ";
    for (int val : vec1) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "vec2: ";
    for (int val : vec2) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

```

### Contextos Úteis para HPC

Manipulações básicas de vetores são frequentemente utilizadas em HPC para inicializar e processar grandes conjuntos de dados. Aqui estão alguns contextos úteis:

1. **Inicialização de Dados:**
    - Vetores podem ser usados para armazenar dados de entrada para simulações ou cálculos.
2. **Operações em Série:**
    - Aplicar operações matemáticas básicas em todos os elementos de um vetor é uma tarefa comum em HPC.
3. **Armazenamento de Resultados Intermediários:**
    - Vetores são úteis para armazenar resultados intermediários em algoritmos iterativos.

### Exemplo: Uso de `Vector` com Classe e Inline

A utilização de classes para encapsular a lógica de manipulação de vetores, junto com o uso de alocação dinâmica de memória e funções `inline`, permite a criação de estruturas de dados flexíveis e de alto desempenho. Neste exemplo, implementaremos uma classe `Vector` que demonstrará esses conceitos.

### Conceitos Fundamentais

**Alocação Dinâmica de Memória:**
A alocação dinâmica permite que a memória para o vetor seja alocada em tempo de execução, proporcionando flexibilidade na gestão do tamanho do vetor. Utilizamos `new` para alocar memória e `delete[]` para liberá-la, garantindo que o uso de memória seja eficiente e controlado.

**Ponteiros:**
Os ponteiros são utilizados para manipular diretamente a memória alocada dinamicamente. No nosso exemplo, `int* dados` é um ponteiro para o array que armazenará os elementos do vetor.

**Funções Inline:**
Funções `inline` são usadas para otimizar o desempenho, especialmente em métodos curtos e frequentemente chamados. A declaração `inline` sugere ao compilador que expanda o código da função no local da chamada, reduzindo a sobrecarga de chamadas de função.

**Redimensionamento Dinâmico:**
Redimensionar dinamicamente o vetor permite que ele cresça conforme necessário. Implementamos um método que duplica a capacidade do vetor quando necessário, copiando os dados existentes para um novo espaço de memória alocado.

### Implementação da Classe `Vector`

A seguir, apresentamos a implementação detalhada da classe `Vector`, que inclui métodos para inicialização, acesso, modificação, inserção e remoção de elementos, além de um método para redimensionamento dinâmico.

```cpp
#include <iostream>

class Vector {
public:
    Vector(int tamanho);                  // Construtor que inicializa o vetor
    ~Vector();                            // Destrutor que libera a memória alocada
    void inicializa(int valor);           // Método para inicializar o vetor
    inline int get(int index) const;      // Método inline para acessar um elemento
    inline void set(int index, int valor); // Método inline para modificar um elemento
    void inserir(int index, int valor);   // Método para inserir um elemento
    void remover(int index);              // Método para remover um elemento
    void imprime() const;                 // Método para imprimir o vetor
    inline int tamanho() const;           // Método inline para obter o tamanho do vetor

private:

```

### Definição da Classe `Vector`

Vamos adicionar funções inline e algumas otimizações para melhorar o desempenho onde for possível.

1. **Atributos:**
    - `int* dados`: Ponteiro para o array dinâmico que armazena os elementos do vetor.
    - `int tam`: Tamanho atual do vetor.
    - `int capacidade`: Capacidade máxima do vetor antes de precisar redimensionar.

```cpp
#include <iostream>

class Vector {
public:
    Vector(int tamanho);                  // Construtor que inicializa o vetor
    ~Vector();                            // Destrutor que libera a memória alocada
    void inicializa(int valor);           // Método para inicializar o vetor
    inline int get(int index) const;      // Método inline para acessar um elemento
    inline void set(int index, int valor); // Método inline para modificar um elemento
    void inserir(int index, int valor);   // Método para inserir um elemento
    void remover(int index);              // Método para remover um elemento
    void imprime() const;                 // Método para imprimir o vetor
    inline int tamanho() const;           // Método inline para obter o tamanho do vetor

private:
    int* dados;                           // Ponteiro para os dados do vetor
    int tam;                              // Tamanho atual do vetor
    int capacidade;                       // Capacidade máxima do vetor
    void redimensiona(int novaCapacidade); // Método para redimensionar o vetor
};

```

1. **Construtor e Destrutor:**
    - `Vector(int tamanho)`: Inicializa o vetor com o tamanho especificado e aloca memória dinamicamente.
        
        ```cpp
        Vector::Vector(int tamanho)
            : tam(tamanho), capacidade(tamanho), dados(new int[tamanho]) {}
        
        ```
        
    - `~Vector()`: Libera a memória alocada para evitar vazamentos de memória.
        
        ```cpp
        Vector::~Vector() {
            delete[] dados; // Libera a memória alocada
        }
        
        ```
        
2. **Métodos Básicos:**
    - `inicializa(int valor)`: Inicializa todos os elementos do vetor com o valor especificado.
        
        ```cpp
        void Vector::inicializa(int valor) {
            for (int i = 0; i < tam; ++i) {
                dados[i] = valor; // Inicializa cada elemento do vetor com o valor especificado
            }
        }
        
        ```
        
    - `get(int index) const`: Método inline para acessar um elemento na posição especificada.
        
        ```cpp
        inline int Vector::get(int index) const {
            if (index >= 0 && index < tam) {
                return dados[index]; // Retorna o elemento na posição especificada
            } else {
                std::cerr << "Índice fora do intervalo!" << std::endl;
                return -1; // Valor de erro
            }
        }
        
        ```
        
    - `set(int index, int valor)`: Método inline para modificar um elemento na posição especificada.
        
        ```cpp
        inline void Vector::set(int index, int valor) {
            if (index >= 0 && index < tam) {
                dados[index] = valor; // Modifica o elemento na posição especificada
            } else {
                std::cerr << "Índice fora do intervalo!" << std::endl;
            }
        }
        
        ```
        
    - `inserir(int index, int valor)`: Insere um elemento na posição especificada.
        
        ```cpp
        void Vector::inserir(int index, int valor) {
            if (index >= 0 && index <= tam) {
                if (tam >= capacidade) {
                    redimensiona(2 * capacidade); // Redimensiona o vetor se necessário
                }
                for (int i = tam; i > index; --i) {
                    dados[i] = dados[i - 1]; // Move os elementos para a direita
                }
                dados[index] = valor; // Insere o novo elemento
                tam++; // Incrementa o tamanho do vetor
            } else {
                std::cerr << "Índice fora do intervalo!" << std::endl;
            }
        }
        
        ```
        

- `remover(int index)`: Remove um elemento na posição especificada.
    
    ```cpp
    void Vector::remover(int index) {
        if (index >= 0 && index < tam) {
            for (int i = index; i < tam - 1; ++i) {
                dados[i] = dados[i + 1]; // Move os elementos para a esquerda
            }
            tam--; // Decrementa o tamanho do vetor
        } else {
            std::cerr << "Índice fora do intervalo!" << std::endl;
        }
    }
    
    ```
    
- `imprime() const`: Imprime todos os elementos do vetor.
    
    ```cpp
    void Vector::imprime() const {
        for (int i = 0; i < tam; ++i) {
            std::cout << dados[i] << " "; // Imprime cada elemento do vetor
        }
        std::cout << std::endl;
    }
    
    ```
    
- `tamanho() const`: Método inline para obter o tamanho atual do vetor.
    
    ```cpp
    inline int Vector::tamanho() const {
        return tam; // Retorna o tamanho atual do vetor
    }
    
    ```
    
- `redimensiona(int novaCapacidade)`: Redimensiona o vetor para a nova capacidade especificada, alocando nova memória e copiando os dados existentes.
    
    ```cpp
    void Vector::redimensiona(int novaCapacidade) {
        int* novoDados = new int[novaCapacidade]; // Aloca nova memória
        for (int i = 0; i < tam; ++i) {
            novoDados[i] = dados[i]; // Copia os dados antigos
        }
        delete[] dados; // Libera a memória antiga
        dados = novoDados; // Atualiza o ponteiro para os novos dados
        capacidade = novaCapacidade; // Atualiza a capacidade do vetor
    }
    
    ```
    

### Uso da Classe `Vector`

1. **Inicialização e Impressão:**
    - Criamos um vetor de tamanho 5 e inicializamos todos os elementos com 0.
    - Imprimimos o vetor inicializado.
        
        ```cpp
        int main() {
            Vector vec(5); // Cria um vetor de tamanho 5
            vec.inicializa(0); // Inicializa todos os elementos com 0
        
            std::cout << "Vetor inicializado: ";
            vec.imprime(); // Imprime o vetor inicializado
        
        ```
        
2. **Modificação:**
    - Modificamos o terceiro elemento para 10 e imprimimos o vetor.
        
        ```cpp
            vec.set(2, 10); // Modifica o terceiro elemento para 10
            std::cout << "Após modificar o terceiro elemento para 10: ";
            vec.imprime(); // Imprime o vetor após a modificação
        
        ```
        
3. **Inserção:**
    - Inserimos o valor 20 na terceira posição e imprimimos o vetor.
        
        ```cpp
            vec.inserir(2, 20); // Insere o valor 20 na terceira posição
            std::cout << "Após inserir 20 na terceira posição: ";
            vec.imprime(); // Imprime o vetor após a inserção
        
        ```
        
4. **Remoção:**
    - Removemos o segundo elemento e imprimimos o vetor.
        
        ```cpp
            vec.remover(1); // Remove o segundo elemento
            std::cout << "Após remover o segundo elemento: ";
            vec.imprime(); // Imprime o vetor após a remoção
        
        ```
        
5. **Tamanho:**
    - Imprimimos o tamanho atual do vetor.
        
        ```cpp
            std::cout << "Tamanho do vetor: " << vec.tamanho() << std::endl; // Imprime o tamanho do vetor
        
            return 0;
        }
        
        ```
        

Neste exemplo, usamos alocação dinâmica de memória e ponteiros para criar e manipular vetores em C++ usando classes e objetos. Também adicionamos funções inline para melhorar o desempenho em operações comuns como acesso e modificação de elementos.