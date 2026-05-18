# Conceitos básicos de C++

Esses são os tipos de variáveis e seus respectivos tamanhos em C++ 

| Tipo de Dados | Tamanho (em bytes) | Valor Mínimo | Valor Máximo |
| --- | --- | --- | --- |
| bool | 1 | false | true |
| char | 1 | -128 | 127 |
| unsigned char | 1 | 0 | 255 |
| short | 2 | -32,768 | 32,767 |
| unsigned short | 2 | 0 | 65,535 |
| int | 4 | -2,147,483,648 | 2,147,483,647 |
| unsigned int | 4 | 0 | 4,294,967,295 |
| long | 8 | -9,223,372,036,854,775,808 | 9,223,372,036,854,775,807 |
| unsigned long | 8 | 0 | 18,446,744,073,709,551,615 |
| float | 4 | 1.2E-38 | 3.4E+38 |
| double | 8 | 2.3E-308 | 1.7E+308 |
| long double | 16 | 3.4E-4932 | 1.1E+4932 |
| wchar_t | 4 | 0 | 4,294,967,295  |

<aside>
⚠️ Esses tamanhos podem variar dependendo da arquitetura do sistema. Esta tabela assume um sistema de 64 bits.

</aside>

### O que é um Namespace?

Um namespace é uma forma de agrupar identificadores (nomes de funções, classes, objetos, etc.) sob um nome comum, evitando conflitos de nome entre diferentes partes de um programa ou entre diferentes bibliotecas.

### Exemplo Simples de Namespace

Imagine duas bibliotecas diferentes que ambas definem uma função chamada `print()`. Se você incluir ambas as bibliotecas em seu programa, o compilador não saberá qual `print()` usar. Para resolver isso, cada biblioteca pode colocar sua função `print()` em seu próprio namespace:

```cpp
// Biblioteca A
namespace A {
    void print() {
        std::cout << "Imprimindo da biblioteca A" << std::endl;
    }
}

// Biblioteca B
namespace B {
    void print() {
        std::cout << "Imprimindo da biblioteca B" << std::endl;
    }
}

int main() {
    A::print(); // Chama a função print() da biblioteca A
    B::print(); // Chama a função print() da biblioteca B
    return 0;
}

```

### Usando Namespaces

Existem várias maneiras de usar namespaces em C++:

1. **Usar o nome completo do namespace (qualificação total):**
    
    ```cpp
    #include <iostream>
    
    int main() {
        std::cout << "Hello, World!" << std::endl; // Usa std::cout e std::endl
        return 0;
    }
    
    ```
    
2. **Usar a declaração `using` para trazer membros específicos do namespace para o escopo atual:**
    
    ```cpp
    #include <iostream>
    using std::cout;
    using std::endl;
    
    int main() {
        cout << "Hello, World!" << endl; // Usa cout e endl sem o prefixo std::
        return 0;
    }
    
    ```
    
3. **Usar a diretiva `using` para trazer todos os membros do namespace para o escopo atual:**
    
    ```cpp
    #include <iostream>
    using namespace std;
    
    int main() {
        cout << "Hello, World!" << endl; // Usa cout e endl sem o prefixo std::
        return 0;
    }
    
    ```
    

### O que é `std`?

`std` é o namespace padrão da biblioteca padrão C++ (Standard Library). Ele contém a maior parte das funções, objetos, tipos e classes fornecidos pela biblioteca padrão do C++. Quando você usa recursos da biblioteca padrão, como `std::vector`, `std::cout`, `std::string`, etc., você está acessando esses elementos do namespace `std`.

Por exemplo:

- `std::cout` é o objeto de fluxo de saída padrão usado para imprimir dados na tela.
- `std::vector` é uma classe de contêiner que representa um array dinâmico.