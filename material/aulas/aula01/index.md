!!! warning "ANTES DE COMEÇAR"
    Preencha o [forumlário](https://forms.gle/K9FK8be9HjoJ4ypGA) para criar o seu acesso ao Cluster Franky, ele será usado a partir da próxima aula!


Durante a aula, vimos que problemas computacionalmente complexos podem ser:

- **Grandes:** uma quantidade de dados absurda, que não cabe em um computador de trabalho comum

- **Intensivo:** Realiza calculos complexos e demorados, demandando horas ou dias de processamento intensivo

- **Combo:** As vezes o problema tem as duas caracteristicas, tem uma grande quantidade de dados, demanda cálculos intesivos.


Para resolver problemas desse tipo, precisamos fazer um bom uso do hardware, podemos começar usando uma linguagem de programação mais eficiente e planejando melhor o nosso código, usando as características da linguagem a nosso favor.


# Compilar/Executar Códigos C++

- Microsoft Visual Studio 
https://visualstudio.microsoft.com/pt-br/downloads/



## Se quiser usar VSCode 

### Passos para Windows

1. **Instalar o Compilador** 
    [Siga este tutorial](https://code.visualstudio.com/docs/cpp/config-mingw)

    [Link para o compilador](https://github.com/msys2/msys2-installer/releases/download/2024-12-08/msys2-x86_64-20241208.exe)

2. **Instalar Extensões Necessárias no VSCode:**
    - Abra o VSCode.
    - Vá para a aba de extensões (ícone de quadrado no lado esquerdo).
    - Pesquise e instale a extensão:
        - **C/C++** (Microsoft)

### Passos para Linux

1. **Instalar o Compilador G++:**
    - Não precisa, já vem instalado <3
        
2. **Instalar Extensões Necessárias no VSCode:**
    - Abra o VSCode.
    - Vá para a aba de extensões (ícone de quadrado no lado esquerdo).
    - Pesquise e instale as seguintes extensão:
        - **C/C++** (Microsoft)

### Passos para macOS

1. **Instalar o compilador:**
    - Não precisa, já vem instalado <3
    - Mas se quiser saber mais detalhes sobre o Clang, sugiro [este material](https://code.visualstudio.com/docs/cpp/config-clang-mac)

2. **Instalar Extensões Necessárias no VSCode:**
    - Abra o VSCode.
    - Vá para a aba de extensões (ícone de quadrado no lado esquerdo).
    - Pesquise e instale a extensões:
        - **C/C++** (Microsoft)

### Compilando um Exemplo em C++ para Testar

**Crie um arquivo `media_movel.cpp` com o seguinte conteúdo:**

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

// Constantes
const size_t N = 100'000'000;
const size_t K = 10;

// Gera vetor com valores entre 12.0 e 189.98
vector<double> gerar_leituras(size_t tamanho) {
    vector<double> dados(tamanho);
    default_random_engine gerador(42);  // Seed fixa
    uniform_real_distribution<double> distribuicao(12.0, 189.98);
    for (auto& valor : dados) {
        valor = distribuicao(gerador);
    }
    return dados;
}

// Média móvel - passagem por valor
vector<double> media_movel_por_valor(vector<double> dados, size_t k) {
    vector<double> resultado;
    resultado.reserve(dados.size() - k + 1);
    double soma = 0.0;

    for (size_t i = 0; i < k; ++i) soma += dados[i];
    resultado.push_back(soma / k);

    for (size_t i = 1; i <= dados.size() - k; ++i) {
        soma = soma - dados[i - 1] + dados[i + k - 1];
        resultado.push_back(soma / k);
    }
    return resultado;
}

// Média móvel - passagem por referência
vector<double> media_movel_por_referencia(const vector<double>& dados, size_t k) {
    vector<double> resultado;
    resultado.reserve(dados.size() - k + 1);
    double soma = 0.0;

    for (size_t i = 0; i < k; ++i) soma += dados[i];
    resultado.push_back(soma / k);

    for (size_t i = 1; i <= dados.size() - k; ++i) {
        soma = soma - dados[i - 1] + dados[i + k - 1];
        resultado.push_back(soma / k);
    }
    return resultado;
}

// Média móvel - passagem por ponteiro
vector<double> media_movel_por_ponteiro(const double* dados, size_t tamanho, size_t k) {
    vector<double> resultado;
    resultado.reserve(tamanho - k + 1);
    double soma = 0.0;

    for (size_t i = 0; i < k; ++i) soma += dados[i];
    resultado.push_back(soma / k);

    for (size_t i = 1; i <= tamanho - k; ++i) {
        soma = soma - dados[i - 1] + dados[i + k - 1];
        resultado.push_back(soma / k);
    }
    return resultado;
}

// Medição de tempo
template <typename Func, typename... Args>
double medir_tempo(Func funcao, Args&&... args) {
    auto inicio = chrono::high_resolution_clock::now();
    funcao(forward<Args>(args)...);
    auto fim = chrono::high_resolution_clock::now();
    chrono::duration<double> duracao = fim - inicio;
    return duracao.count();
}

int main() {
    cout << "Gerando dados..." << endl;
    vector<double> leituras = gerar_leituras(N);

    cout << "\n[1] Media movel por valor:" << endl;
    double t_valor = medir_tempo(media_movel_por_valor, leituras, K);
    cout << "Tempo: " << t_valor << " s" << endl;

    cout << "\n[2] MMedia movel por referencia:" << endl;
    double t_ref = medir_tempo(media_movel_por_referencia, leituras, K);
    cout << "Tempo: " << t_ref << " s" << endl;

    cout << "\n[3] MMedia movel por ponteiro:" << endl;
    const double* ptr = leituras.data();
    double t_ptr = medir_tempo(media_movel_por_ponteiro, ptr, N, K);
    cout << "Tempo: " << t_ptr << " s" << endl;

    return 0;
}

```

### **Windows** → Compilar e Executar

```
g++ media_movel.cpp -o movel.exe
./movel.exe

```

### **Linux** → Compilar e Executar:

```
g++ media_movel.cpp -o movel
./movel


```

### **MacOS** → Compilar e Executar

```
clang++ media_movel.cpp -o movel
./movel

```

Se quiser comparar o desempenho com a versão do código em python:

??? note "Versão do código em python aqui"
    ```
    import time
    import random

    # N = 100 milhões de leituras
    N = 100_000_000

    # Janela K = 10
    K = 10

    # Gera os dados
    print("Gerando dados aleatórios...")
    start_gen = time.time()
    dados = [random.uniform(12.0, 189.98) for _ in range(N)]
    print(f"Tempo para gerar os dados: {(time.time() - start_gen):.2f} segundos")

    # Calcula média móvel
    start_avg = time.time()
    media = []
    soma = sum(dados[:K])
    media.append(soma / K)

    for i in range(1, N - K):
        soma = soma - dados[i - 1] + dados[i + K - 1]
        media.append(soma / K)
    tempo = f"{time.time() - start_avg:.2f}"

    print("Tempo para calcular média móvel:", tempo, "segundos")
    ```

# Exercícios
Para você entender se você entendeu 

## Ponteiros

Ponteiros são os construtos mais básicos do C/C++ para trabalhar com a memória.
Eles carregam informações relevantes para sua manipulação, o endereço apontado,
e o tipo de dado.

**Definindo um ponteiro:** Para se definir um ponteiro adicionamos o caractere
`*` ao lado do tipo. Lembrando que o tipo agora é *Ponteiro para tipo*

```cpp
int* ponteiro_int //Ponteiro para int

MinhaClasse* ponteiro_classe //Ponteiro para minha classe
```

**Atribuindo ponteiros:** Os ponteiros são um tipo de variável que salvam um
*endereço de memória*, assim para fazer atribuição a eles utilizamos o operador
`&`, que retorna exatamente isso. 


```cpp
int a = 10;
ponteiro_int = &a;
```

**Dereferênciando um ponteiro:** Para manipular o dado que o ponteiro se
referência, é necessário utilizar o operador `*`. 

```cpp
*ponteiro_int = 20; //Altero o dado (em nosso exemplo int) na qual o ponteiro faz referência. 
std::cout << a; //Resultado será 20 pois alteramos o dado pelo ponteiro
```

## Referências

Referências são uma versão mais robusta e conveniente de ponteiros. Não podem
ser atribuídas a `null`, e não podem ser re-atribuídas também. Só essas duas
características em conjunto ajudam a evitar uma série de bugs comuns com
ponteiros. Para além disso elas simplificam o uso, já que não é necessário
utilizar os operadores de dereferência, e afins.

**Definindo uma referência:** Para declarar uma referência, utilizamos o
caractere `&` junto ao tipo. Contudo, como referências não podem ser nulas,
é obrigatório inicializá-la; 

```cpp
int b = 42;

int& referencia_int = b;
```

**Utilizando referências:** Depois de inicializadas, as referências não são mais
re-atribuídas, e quando manipulamos, manipulamos diretamente o objeto
referenciado. 

```cpp
referencia_int = 420; //Alteramos o dado na qual a referência foi definida

std::cout << b; // O resultado será 420 pois alteramos pela referência.
```

Utilizamos muito as referências, como parâmetros em nossas funções/métodos, pois
assim conseguimos um código possivelmente mais seguro, e visualmente mais limpo.


# Passagem por valor e passagem por referências (a ponteiros):

Ao criar uma função/métodos, temos a opção de fazer a transmissão do dado por
valor, ou por referência, e essa decisão tem impacto direto na memória e
velocidade de execução de um programa. 
A passagem por valor cria uma cópia completa do objeto em questão, impactando
em um maior uso de memória, e cria-se uma latência por conta da operação de
cópia. 
Já a passagem por referência, podendo ser por ponteiro, evita essas operações e
impacto em nosso programa. Para objetos grandes, isso torna-se ainda mais
crítico!
Em nosso curso é preferível se optar por passagem por referência. 

# Exercícios

1. Compilando e executando o seguinte trecho de código: 

```cpp
#include<cstdlib>
#include<iostream>

void funcao(char mensagem[]){
    mensagem = "Ola mundo";
}

int main(){

    char mensagem[] = "Bem vindo";
    funcao(mensagem);
    std::cout << mensagem;
    return 0;
}
```

Qual será a saída?

    a. Não irá compilar
    b. Ola mundo
    c. Bem vindo
    d. Executa mas dá erro.

??? note "Ver a resposta"
    Resposta: b.

2. Modifique o código da questão anterior para utilizar ponteiros.
??? note "Ver a resposta"
    Resposta:
    ```cpp
    #include<cstdlib>
    #include<cstring>
    #include<iostream>

    void funcao(char* mensagem){
        std::strcpy(mensagem,"Ola mundo");
    }

    int main(){

        char mensagem[] = "Bem vindo";
        funcao(mensagem);
        std::cout << mensagem;
        return 0;
    }
    ```

3. Analise o seguinte código:

```cpp
#include <iostream>
#include <vector>
#include <chrono>  // biblioteca para medir tempo

/**
 * Versão simples (SEM otimização)
 * Os vetores são passados por VALOR.
 * Isso significa que eles são copiados.
 */
std::vector<int> soma_vetorial(std::vector<int> a,
                               std::vector<int> b){

    std::vector<int> resultado;

    for(int i = 0; i < a.size(); i++){
        resultado.push_back(a[i] + b[i]);
    }

    return resultado;
}

int main(){

    // Criamos dois vetores com 1000 posições
    std::vector<int> a;
    std::vector<int> b;

    for(int i = 0; i < 1000; i++){
        a.push_back(i);
        b.push_back(i * 2);
    }

    // ===== INÍCIO DA MEDIÇÃO =====
    auto inicio = std::chrono::high_resolution_clock::now();

    std::vector<int> resultado = soma_vetorial(a, b);

    auto fim = std::chrono::high_resolution_clock::now();
    // ===== FIM DA MEDIÇÃO =====

    std::chrono::duration<double, std::milli> tempo = fim - inicio;

    std::cout << "Tempo de execucao: "
              << tempo.count()
              << " ms\n";

    // Imprimimos apenas os 100 primeiros elementos
    for(int i = 0; i < 100; i++){
        std::cout << resultado[i] << " ";
    }

    std::cout << std::endl;

    return 0;
}

```

Otimize o código usando referência (&) para evitar cópias desnecessárias:


??? note "Ver a resposta"
    Resposta:
    ```cpp
    #include <iostream>
    #include <vector>
    #include <chrono>  // biblioteca para medir tempo

    /**
    * Versão otimizada
    * Os vetores são passados por REFERÊNCIA constante.
    * Isso evita cópia e garante que não serão modificados.
    */
    std::vector<int> soma_vetorial(const std::vector<int>& a,
                                const std::vector<int>& b){

        std::vector<int> resultado;

        for(size_t i = 0; i < a.size(); i++){
            resultado.push_back(a[i] + b[i]);
        }

        return resultado;
    }

    int main(){

        // Criamos dois vetores com 1000 posições
        std::vector<int> a;
        std::vector<int> b;

        for(int i = 0; i < 1000; i++){
            a.push_back(i);
            b.push_back(i * 2);
        }

        // ===== INÍCIO DA MEDIÇÃO =====
        auto inicio = std::chrono::high_resolution_clock::now();

        std::vector<int> resultado = soma_vetorial(a, b);

        auto fim = std::chrono::high_resolution_clock::now();
        // ===== FIM DA MEDIÇÃO =====

        std::chrono::duration<double, std::milli> tempo = fim - inicio;

        std::cout << "Tempo de execucao: "
                << tempo.count()
                << " ms\n";

        // Imprimimos apenas os 100 primeiros elementos
        for(int i = 0; i < 100; i++){
            std::cout << resultado[i] << " ";
        }

        std::cout << std::endl;

        return 0;
    }

    ```
4. Executando a versão por valor (sem otimização) e a versão por referência (diminuindo as cópias desnecessárias) houve diferença de desempenho?

    E se usar as flags de otimização (-O2, -O3, -Ofast), houve impacto no tempo de execução do código?
    ```
    g++ Exercicio3.cpp -O2 -o Otimizado_O2
    ./Otimizado_O2

    ```

    ```
    g++ Exercicio3.cpp -O3 -o Otimizado_O3
    ./Otimizado_O3

    ```
    ```
    g++ Exercicio3.cpp -Ofast -o Otimizado_Ofast
    ./Otimizado_Ofast

    ```
