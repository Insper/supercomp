
## Objetivo
Explorar técnicas de otimização de código sequencial em C++ a partir da análise de desempenho.
O foco será:

* Compreender a relação entre **hierarquia de memória (L1, L2, L3)** e desempenho.
* Aplicar **tiling (fateamento em blocos)** para melhorar o aproveitamento da memória cache.
* Reorganizar estruturas de dados para um melhor aproveitamento do principio da localidade espacial.


##  Contexto

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


### Perguntas para responder no relatório:

1. A troca de ordem dos loops melhorou ou piorou o tempo de execução? Por quê?
2. Houveram diferenças entre os nós **monstrao** e **gpu**? Quais?
3. Qual o **tamanho de bloco** que apresentou o melhor equilíbrio entre tempo de execução e aproveitamento de cache em cada fila?


## Entregáveis:
Submeta via [Classroom](https://classroom.github.com/a/DnVGRWYD) um relatório contendo obrigatoriamente:

* Identificação: seu nome completo

* Tabelas comparativas: contendo os resultados obtidos

* Gráficos comparativos: que ilustrem as diferenças de desempenho entre as versões testadas

* Respostas às perguntas: análise com base nos resultados observados 


 **Faça a submissão do relatório até 22/08, 08h30 pelo link do [Github Classromm](https://classroom.github.com/a/DnVGRWYD)**