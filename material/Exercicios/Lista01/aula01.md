# Otimizações Fundamentais

## Questão 1 — Classificação de problemas em HPC (Teórica)

**Tipo:** múltipla escolha (múltiplas corretas)  
**Enunciado:** Classifique cada cenário como Grande (G), Intensivo (I) ou Combo (C).

a) Treinar um modelo com bilhões de parâmetros, mas com dataset moderado.  
b) Processar todos os frames de câmeras de uma cidade por 30 dias.  
c) Resolver simulações de mecânica dos fluidos 3D com malha fina para um avião.  
d) Indexar petabytes de logs e calcular métricas em janelas de tempo curtas.  

**Recaptulando...** “Grandes” se refere a um conjunto de dados muito grande, “Intensivos” se refere a algoritimos demorados, resolução complexa de equações, etc.. e “Combo” são as duas características juntas. 

??? note "Ver a resposta"

    **Gabarito:**  
    a) **I** (Intensivo)  
    b) **G** (Grande)  
    c) **I** (Intensivo)  
    d) **C** (Combo)

    **Por quê?**  
    - **a)** Treinar um modelo com bilhões de parâmetros é **Intensivo**, o dataset sendo moderado, não é crítico então excluimos a opção de *Combo*.  
    - **b)** Volume massivo de dados ⇒ **Grande**.  
    - **c)** CFD 3D com malha fina é clássico **Intensivo**.  
    - **d)** Petabytes + métricas em janelas curtas ⇒ **Combo**.

---

## Questão 2 — Média móvel por valor e por referência

**Enunciado:**  
Implemente as duas versões abaixo da média móvel simples com janela `K`:  

(a) recebendo os dados por valor  
(b) recebendo os dados por referência constante  

```cpp
// =========================================
// Função para calcular a média móvel (passagem por valor)
// =========================================
vector<double> media_movel_por_valor(vector<double> dados, size_t K) {
    // TODO: Usar soma inicial dos K primeiros elementos
    // TODO: Atualizar soma a cada passo removendo o primeiro e adicionando o próximo
    // TODO: Retornar vetor com médias
    return {};
}

// =========================================
// Função para calcular a média móvel (passagem por referência)
// =========================================
vector<double> media_movel_por_referencia(const vector<double>& dados, size_t K) {
    // TODO: Mesma lógica da versão por valor
    // TODO: Não copiar o vetor original
    return {};
}
```

??? note "Ver Resposta"
    
        #include <vector>
        using std::vector;

        // =====================================================
        // (a) Função que recebe o vetor por VALOR
        //     → Uma cópia de 'dados' é criada dentro da função.
        //     → Essa cópia aumenta custo de memória/tempo se
        //       o vetor for muito grande.
        // =====================================================
        vector<double> media_movel_por_valor(vector<double> dados, size_t K) {
            const size_t N = dados.size();

            // Se a janela for inválida, retorna vetor vazio
            if (K == 0 || K > N) return {};

            // Vetor de saída que conterá as médias
            vector<double> medias;

            // Calcula a soma inicial dos K primeiros elementos
            double soma = 0.0;
            for (size_t i = 0; i < K; i++) {
                soma += dados[i];
            }
            medias.push_back(soma / K); // primeira média

            // Desliza a janela: remove o elemento que saiu
            // e adiciona o próximo elemento
            for (size_t i = K; i < N; i++) {
                soma += dados[i] - dados[i - K];
                medias.push_back(soma / K);
            }

            // Retorna o vetor de médias
            return medias;
        }

        // =====================================================
        // (b) Função que recebe o vetor por REFERÊNCIA CONST
        //     → Não há cópia: a função acessa o mesmo vetor.
        //     → Mais eficiente em memória e tempo.
        //     → 'const' garante que a função NÃO altera os dados.
        // =====================================================
        vector<double> media_movel_por_referencia(const vector<double>& dados, size_t K) {
            const size_t N = dados.size();

            // Se a janela for inválida, retorna vetor vazio
            if (K == 0 || K > N) return {};

            // Vetor de saída que conterá as médias
            vector<double> medias;

            // Calcula a soma inicial dos K primeiros elementos
            double soma = 0.0;
            for (size_t i = 0; i < K; i++) {
                soma += dados[i];
            }
            medias.push_back(soma / K);

            // Desliza a janela e calcula as médias seguintes
            for (size_t i = K; i < N; i++) {
                soma += dados[i] - dados[i - K];
                medias.push_back(soma / K);
            }

            // Retorna o vetor de médias
            return medias;
        }

        
---

## Questão 3 — Média móvel por ponteiro

**Enunciado:**  
Implemente a função `media_movel_por_ponteiro` que recebe um ponteiro para `double` (`const double*`) e calcula a média móvel com aritmética de ponteiros. Retorne o vetor de médias.

```cpp
// =========================================
// Função para calcular a média móvel (passagem por ponteiro)
// =========================================
vector<double> media_movel_por_ponteiro(const double* ptr, size_t N, size_t K) {
    // TODO: Implementar usando aritmética de ponteiros
    // Exemplo: *(ptr + i) acessa o elemento i
    return {};
}
```
??? note "ver resposta"
        #include <vector>
        using std::vector;

        // Função para calcular a média móvel usando ponteiros
        vector<double> media_movel_por_ponteiro(const double* ptr, size_t N, size_t K) {
            // Verifica se o ponteiro é válido e se a janela K faz sentido
            if (!ptr || K == 0 || K > N) return {};

            vector<double> medias;      // vetor que vai guardar os resultados
            double soma = 0.0;          // acumulador da soma da janela

            // Calcula a soma inicial dos K primeiros elementos
            for (size_t i = 0; i < K; i++) {
                soma += *(ptr + i);     // *(ptr + i) é o mesmo que ptr[i]
            }
            medias.push_back(soma / K); // primeira média

            // Desliza a janela até o final
            for (size_t i = K; i < N; i++) {
                soma += *(ptr + i) - *(ptr + i - K); // atualiza a soma
                medias.push_back(soma / K);          // guarda a média
            }

            return medias; // retorna o vetor de médias
        }


## Questão 4 — Teórica (Hierarquia de Memória e Localidade)
**Tipo:** múltipla escolha (múltiplas corretas)  
**Enunciado:**  
Sobre a hierarquia de memória (L1, L2, L3, RAM) e os princípios de localidade:  

a) L1 é a memória cache mais rápida, porém é a maior.  
b) O princípio da localidade temporal sugere que dados acessados recentemente provavelmente serão acessados novamente em breve.  
c) O princípio da localidade espacial sugere que percorrer um vetor de 3 dimensões varrendo coluna por coluna em um laço for encadeado é a forma mais eficiente.  
d) Se um loop acessa elementos distantes no vetor (p. ex., `array[i*1000]`), isso prejudica a localidade espacial.  

??? note "Ver a resposta"
    - a) Incorreta. L1 é realmente a mais rápida, mas não é a maior. Entre caches, a maior costuma ser a L3; em geral, a RAM é muito maior, mas bem mais lenta.

    - b) Correta. Esse é o conceito clássico de **localidade temporal**: reutilizar dados acessados há pouco tempo.

    - c) Incorreta. Isso **fere o princípio da localidade espacial**, porque os dados são armazenados em ordem de linhas na memória (RAM e caches). Se percorrermos **coluna por coluna**, os acessos não serão contíguos. O correto seria percorrer **linha por linha** para aproveitar os blocos que já estão em cache.

    - d) Correta. Pular elementos gera desperdício de cache, porque várias posições carregadas não serão usadas. Além disso, há overhead de instruções para calcular o índice distante, em vez de simplesmente incrementar o endereço contíguo.

---

## Questão 5 — Versão Ingênua (Cálculo de Produto Vetorial)
**Enunciado:**  
Implemente uma função que recebe dois vetores `A` e `B` de tamanho `N` e calcula o vetor `C`, onde:  

\[ C[i] = \sum_{k=0}^{N-1} A[i] \times B[k] \]  

ou seja, cada posição de `C` é a soma do produto de `A[i]` com todos os elementos de `B`.  
Depois, compile o programa e rode no cluster com SLURM utilizando a fila express.  

```cpp
void produtoIngenuo(vector<double>& A, vector<double>& B, vector<double>& C, int N) {
    // TODO: loop i
    // TODO: loop k
    // TODO: C[i] += A[i] * B[k];
}
```

??? note "Ver resposta"
    **Explicação:**

    O loop **externo (`i`)** percorre todos os elementos de `A` e `C`.  
    O loop **interno (`k`)** percorre todos os elementos de `B`.  
    Essa é a **versão ingênua**, porque sempre percorremos `B` inteiro para cada `i`.  
    **Localidade temporal:** cada `A[i]` é reutilizado N vezes.  
    **Localidade espacial:** `B[k]` é percorrido de forma sequencial, o que é bom para cache.  
        
        #include <vector>
        using std::vector;

        // Implementação ingênua do cálculo
        void produtoIngenuo(vector<double>& A, vector<double>& B, vector<double>& C, int N) {
            // Percorre cada posição do vetor C
            for (int i = 0; i < N; i++) {
                C[i] = 0.0; // inicializa C[i] em zero

                // Para cada A[i], percorre todo o vetor B
                for (int k = 0; k < N; k++) {
                    C[i] += A[i] * B[k]; // acumula o produto
                }
            }
        }
        

    Exemplo de script `run.slurm` para rodar no cluster:

        #!/bin/bash
        #SBATCH --job-name=produtoIngenuo   # nome do job
        #SBATCH --output=saida.out          # arquivo de saída
        #SBATCH --partition=express         # fila express
        #SBATCH --nodes=1                   # 1 nó
        #SBATCH --ntasks=1                  # 1 tarefa
        #SBATCH --mem=2G
        #SBATCH --time=00:05:00             # tempo 5 min
        
        ./produtoIngenuo
        


---

## Questão 6 — Versão com Tiling (Multiplicação de Blocos de Vetores)
**Enunciado:**  
Agora implemente uma versão **com tiling**:  
- Divida o vetor `B` em blocos de tamanho `Bsize`.  
- Para cada `A[i]`, some os produtos em blocos de `B` antes de acumular no resultado `C[i]`.  
- Compare o desempenho com a versão ingênua usando SLURM.  

```cpp
void produtoTiling(vector<double>& A, vector<double>& B, vector<double>& C, int N, int Bsize) {
    // TODO: loop externo sobre blocos de B
    // TODO: loop i sobre elementos de A
    // TODO: loop k dentro do bloco de B
}
```

??? note "Ver Resposta"
    Usando a **cache L2** latência menor que RAM e a L3 capacidade bem maior que L1. Deixe **margem** para A, C e demais dados. Uma boa heurística é usar **60%–75% da L2** para o bloco de `B`.
  
    Ganhos sobre a versão ingênua:
    - Acesso **espacial** contíguo a `B` dentro do bloco.  
    - Maior chance de **vetorização** no laço de `i`.  

        #include <vector>
        #include <algorithm>
        using namespace std;

        // ======================================================
        // Função: produtoTiling
        // Objetivo: calcular C[i] = soma_k ( A[i] * B[k] )
        // Estratégia: TILING (processar B em blocos contíguos)
        //   - Varremos B em blocos [kk, kend)
        //   - Para cada bloco de B, atualizamos todos os C[i]
        // Benefício: melhor reuso temporal/espacial de B (cache), reduzindo misses
        // ======================================================
        void produtoTiling(vector<double>& A, vector<double>& B, vector<double>& C, int N, int Bsize) {
            // Inicializa C uma única vez
            for (int i = 0; i < N; i++) {
                C[i] = 0.0;
            }

            // Loop externo: percorre B em blocos de tamanho Bsize
            for (int kk = 0; kk < N; kk += Bsize) {
                // Limite do bloco (cuida da "sobra" no final)
                int kend = min(kk + Bsize, N);

                // Para cada elemento de A (e C correspondente)...
                for (int i = 0; i < N; i++) {
                    // ...percorrermos APENAS o bloco atual de B
                    for (int k = kk; k < kend; k++) {
                        // A[i] * B[k] contribui para C[i]
                        C[i] += A[i] * B[k];
                    }
                }
            }
        }
     

---

## Questão 7 — Reordenação de Loops + Flags de Otimização
**Enunciado:**  
Implemente a versão reordenada:  
- Para cada `k`, carregue uma cópia temporária de `B[k]` em cache.  
- Depois percorra todos os elementos de `A` para atualizar `C[i]`.  
- Qual a flag de compilação que resulta no binário mais eficiente?  

```cpp
void produtoReordenado(vector<double>& A, vector<double>& B, vector<double>& C, int N) {
    // TODO: loop k
    // TODO: armazenar tempB = B[k]
    // TODO: loop i → C[i] += A[i] * tempB
}
```
??? note "Ver Resposta"
    Usando a **cache L2** latência menor que RAM e a L3 capacidade bem maior que L1. Deixe **margem** para A, C e demais dados. Uma boa heurística é usar **60%–75% da L2** para o bloco de `B`.
  
    Ganhos sobre a versão ingênua:
    - Reuso **temporal** forte de `B[k]` e de `tempB`.  
    - Acesso **espacial** contíguo a `B` dentro do bloco.  
    - Maior chance de **vetorização** no laço de `i`.  

        #include <vector>
        using std::vector;

        // TILING + HOIST:
        // - Varremos B em BLOCOS contíguos (tiles) que caibam na CACHE L2.
        // - Para cada k do bloco, copiamos B[k] para tempB.
        // - Em seguida, percorremos todos os i e atualizamos C[i] usando A[i] * tempB.
        //
        // Vantagens:
        // Garantimos que a fatia do bloco B que será utilizada está na L2 (bom reuso temporal).
        // Loop interno (em i) costuma vetorizar bem e tem ótimo acesso contíguo a A e C.
        void produtoTilingHoistB(vector<double>& A, vector<double>& B, vector<double>& C, int N, int Bsize) {
            // Inicializa C uma única vez
            for (int i = 0; i < N; i++) {
                C[i] = 0.0;
            }

            // Varre B em blocos [start, end)
            for (int start = 0; start < N; start += Bsize) {
                int end = (start + Bsize < N) ? (start + Bsize) : N;

                // Para cada elemento do bloco de B...
                for (int k = start; k < end; k++) {
                    // temporário de B[k]: carrega uma vez e reutiliza
                    double tempB = B[k];

                    // Atualiza todos os C[i] com esse B[k]
                    for (int i = 0; i < N; i++) {
                        C[i] += A[i] * tempB;
                    }
                }
            }
        }
        
