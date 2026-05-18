
## Questão 1 — Teórica (Hierarquia de Memória e Localidade)
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

## Questão 2 — Versão Ingênua (Cálculo de Produto Vetorial)
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

## Questão 3 — Versão com Tiling (Multiplicação de Blocos de Vetores)
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

## Questão 4 — Reordenação de Loops + Flags de Otimização
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
        
