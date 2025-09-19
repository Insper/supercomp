
## Questão 1 — Teórica (Heurísticas e Aleatoriedade)
**Tipo:** múltipla escolha (múltiplas corretas)  
**Enunciado:**  
Sobre heurísticas com aleatoriedade em problemas de otimização:  

a) Uma heurística sempre garante encontrar a solução ótima.  
b) O uso de aleatoriedade pode ajudar a escapar de mínimos locais.  
c) Estratégias puramente determinísticas podem explorar repetidamente as mesmas regiões do espaço de busca.  
d) Uma heurística aleatória nunca pode ser menos eficiente que uma busca determinística.  

??? note "Ver resposta"
    a) Incorreta. Heurísticas **não garantem a solução ótima**, mas sim uma **boa solução em tempo razoável**. 

    b) Correta. Técnicas como Algoritmos Genéticos usam aleatoriedade justamente para evitar que a busca fique presa em soluções subótimas.

    c) Correta. Sem variação aleatória, a busca pode focar em regiões já visitadas, perdendo diversidade na exploração.

    d) Incorreta. O uso de aleatoriedade não garante eficiência. Pode inclusive aumentar o custo (mais iterações, soluções piores) se mal calibrada. A eficiência depende da implementação.



## Questão 2 — Busca linear vs aleatória em vetor
**Enunciado:**  
Implemente duas funções para encontrar um valor `alvo` em um vetor:  

- **Versão linear**: percorre o vetor de `0` até `N-1`.  
- **Versão aleatória**: sorteia índices aleatórios até encontrar o alvo (ou até `maxTentativas`).  

Depois:  
- Compile no cluster.  
- **Crie um script SLURM** para rodar ambas versões.  
- Compare o tempo e número de tentativas de cada abordagem.  

```cpp
int busca_linear(const std::vector<int>& v, int alvo);
int busca_aleatoria(const std::vector<int>& v, int alvo, unsigned long long maxTentativas);
```

??? note "Ver resposta"

    `busca_linear` percorre **sequencialmente** → excelente **localidade espacial**.  
    `busca_aleatoria` tem **acessos aleatórios** → pouca localidade, alta variância; pode repetir índices.  
     Com o vetor `v[i]=i`, a **linear encontra rápido** (especialmente se `alvo` for pequeno); a aleatória pode demorar mesmo com alvo “próximo”.  

        #include <vector>
        #include <random>

        // ---------------------------------------------------------
        // Função: busca_linear
        // Objetivo: procurar o valor 'alvo' de forma sequencial
        // Estratégia: percorre o vetor do início ao fim (índices 0..N-1)
        // ---------------------------------------------------------
        int busca_linear(const std::vector<int>& v, int alvo) {
            // Percorre todos os elementos do vetor
            for (size_t i = 0; i < v.size(); i++) {
                // Se encontrou o alvo, retorna o índice
                if (v[i] == alvo) return (int)i;
            }
            // Se chegou até aqui, o alvo não existe no vetor
            return -1;
        }

        // ---------------------------------------------------------
        // Função: busca_aleatoria
        // Objetivo: procurar o valor 'alvo' de forma aleatória
        // Estratégia: sorteia índices ao acaso até achar o alvo
        // ou até atingir o limite de tentativas (maxTentativas).
        // ---------------------------------------------------------
        int busca_aleatoria(const std::vector<int>& v, int alvo, unsigned long long maxTentativas) {
            // Caso o vetor esteja vazio, não há o que buscar
            if (v.empty()) return -1;

            // Gerador de números aleatórios
            std::random_device rd;        // fonte de entropia (pode variar a cada execução)
            std::mt19937 gen(rd());       // gerador pseudo-aleatório (Mersenne Twister)
            std::uniform_int_distribution<size_t> dist(0, v.size() - 1); 
            // distribuição uniforme de índices válidos [0, N-1]

            // Tenta encontrar o alvo até atingir o número máximo de tentativas
            for (unsigned long long t = 0; t < maxTentativas; t++) {
                size_t idx = dist(gen);   // sorteia um índice válido
                if (v[idx] == alvo) {
                    // se encontrou, retorna o índice
                    return (int)idx;
                }
            }
            // Se não encontrou dentro do limite de tentativas, retorna -1
            return -1;
        }

    Script SLURM

        #!/bin/bash
        #SBATCH --job-name=busca            # nome do job
        #SBATCH --output=busca.%j.txt       # saída em arquivo
        #SBATCH --time=00:05:00             # tempo limite
        #SBATCH --nodes=1                   # 1 nó
        #SBATCH --ntasks=1                  # 1 tarefa
        #SBATCH --cpus-per-task=1           # 1 CPU
        #SBATCH --partition=express         # ou 'normal', se preferir
        #SBATCH --mem=1GB                   # memória

        echo "=== Rodada 1: N=1e6 alvo=500000 maxTent=1e6 seed=42 ==="
        ./busca 1000000 500000 1000000 42
        echo "======================================================="

        echo "=== Rodada 2: N=1e6 alvo=10 maxTent=1e6 seed=123 ==="
        ./busca 1000000 10 1000000 123
        echo "===================================================="

        echo "=== Rodada 3: N=1e6 alvo=999999 maxTent=1e6 seed=777 ==="
        ./busca 1000000 999999 1000000 777
        echo "======================================================="

        echo "=== Rodada 4: N=1e6 alvo=37 (usa N/2) maxTent=1e6 seed=2025 ==="
        ./busca 1000000 37 1000000 2025
        echo "============================================================="


## Questão 3 — Estratégia híbrida (busca sequencial + aleatória)
**Enunciado:**  
Implemente uma função que:  
- Primeiro verifica os **K primeiros elementos** do vetor sequencialmente.  
- Se não encontrar, passa a buscar usando índices aleatórios até `maxTentativas`.  

Depois:  
- Compile e rode no cluster com SLURM.  
- Compare os resultados com as funções da Questão 2.  

```cpp
int busca_hibrida(const std::vector<int>& v, int alvo, int K, unsigned long long maxTentativas);
```

??? note "Ver resposta"
   
        #include <vector>
        #include <random>

        // ---------------------------------------------------------
        // Função: busca_hibrida
        // Objetivo: combinar duas estratégias de busca:
        //   (1) varre sequencialmente os K primeiros elementos (bom p/ localidade)
        //   (2) se não achar, usa tentativas aleatórias até maxTentativas
        //
        // Quando é útil?
        // - Se o alvo tem maior probabilidade de estar no início do vetor,
        //   a parte sequencial encontra rápido.
        // - Caso contrário, a fase aleatória pode "acertar" em elementos distantes
        //   sem percorrer todo o vetor sequencialmente.
        // ---------------------------------------------------------
        int busca_hibrida(const std::vector<int>& v, int alvo, int K, unsigned long long maxTentativas) {
            // -------- Parte 1: busca sequencial nos primeiros K elementos --------
            // Varre de 0 até K-1 (limitando também pelo tamanho do vetor).
            // Vantagem: acesso contíguo → melhor localidade de memória.
            for (int i = 0; i < K && i < (int)v.size(); i++) {
                if (v[i] == alvo) return i;  // achou no prefixo
            }

            // -------- Parte 2: busca aleatória no vetor inteiro --------
            // Observação: esta versão sorteia em [0, N-1]. Em muitos casos,
            // restringir a [K, N-1] faz mais sentido (evita repetir o prefixo já checado).
            if (v.empty()) return -1;        // vetor vazio → não há o que buscar

            // Geradores para sorteio de índices válidos
            std::random_device rd;           // fonte de entropia (não determinística)
            std::mt19937 gen(rd());          // PRNG (Mersenne Twister)
            std::uniform_int_distribution<size_t> dist(0, v.size() - 1); // sorteia 0..N-1

            // Tenta até atingir o limite de tentativas aleatórias
            for (unsigned long long t = 0; t < maxTentativas; t++) {
                size_t idx = dist(gen);      // sorteia um índice
                if (v[idx] == alvo) {        // compara com o alvo
                    return (int)idx;         // achou durante a fase aleatória
                }
            }

            // Não encontrou nem na parte sequencial, nem na aleatória
            return -1;
        }




## Questão 4 — Heurística com pré-filtro
**Enunciado:**  
Implemente uma função de busca que usa uma **heurística simples de pré-filtro**:  
- Antes de acessar `v[i]`, só considere o índice se `i % 2 == 0` (ou seja, só olha posições pares).  
- Se não encontrar após `maxTentativas`, faça busca linear completa como fallback.  

Depois:  
- Compile e rode no cluster com SLURM.  
- Compare desempenho e número de acessos com as versões anteriores.  

```cpp
int busca_com_filtro(const std::vector<int>& v, int alvo, unsigned long long maxTentativas);
```

??? note "Ver resposta"
    
    A busca aleatória é **restrita a índices pares**, tentando reduzir acessos.  
    Caso não encontre dentro de `maxTentativas`, entra uma busca linear 
    O pré-filtro pode ser vantajoso se há **maior probabilidade** de o alvo estar em posições pares.  
    Porém, pode desperdiçar tentativas (índices ímpares descartados) e no pior caso cair na busca linear.


        #include <vector>
        #include <random>

        int busca_com_filtro(const std::vector<int>& v, int alvo, unsigned long long maxTentativas) {
            if (v.empty()) return -1;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<size_t> dist(0, v.size() - 1);

            // Heurística: só verifica índices pares
            for (unsigned long long t = 0; t < maxTentativas; t++) {
                size_t idx = dist(gen);
                if (idx % 2 != 0) continue;       // ignora índices ímpares
                if (v[idx] == alvo) return (int)idx;
            }

            // Fallback: se não achou, faz busca linear completa
            for (size_t i = 0; i < v.size(); i++) {
                if (v[i] == alvo) return (int)i;
            }
            return -1; // não encontrado
        }

