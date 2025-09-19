 

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

## Questão 2 — Geração de dados aleatórios em C++

**Enunciado:**  
Complete a função `gerar_leituras` para criar um vetor de tamanho `N` contendo valores `double` aleatórios entre 12.0 e 189.98, usando `std::vector`, `std::mt19937` e `std::uniform_real_distribution`.

```cpp
// =========================================
// Função para gerar um vetor com valores aleatórios
// =========================================
vector<double> gerar_leituras(size_t tamanho) {
    // TODO: Criar um vetor de tamanho `tamanho`
    // TODO: Criar gerador de números aleatórios com seed fixa (ex: 42)
    // TODO: Definir distribuição entre 12.0 e 189.98
    // TODO: Preencher o vetor com números aleatórios

    return {}; // Substitua pelo vetor preenchido
}
```

??? note "Ver a resposta"

        #include <vector>
        #include <random>
        using std::vector;

        // =========================================
        // Função para gerar leituras aleatórias
        // Preenche o vetor passado por referência
        // =========================================
        void gerar_leituras_ref(vector<double>& v) {
            // Criamos o gerador Mersenne Twister com seed fixa
            std::mt19937 gen(42);

            // Distribuição uniforme no intervalo [12.0, 189.98]
            std::uniform_real_distribution<double> dist(12.0, 189.98);

            // Percorremos o vetor pelo índice (sem auto)
            for (size_t i = 0; i < v.size(); i++) {
                v[i] = dist(gen);  // gera número e armazena direto no vetor
            }
        }


---

## Questão 3 — Média móvel por valor e por referência

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

## Questão 4 — Média móvel por ponteiro

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
