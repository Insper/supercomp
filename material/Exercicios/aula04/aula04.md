# Qual a principal vantágem da técnica "Branch & Bound" em relação a busca
exaustiva? 

a. Ela Reduz o espaço de busca ao interromper caminhos que já ultrapassaram o
melhor custo atual. 

b. Ela transforma o problema de otimização em um problema de busca linear
simples. 

c. Ela garante encontrar uma solição melhor que solução ótima teórica. 

d. Ela utiliza de aleatoriedade para saltar entre diferentes partes dos grafo de
caminhos testando diversas soluções. 

# Sobre os algoritmos que aplicam a técnica de Hill Climbing:

a. É garantido que aquela é a melhor solução na vizinhança imediada. 

b. O algoritimo sempre irá encontra a solução ótima para o problema. 

c. Aumentar o numero de iterações do algorimo sempre irá melhorar o resultado. 

d. É um algoritimo muito custoso e, a depender da situação não vale a pena. 

# Nas aulas foram apresentadas as heurísticas de "Branch & Bound", "Hill
Climbing" e Aleatoriedade. Encontre uma heurística "nova", e explique como ela
pode nos ajudar no problema do caixeiro viajante.



# Na busca exaustiva, além da poda por custo (Branch and Bound), podemos
podar por viabilidade. Imagine um problema de entrega onde cada pacote tem um
peso e o veículo tem uma capacidade máxima.

Complete o código da busca exaustiva para interromper a exploração de um ramo
assim que a capacidade for excedida:



```cpp

void busca_com_capacidade(int atual, int peso_acumulado, double custo_atual) {
    if (caminho_completo()) {
        atualizar_melhor_solucao(custo_atual);
        return;
    }
}
```



# O Problema das N-Rainhas

O problema das N-Rainhas é um desafio clássico de lógica e otimização
combinatória que consiste em posicionar N rainhas num tabuleiro de xadrez de
dimensão N×N de forma a que nenhuma delas consiga atacar outra.

Abaixo temos as funções que serão a base dos próximos exercícios: 

```cpp

struct Solucao {
    vector<int> rainhas; // rainhas[coluna] = linha
    int ataques;
};

/* Retorna o número total de conflitos (pares de rainhas que se atacam).
 * Uma solução com 0 ataques é uma solução válida para o problema. */
int calcular_ataques(const vector<int>& tabuleiro);
Solucao gerar_configuracao_aleatoria(int n);
Solucao executar_hill_climbing(Solucao s);
```

## Vizinhança 

Para otimizar o posicionamento das rainhas, o Hill Climbing precisa de uma
estratégia de "vizinhança". Uma estratégia comum é o swap (troca): escolhemos
duas colunas e trocamos as linhas das rainhas entre elas.

Na função gerar_vizinho realize o swap entre `atual.rainhas[i]` e `atual.rainhas[j]`
Calcule o novo número de ataques usando a função pronta
Se o novo estado for melhor ou igual, mantenha. Se for pior, desfaça o swap.

```cpp

void gerar_vizinho(Solucao& atual) {
    int n = atual.rainhas.size();
    int i = rand() % n;
    int j = rand() % n;

}
```

## Mínimos Locais
O Hill Climbing puro frequentemente fica preso em "mínimos locais", uma forma
que já vimos de minimizar esse problema é utilizar reinicios aleatórios. 

- Implementar o reinício com aleatorieadade (ponto de partida);
- Refinar a partir deste ponto de partida;
- Atualizar valores com a melhor solução;
- Possiveis melhorias de eficiências;

```cpp
Solucao resolver_n_rainhas_com_restart(int n, int max_tentativas) {

    return melhor_global;
}
```



<!-- # Sudoku -->
<!---->
<!-- Resolução de Sudoku é frequentemente abordada como um Problema de Satisfação de -->
<!-- Restrições (CSP) que serve para ilustrar a eficiência de algoritmos de -->
<!-- backtracking e a importância de podas (pruning) no espaço de busca. Embora um -->
<!-- tabuleiro 9x9 pareça simples, variações maiores (como 16x16 ou 25x25) tornam o -->
<!-- custo computacional proibitivo para buscas exaustivas. exigindo que se use  -->
<!-- heurísticas diversas, e técnicas de paralelismo de tarefas para explorar -->
<!-- diferentes soluções.  -->
<!---->
