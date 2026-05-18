# Métodos de Busca Heurística 

As heurísticas de busca são estratégias para encontrar soluções em problemas difíceis sem precisar explorar todo o espaço de possibilidades. Elas não garantem o resultado ótimo, mas entregam respostas úteis com um custo de tempo menor. Temos aqui algumas famílias de heurísticas e uma implementação em pseudocódigo para servir de inspiração, mas não se sinta satisfeito apenas com o que temos aqui, existem inúmeras heuristicas que não estão descritas aqui.

## Heurísticas construtivas

A ideia é construir passo a passo a solução seguindo uma regra de decisão. Essa regra pode ser gulosa (sempre escolher o melhor incremento local) ou conter aleatoriedade para gerar diversidade. No cacheiro viajante, por exemplo, a cidade mais próxima ainda não visitada é escolhida; em mochila, o iten é escolhido por melhor relação valor/peso até atingir a capacidade máxima da mochila.

```python
def heuristica_construtiva():
    S = solucao_vazia()
    while not solucao_completa(S):
        candidatos = candidatos_admissiveis(S)
        # regra gulosa: minimiza custo incremental 
        c_escolhido = min(candidatos, key=lambda c: custo_incremental(S, c))
        S = inserir(S, c_escolhido)           # atualizar estado parcial
    return S
```

## Busca local

Partimos de uma solução inicial e tentamos melhorá-la explorando “vizinhos” obtidos por pequenas alterações. Se um vizinho melhora o custo, aceitamos a mudança e repetimos o processo até não existir mais melhora. 

```python
def busca_local(S0):
    S = S0
    while True:
        melhorou = False
        for S_viz in gerar_vizinhos(S):
            if custo(S_viz) < custo(S):
                S = S_viz          # estratégia "first improvement"
                melhorou = True
                break
        if not melhorou:
            break
    return S
```

## Heurísticas baseadas em construção

Combinam construção e refino: primeiro geramos uma boa solução inicial de forma gulosa ou com aleatoriedade e, em seguida, aplicamos busca local para polir o resultado. 

```python
def construcao_mais_refino():
    S = heuristica_construtiva()  # ou uma variação com aleatoriedade (RCL)
    S = busca_local(S)            # refino por movimentos locais
    return S
```

## Heurísticas baseadas em modificação

Aqui trabalhamos sobre uma solução existente aplicando uma perturbação para escapar de ótimos locais. Após perturbar, refinamos novamente com busca local. Esse padrão é a base do ILS (Iterated Local Search) e do VNS (Variable Neighborhood Search).

```python
def modificacao_baseada(S_inicial, max_iter=100):
    S_best = S_inicial
    for _ in range(max_iter):
        S_pert = perturbar(S_best, intensidade=ajustar_intensidade())
        S_ref = busca_local(S_pert)
        if custo(S_ref) < custo(S_best):
            S_best = S_ref
    return S_best
```

## Heurísticas baseadas em recombinação

Inspiradas em algoritmos evolutivos, combinam duas (ou mais) soluções “pais” para gerar uma nova solução “filha”, aproveitando blocos de boa qualidade de cada pai. Depois, refinam a filha com busca local. 

```python
def recombinacao_baseada(populacao, criterio_parada):
    P = inicializar_populacao(populacao)
    while not criterio_parada(P):
        A, B = selecionar_pais(P)             # torneio, roleta, ranking etc.
        C = recombinar(A, B)                  # OX/PMX/path-relinking/união+reparo
        C = busca_local(C)                    # passo memético (refino)
        P = atualizar_populacao(P, C)         # elitismo/substituição
    return melhor_solucao(P)
```

## Hibridização de heurísticas

Misturamos estratégias para buffar o algorítmo: construção gulosa aleatória para diversidade, seguida de busca local para intensificação, e ILS para escapar de ótimos locais. 

```python
def hibrida_grasp_ils(max_iter=50):
    S_best = None
    for _ in range(max_iter):
        S = construtiva_aleatoria_com_RCL()   # construção com lista restrita de candidatos
        S = busca_local(S)                    # intensificação
        S = ILS(S)                            # diversificação controlada
        if S_best is None or custo(S) < custo(S_best):
            S_best = S
    return S_best

def ILS(S, limite_sem_melhora=10):
    sem_melhora = 0
    while sem_melhora < limite_sem_melhora:
        S_p = perturbar(S)                    # ex.: ruin&recreate, 3-opt forte, shake(VNS)
        S_p = busca_local(S_p)
        if custo(S_p) < custo(S):
            S = S_p
            sem_melhora = 0
        else:
            sem_melhora += 1
    return S
```

## Híper-heurísticas

Em vez de desenhar uma única heurística, criamos um “orquestrador” que escolhe dinamicamente qual heurística de baixo nível aplicar a cada momento, com base em desempenho observado. Assim, o sistema alterna entre operadores como 2-opt, swap e reinserção, aprendendo quais funcionam melhor ao longo da execução.

```python
from math import sqrt, log
import random

def hiper_heuristica(S0, heuristicas, T, epsilon=0.1):
    # heuristicas: lista de funções do tipo h(S) -> S'
    estat = {h: {"ganho": 0.0, "usos": 0} for h in heuristicas}
    S = S0
    for t in range(1, T + 1):
        if random.random() < epsilon:
            h = random.choice(heuristicas)    # exploração
        else:
            h = selecionar_por_ucb(estat, t)  # exploração vs. exploração

        S_novo = h(S)
        ganho = max(0.0, custo(S) - custo(S_novo))
        estat[h]["ganho"] += ganho
        estat[h]["usos"]  += 1

        if custo(S_novo) < custo(S):
            S = S_novo
    return S

def selecionar_por_ucb(estat, t):
    # UCB1 simples: média + sqrt(2*ln(t)/n)
    melhor_h, melhor_score = None, float("-inf")
    for h, info in estat.items():
        n = max(1, info["usos"])
        media = info["ganho"] / n
        bonus = sqrt(2.0 * log(max(2, t)) / n)
        score = media + bonus
        if score > melhor_score:
            melhor_h, melhor_score = h, score
    return melhor_h
```

Entendi! Você quer exemplos de **heurísticas aleatórias** (estratégias que exploram o espaço com escolhas ao acaso) e de **heurísticas com filtro** (estratégias que usam algum critério para selecionar candidatos, descartando opções ruins). Vou explicar de forma didática e depois te dar exemplos em pseudocódigo estilo Python.

---

## Heurísticas aleatórias

A ideia aqui é simples: em vez de sempre escolher o “melhor” próximo passo, escolhemos **aleatoriamente** uma opção, ou entre todas, ou entre um subconjunto. Isso permite explorar soluções diferentes e fugir de caminhos muito determinísticos.

```python
import random

def heuristica_aleatoria(items, capacidade):
    mochila = []
    peso = 0
    while True:
        candidatos = [i for i in items if peso + i.peso <= capacidade]
        if not candidatos:
            break
        escolhido = random.choice(candidatos)   # sorteia qualquer candidato viável
        mochila.append(escolhido)
        peso += escolhido.peso
        items.remove(escolhido)
    return mochila
```

## Heurísticas com filtro 

Aqui, em vez de aceitar qualquer candidato, aplicamos um **filtro** para limitar as opções a um subconjunto de candidatos considerados “bons”. Depois, escolhemos um deles (às vezes aleatoriamente, às vezes pelo melhor custo). Essa ideia é a base do **GRASP**.

```python
import random

def heuristica_com_filtro(items, capacidade, alpha=0.3):
    mochila = []
    peso = 0
    while True:
        candidatos = [i for i in items if peso + i.peso <= capacidade]
        if not candidatos:
            break
        # filtro: mantém apenas candidatos dentro do top α (percentil de valor/peso)
        ratio = [i.valor / i.peso for i in candidatos]
        limite = min(ratio) + alpha * (max(ratio) - min(ratio))
        RCL = [i for i in candidatos if i.valor / i.peso >= limite]
        
        escolhido = random.choice(RCL)   # escolhe entre bons candidatos
        mochila.append(escolhido)
        peso += escolhido.peso
        items.remove(escolhido)
    return mochila
```

