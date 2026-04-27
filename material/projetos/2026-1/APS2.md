# APS 2 - Otimizações e Paralelismo em GPU

Um cliente precisa tomar decisões rápidas no mercado de ações. Para isso, ele utiliza simulações estocásticas para prever o comportamento futuro de um ativo com base em dados históricos.

O modelo utilizado pelo cliente é baseado no [movimento browniano geométrico](https://fiscomp.if.ufrgs.br/index.php/Movimento_Browniano_Geom%C3%A9trico_para_Previs%C3%A3o_no_Mercado_de_A%C3%A7%C3%B5es), implementado através da [equação de Black-Scholes](https://www.investopedia.com/terms/b/blackscholes.asp). A partir de um preço inicial e da volatilidade do mercado, ele simula milhares de cenários possíveis para o preço da ação ao longo do tempo.

Cada linha gerada representa um possível “caminho” do preço do ativo, enquanto a linha final média representa o comportamento mais provável segundo o [método de Monte Carlo](https://pt.wikipedia.org/wiki/M%C3%A9todo_de_Monte_Carlo).

No entanto, o sistema atual é extremamente lento, dificultando a tomada de decisões, especialmente em um ambiente onde o tempo de resposta é decisivo.

Seu trabalho será melhorar esse sistema.

[Acesse o seu repositório peli link do Classroom](https://classroom.github.com/a/h9hTmdSW) 
# **A data limite de entrega é 25/05/2026 ás 23h59**


A APS será avaliada em dois aspectos principais:

* **Implementação (código)** → 4 pontos
* **Relatório (.ipynb)** → 6 pontos


# Implementação (4 pontos)

A nota da implementação será atribuída de acordo com o nível máximo atingido.

## 2 pontos

### Requisitos de código:

* Pelo menos **uma função portada para GPU**
* Uso de **pelo menos uma técnica de otimização**

### Requisitos de execução:

* Arquivo SLURM corretamente configurado para submissão dos testes no **Cluster Franky**

### Configuração obrigatória nos testes:

* Laços Internos: 100
* Laços Externos: 50000
* Histórico (dias): 948
* Previsão (dias): 948
* Preço Inicial: 0.5
* Taxa de Risco: 0.5

### Observação:

Sem execução comprovada nessa configuração, **a pontuação não será atribuída**.

## 3 pontos

### Requisitos de código:

* Pelo menos **uma função portada para GPU**
* Uso de **pelo menos uma técnica de otimização**

### Requisitos de execução:

* Arquivos SLURM corretamente configurados para submissão dos testes nos dois clusters:

  * **Cluster Franky**
  * **Cluster Santos Dumont**
* Comparação de desempenho entre os dois ambientes

### Requisitos de análise:

* Deve evidenciar diferenças de desempenho entre as arquiteturas

### Configuração obrigatória:

* Laços Internos: 100
* Laços Externos: 80000
* Histórico (dias): 1422
* Previsão (dias): 1422
* Preço Inicial: 0.5
* Taxa de Risco: 0.5

### Observação:

Sem execução comprovada nessa configuração, **a pontuação não será atribuída**.


## 4 pontos

### Requisitos de código:

* Pelo menos **duas funções portadas para GPU**
* Uso de **duas ou mais técnicas de otimização**

### Requisitos de execução:

* Arquivos SLURM corretamente configurados para submissão dos testes nos dois clusters:

  * **Cluster Franky**
  * **Cluster Santos Dumont**


### Requisitos de análise:

* Deve demonstrar entendimento do impacto de:

  * arquitetura da GPU
  * memória
  * paralelismo
  * escalabilidade

### Configuração obrigatória:

* Laços Internos: 100
* Laços Externos: 100000
* Histórico (dias): 2370
* Previsão (dias): 2370
* Preço Inicial: 0.5
* Taxa de Risco: 0.5

### Observação:

Sem execução comprovada nessa configuração, **a pontuação não será atribuída**.



# Relatório (.ipynb) (6 pontos)

O relatório deve conter **os logs de execução, análise e explicação dos resultados**.

## Critérios de avaliação

| Conceito          | Estrutura                              | Integração e Visualização                         | Profundidade da Análise                                                                       |
| :---------------- | :------------------------------------- | :------------------------------------------------ | :-------------------------------------------------------------------------------------------- |
| **Até 2 pts** | Estrutura desorganizada ou fragmentada | Gráficos/tabelas sem conexão com o texto          | Descrição superficial                                                                         |
| **Até 4 pts** | Organização clara e técnica            | Visualizações próximas das explicações            | Relaciona resultados com conceitos básicos                                                    |
| **Até 6 pts** | Estrutura integrada (nível científico) | Narrativa fluida com dados sustentando argumentos | Explica desempenho com base em arquitetura (CPU vs GPU, memória, paralelismo, escalabilidade) |


## Requisitos mínimos do relatório

Para qualquer nota diferente de zero, o relatório deve conter:

* Descrição dos experimentos realizados
* Parâmetros utilizados (loops, tempo, etc.)
* Evidência de execução (prints, logs ou tabelas)
* Pelo menos:

  * **1 tabela**
  * **1 gráfico**

## Observações importantes

* Resultados sem evidência de execução **não serão considerados**
* Execuções fora das configurações obrigatórias **não pontuam**
* Análises inconsistentes com os dados **não serão considerados**

