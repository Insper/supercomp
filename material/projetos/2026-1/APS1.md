# APS 1 - Otimizações e Paralelismo em CPU

Cláudio, um motorista de caminhão de entregas possui uma lista de pontos em que deverá passar para deixar  encomendas. Não existe uma ordem definida, apenas a missão de realizar todas as entregas até o fim do dia. Interessado em realizar a tarefa com o menor custo possível, ele precisa encontrar a sequência de visitas que resulta no menor caminho. 

![Cada ponto vermelho é um endereço a ser visitado e a linha preta é o melhor trajeto](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/GLPK_solution_of_a_travelling_salesman_problem.svg/1280px-GLPK_solution_of_a_travelling_salesman_problem.svg.png)


O seu trabalho será encontrar o caminho que oferece o menor custo e fornecê-lo ao Claúdio. 

Durante as atividades da [Aula04](../../aulas/aula04/index.md) foi fornecido um exemplo de implementação de busca exaustiva considerando uma moto. A implementacão está correta, mas é claramente muito lenta e e precisa de algumas adaptações para resolver esta missão. Seu trabalho será:

1. Adaptar o exemplo para utilizar os pontos fornecidos em [pontos.txt](pontos.txt), considerando que o caminhão do Senhor Claúdio consegue **carregar 60 pacotes por vez**.
2. Otimizar o código de forma a gerar a rota para o Senhor Claúdio fazer as entregas no menor tempo possível.


# Avaliação

A APS será avaliada em 2 aspectos principais; 

* **A implementação (4 pontos)**
* **O relatório (6 pontos)**

## Implementação (4 pontos)

A implementação seguirá a seguinte rubrica. 

* **0** - Não compila, não é possível testar a implementação. 
* **+1** - Implementou alguma otimização que melhora o algoritmo exaustivo sequencial, mas ainda leva muito tempo para chegar na solução. 
* **+1** - Otimizou a heurísitca e conseguiu chegar na solução, mas ainda tem uma implementação sequencial. 
* **+2** - Paralelizou o algorítimo e conseguiu chegar na melhor solução no menor tempo possível.
    


## Relatório

O relatório deve apresentar evidências de que os experimentos foram executados e analisados pelo aluno.
A avaliação será feita de acordo com os critérios abaixo:


### 1. Reprodutibilidade dos experimentos (1 ponto)

O relatório deve permitir que qualquer pessoa entenda exatamente como os experimentos foram realizados e o como reproduzir os testes.

* **0** - Não é possível entender como realizar os testes nem como eles foram executados.

* **+1** - O repositório disponibiliza os arquivos SLURM utilizados e todos os arquivos necessários para realizar os testes, há uma descrição clara dos testes realizados e como reproduzir os experimentos.


### 2. Evidência de execução dos experimentos (2 pontos)

O relatório deve apresentar resultados obtidos a partir da execução do código.
Esses resultados devem ser apresentados em formas de tabela e gráficos.

* **0 pontos** - O relatório não apresenta resultados em forma de tabelas e gráficos.

* **+0.5 ponto** - Existe a apresentação dos dados em tabela **ou** gráfico.

* **+1.5 ponto** - O relatório apresenta tabela **e** gráfico contendo os principais parâmetros de execução dos experimento.

### 3. Qualidade da análise dos resultados (3 pontos)

A análise deve mostrar que o aluno entendeu as otimizações que realizou e como elas impactaram os resultados obtidos.

* **0 pontos** - Texto do relatório genérico, análises claramente geradas por ferramentas de IA ou explicações inconsistentes com os resultados apresentados.

* **+1 ponto** - O relatório menciona resultados, mas a interpretação ainda é superficial.

* **+2 pontos** - O relatório relaciona as otimizações implementadas com os dados obtidos de forma clara e objetiva.


# A entrega da APS 1 deverá ser realizada até o dia 20/03 as 23h59 pelo [Github Classroom](https://classroom.github.com/a/YdsvpTBm) 