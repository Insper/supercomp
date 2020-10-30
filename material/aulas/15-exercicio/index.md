# 15 - Exercício prático

Nesta aula final de de paralelismo multi-core vamos fazer um exercício aberto. A única restrição é criar uma implementação do código no arquivo *pi_recursivo.cpp* usando tarefas e uma usando `for` paralelo. Esse roteiro tem apenas algumas perguntas que visam orientar seu trabalho. 

!!! example
    Examine o código. Procure entender bem o que está acontecendo antes de prosseguir. 

!!! question medium
    O código tem efeitos colaterais? Existem variáveis que poderiam ser compartilhadas de maneira não intencional? Se sim, você conseguiria refatorar o código para minimizar ou até eliminar esses efeitos colaterais?

!!! question medium
    Quantas níveis de chamadas recursivas são feitas? Quando o programa para de chamar recursivamente e faz sequencial?

## Tarefas 

!!! example
    Crie uma implementação do *pi_recursivo* usando tarefas. Meça seu tempo e anote.

!!! question short
    Os ganhos de desempenho foram significativos?

!!! question short
    Quantas tarefas foram criadas? Você escolheu essa valor como?

!!! example
    Tente números diferentes de tarefas e verifique se o desempenho melhora ou piora. Anote suas conclusões abaixo. 

## `for` paralelo

!!! example
    Crie uma implementação do *pi_recursivo* usando for paralelo. Meça seu tempo e anote.

!!! question short
    Os ganhos de desempenho foram significativos?

!!! question short
    Como você fez o paralelismo? Precisou definir o número do `for` manualmente ou conseguiu realizar a divisão automaticamente? Comente abaixo sua implementação.

## Comparação

!!! question short
    Compare seus resultados das duas abordagens. Anote abaixo seus resultados.

!!! warning
    É possível conseguir tempos muito parecidos com ambas, então se uma delas ficou muito mais lenta é hora de rever o que foi feito. 