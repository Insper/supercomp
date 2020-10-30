# 14 - Efeitos colaterais

Nesta aula iremos aprender a controlar os efeitos colaterais de programas paralelos. Nossa estratégia se baseará na ideia de que o resultado de um programa paralelo deverá se manter igual em toda execução, a não ser que explicitamente seja pedido o contrário. 

## Geradores de números aleatórios

Na última aula vimos que nossa implementação de `discard` era `O(N)`, o que tornava nosso algoritmo $O(N^2)$ e invalidava qualquer ganho vindo do paralelismo. Vamos então abdicar de uma implementação literalmente igual à sequencial e tomar como boa uma implementação que, mantendo fixo o número de threads, retorna sempre os mesmos resultados. Ou seja, adicionamos como parâmetro do programa o número de threads a serem usadas. Se mantivermos os mesmos parâmetros (agora incluindo número de threads) os resultados serão mantidos.

Claramente não é o ideal, mas é suficiente se formos explícitos nessa condição. Lembrando que um dos nossos objetivos é ganhar desempenho mantendo a previsibilidade e a reprodutibilidade.

Nossa ideia será então,

**criar um gerador de números aleatórios por thread.**

!!! question short
    Explique por que esta ideia controla os efeitos colaterais. Quais funcionalidades do OpenMP nos ajudariam a fazer isto?

    ??? details "Resposta"
        Ao criar um gerador para cada thread evitamos o acesso desordenado a um único gerador. Com somente um gerador global precisaríamos forçar uma ordem completa para ter resultados reprodutíveis (já que não controlamos quando cada thread roda). Com um gerador por thread conseguimos garantir que cada thread verá sempre a mesma sequência de números. Faltaria só garantir que cada thread executa sempre as mesmas iterações do loop. Isso pode ser feito usando o agendamento `static` do `for` paralelo. 


!!! example 
    Implemente a ideia descrita acima. Você pode quebrar nos seguintes passos:

    1. criar um gerador para cada thread. Use `omp_get_max_threads` para saber o máximo de threads criáveis durante a execução.
    2. use uma `seed` diferente para cada gerador. Você pode usar alguma conta relacionada ao `id` da thread para isto.
    3. dentro do loop, acesse somente o gerador da thread em execução. Veja como pegar esse *id* nos exemplos da aula passada
    
!!! question short
    Execute diversas vezes o programa acima. Os resultados se mantém? Use a *imagem1.pgm* como teste. 
    
!!! question short 
    Teste agora rodando com 3 threads. Os resultados se mantém? Eles são iguais ao item anterior?

## Sincronização com regiões críticas

Na expositiva vimos vários conceitos. Vamos agora ver sua execução no OpenMP. O snippet abaixo mostra como indicar um bloco de código como seção crítica. 

```cpp
#pragma omp critical
{
    // só uma thread entra por vez
}
```

Os principais usos são:

1. trabalhar com arquivos e entrada/saída. Muito comum para acessar o terminal ou para arquivos de log;
2. usar classes ou estruturas de dados complexas. Acessos de escrita a vetores ou dicionários, por exemplo;
3. realizar operações lentas que não podem ser interrompidas.

!!! example
    Abra novamente o `exemplo1.cpp` da aula passada. Ao executá-lo é comum que os prints saiam embaralhados. Use `critical` para ter acesso exclusivo ao terminal e evitar isso. 
    

Vamos agora aprender exatamente como usar variáveis compartilhadas e a diretiva `critical` para fazer o mesmo trabalho que a direriva `reduction`. Apesar de estarmos teoricamente reproduzindo um trabalho já bem implementado, temos duas vantagens pedagógicas:

1. conseguimos conferir a qualidade de nossos resultados rapidamente
2. as técnicas usadas são aplicáveis também para paralelismo de tarefas
3. `reduction` só funciona dentro de um `for` paralelo em que a variável for um tipo básico. Podemos precisar de reduções com mais de uma variável ou de um `struct`, por exemplo.

!!! example
    Para fazermos comparações e deixarmos nossos experimentos organizados, crie uma função `pi_sequencial(long num_steps)` que calcula o pi sequencialmente.
    
Nos próximos exercícios iremos criar versões paralelas do cálculo do `pi` explorando construções de sincronização do OpenMP para melhorar nosso desempenho.

!!! question short
    Um dos problemas de não usarmos a cláusula `reduction` é que acessos simultâneos a `sum` estragam seu valor (pois `+=` não é representado por apenas uma instrução em Assembly). Explique como `critical` pode ser usado para resolver este problema. 

!!! example
    Crie uma função `pi_omp_critical1` que usa `critical` da maneira explicada acima para resolver os problemas de compartilhamento. Verifique que os resultados numéricos continuam iguais.
    
!!! question short
    Você consegue explicar o desempenho do programa acima? 
    
    ??? details "Resposta"
        A utilização de `critical` é cara. Cada vez que uma thread chega na região crítica ela gasta tempo de processamento. Ou seja, entrar na região crítica muito frequentemente resulta em gastar tempo precioso em sincronização ao invés de gastar resolvendo nosso problema.

!!! tip 
    Sincronização é caro e é um custo relevante de qualquer projeto de alto desempenho que não seja ingenuamente paralelizável. Nosso objetivo é sempre quere **minimizar** a sincronização feita pelas threads. 
    
Tendo em vista o quadro acima, faça os próximos exercícios.

!!! question short
    Na parte 1 usamos a estratégia de criar uma cópia do gerador de números aleatórios para cada thread. Como poderiamos adaptar esta ideia para evitar o uso da região crítica?
    
    ??? details "Resposta"
        Basta novamente criar uma cópia para cada thread e acumular os resultados na cópia. Essa alocação deverá ser feita antes do início da região paralela. Ao finalizar podemos simplesmente somar as somas parciais e tudo estará OK. 
    
!!! example
    Implemente a ideia acima e chame sua função de `pi_omp_copias_parciais`. Verifique que os valores continuam iguais.

!!! question short
    Meça o tempo acima e verifique que houve ganho de desempenho em relação ao uso de `critical`. 

A solução acima é funcional mas inconveniente: precisamos ficar alocando arrays a cada região paralela e depois ficar somando tudo no final. Podemos eliminar  esse vetor separando as diretivas `parallel` e `for` e tornar nosso código mais enxuto e legível. Veja abaixo um exemplo.

```cpp
#pragma omp parallel
{
    double local;
    #pragma omp for
    for (...) {
        // cada thread mexe na sua cópia de local
    }
}
```

!!! question short
    Como você usaria a estrutura acima para chamar `critical` somente uma vez por thread? Explique por que sua solução funciona e por que ela deverá trazer grandes ganhos de desempenho.

!!! example 
    Implemente sua função como `pi_omp_critical_local` e verifique que os resultados continuam corretos.

!!! question short
    Meça novamente o tempo de execução. Como essa função se compara com as outras implementações?
    
!!! done
    Finalizamos esta atividade com a seguinte conclusão: **sincronização é cara, mas por vezes é o melhor que temos**. Se precisamos fazer uma operação que torna um bloco de código uma **região crítica**, então não temos opção e tudo o que podemos é tornar essa região **o menor possível** e somente entrar nela **quando for estritamente necessário**.
    
