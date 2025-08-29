# Efeitos Colaterais do Paralelismo

Nesta atividade vamos explorar os problemas que aparecem quando paralelizamos de forma ingênua e ver como corrigir logo em seguida. 

## TAREFA A: Transformação elemento-a-elemento (map)

Na aula passada foi pedido que você paralelizasse e analisasse este laço:

```cpp
#pragma omp parallel for schedule(runtime)
for (int i = 0; i < N; i++) {
    c[i] = alpha * a[i] + beta;
}
```

* Você executou com 1, 2, 4 e 8 threads.
* Testou diferentes `OMP_SCHEDULE` (`static`, `dynamic`, `guided`).
* Observou que o **resultado não muda nunca**, apenas o **tempo de execução** varia.

Este é um exemplo de paralelismo **seguro**, pois cada iteração escreve em posições diferentes do vetor `c`.

**Pergunta para pensar:** por que este laço é naturalmente paralelizável sem dar problemas?


## TAREFA B: Soma (redução) da norma L2 parcial

Outro laço analisado na aula passada foi:

```cpp
double soma = 0.0;
#pragma omp parallel for schedule(runtime)
for (int i = 0; i < N; i++) {
    soma += static_cast<double>(c[i]) * static_cast<double>(c[i]); // <- condição de corrida
}
```

Neste caso os valores da soma ficaram inconsistentes.
Isso acontece porque várias threads tentam atualizar a mesma variável ao mesmo tempo → condição de corrida (*race condition*).

**Pergunta para pensar:** se for pedido ao SLURM cpus-per-task=1 não vemos inconsistências no resultado da conta?


## Corrigindo com reduction

A correção é simples: usar uma redução.

```cpp
double soma = 0.0;
#pragma omp parallel for schedule(runtime) reduction(+:soma)
for (int i = 0; i < N; i++) {
    soma += (double)c[i] * c[i]; // <- corrigido
}
```

Agora cada thread acumula uma soma local e no final todas são combinadas.
O resultado fica estável e correto em qualquer número de threads.


## Dependência de dados

Nem todos os laços podem ser paralelizados:

```cpp
for (int i = 1; i < N; i++) {
    a[i] = a[i-1] + 1; // depende da iteração anterior
}
```

Aqui há uma dependência entre iterações: para calcular `a[i]` é necessário já ter calculado `a[i-1]`.
Paralelizar assim gera resultado incorreto.

Só é possível resolver reformulando o algoritmo. O objetivo é apenas perceber que nem todo loop é paralelizável.

Maaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaas se observarmos bem, esse cálculo é apenas uma progressão aritmética:

* `a[1] = a[0] + 1`
* `a[2] = a[0] + 2`
* `a[3] = a[0] + 3`
* …
* `a[i] = a[0] + i`

Ou seja, o valor de `a[i]` não precisa necessariamente `a[i-1]`, pode ser calculado diretamente.

### Versão paralelizável:

```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 1; i < N; i++) {
    a[i] = a[0] + i;
}
```

O loop original tem dependência sequencial, mas ao analisar o padrão, vemos que é uma progressão. Reformulando o cálculo, eliminamos a dependência, agora cada iteração é independente e pode ser distribuída entre threads sem problemas.
Esse exemplo é ótimo para mostrar que paralelizar não é só usar `#pragma`, às vezes é preciso pensar no algoritmo.


## Recursão com tasks

O OpenMP também permite paralelizar recursão, mas com cuidado.

Exemplo: Fibonacci recursivo.

```cpp
int fib(int n) {
    if (n <= 1) return n;
    int x, y;

    #pragma omp task shared(x)
    x = fib(n-1);

    #pragma omp task shared(y)
    y = fib(n-2);

    #pragma omp taskwait
    return x + y;
}
```

Se você testar `fib(30)` ou `fib(35)`. Verá que muitas tasks pequenas podem até piorar o tempo, devido ao overhead.

Pois é, nem sempre mais paralelismo significa mais velocidade.

## Conclusão

Na aula de hoje vimos que nem todo problema é igual. No caso da transformação elemento a elemento (map), o paralelismo funciona sem complicações porque cada iteração é totalmente independente. Já na soma parcial, o acesso simultâneo a uma variável compartilhada gera condições de corrida, que precisam ser resolvidas com mecanismos como `reduction`.

Também vimos que alguns laços possuem dependência entre iterações, nestes casos, não basta inserir diretivas OpenMP: é necessário repensar o algoritmo para eliminar a dependência.

Por fim, a experiência com recursão e tasks mostrou que o paralelismo pode gerar overhead se não for bem controlado. Criar muitas tarefas pequenas pode ser pior do que executar de forma sequencial.



**Esta atividade não tem entrega, bom fim de semana!!!**