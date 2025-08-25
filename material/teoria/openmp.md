# Guia de Pragmas OpenMP

### Funções da API OpenMP

* `omp_get_thread_num()` → retorna o ID da thread.
* `omp_get_num_threads()` → total de threads na região paralela.
* `omp_get_wtime()` → cronômetro de alta resolução.
* `omp_get_max_threads()` → número máximo de threads disponíveis.
* `OMP_NUM_THREADS` → número de threads usadas no programa
* `OMP_SCHEDULE` → define a política de escalonamento quando se usa `schedule(runtime)`

### Criando regiões paralelas

```cpp
#pragma omp parallel
{
    // código aqui roda em paralelo (todas as threads executam)
}
```



### Paralelizando laços (`for`)

```cpp
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    a[i] = b[i] + c[i];
}
```

* **Cláusula `schedule`**: define como dividir as iterações entre threads

  * `schedule(static)` → divide blocos iguais e fixos
  * `schedule(dynamic, chunk)` → distribui em blocos de `chunk` de forma dinâmica
  * `schedule(guided, chunk)` → blocos começam grandes e vão diminuindo
  * `schedule(runtime)` → definido pela variável de ambiente `OMP_SCHEDULE`



###  Variáveis privadas e compartilhadas

```cpp
#pragma omp parallel for private(x) shared(y)
for (int i = 0; i < N; i++) {
    int x = i;        // cada thread tem sua cópia
    y[i] = f(x);      // y é visível por todas
}
```

* `private(var)` → cada thread cria sua própria cópia
* `shared(var)` → todas as threads acessam a mesma variável



### Reduções (somatórios, produtos, etc.)

```cpp
double soma = 0.0;
#pragma omp parallel for reduction(+:soma)
for (int i = 0; i < N; i++) {
    soma += a[i];
}
```

* `+` → soma (ex.: `soma += ...`)
* `*` → produto (ex.: `prod *= ...`)
* `max` → máximo (ex.: encontra o maior valor)
* `min` → mínimo (ex.: encontra o menor valor)
* `&&` → AND lógico
* `||` → OR lógico
* `^`  → XOR bit a bit



### Seções paralelas

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    tarefa1();

    #pragma omp section
    tarefa2();
}
```

* Divide blocos de código independentes entre threads.



### Áreas críticas e exclusão mútua

```cpp
#pragma omp critical
{
    contador++;
}
```

* Apenas **uma thread por vez** entra nesse bloco.
* Útil para proteger atualizações em variáveis compartilhadas.



###  Diretiva `single`

```cpp
#pragma omp parallel
{
    #pragma omp single
    {
        std::cout << "Executado por apenas 1 thread\n";
    }
}
```

* Apenas **uma thread** executa esse trecho, mas as outras esperam.



### Barreira de sincronização

```cpp
#pragma omp barrier
```

* Faz todas as threads esperarem umas pelas outras antes de seguir adiante.



Sim 🙌 além das diretivas básicas que já coloquei no guia, existem outras **muito usadas na prática** que valem a pena aparecer num material de referência rápida para os alunos. Vou complementar a lista com as mais úteis/didáticas:



###  `#pragma omp parallel for collapse(n)`

O `collapse(n)` junta `n` loops aninhados em **um só loop paralelo**. Muito útil em matrizes e tensores.

```cpp
#pragma omp parallel for collapse(2)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        A[i][j] = i + j;
    }
}
```


###  `#pragma omp task`
Permite criar **tarefas assíncronas** dentro de uma região paralela. Muito usado para grafos, árvores e pipelines.

```cpp
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task
        f1();

        #pragma omp task
        f2();

        #pragma omp taskwait   // sincroniza as tasks
    }
}
```


### `#pragma omp atomic`
Protege uma operação simples (ex.: incremento) de condições de corrida, com overhead menor que `critical`.


```cpp
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    #pragma omp atomic
    soma += a[i];
}
```




### `#pragma omp master` e `#pragma omp single nowait`

* `master`: só a thread 0 roda.
* `single`: apenas uma thread roda (não necessariamente a 0).
* `nowait`: libera as threads de esperarem.

```cpp
#pragma omp parallel
{
    #pragma omp master
    { std::cout << "Apenas a thread master executa\n"; }

    #pragma omp single nowait
    { std::cout << "Uma thread qualquer executa e não há barreira\n"; }
}
```



###  `#pragma omp simd`
Força a vetorização SIMD (Single Instruction Multiple Data). 
Pode ser combinado com `parallel for` → `#pragma omp parallel for simd`.

```cpp
#pragma omp simd
for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
}
```



### Controlando variáveis

* `firstprivate(var)` → cada thread ganha uma cópia inicializada com o valor original.
* `lastprivate(var)` → garante que, ao final, o valor da última iteração fique na variável global.
* `default(shared)` → define política padrão de variáveis (bom para pegar erros!).




Documentação disponível em [openmp.org](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf)