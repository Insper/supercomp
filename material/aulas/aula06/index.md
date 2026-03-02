# Paralelismo em CPU 

Os processadores atuais possuem múltiplos núcleos de execução (*cores*). Para aproveitar essa capacidade, podemos dividir o trabalho em partes menores que possam ser executadas simultaneamente. Essa divisão é feita por meio de **threads**, onde cada thread executa uma fração das instruções.

No caso de **laços de repetição**, o paralelismo é obtido ao repartir as iterações entre várias threads. Em vez de uma única thread percorrer todo o laço, cada thread recebe um subconjunto de iterações, reduzindo o tempo total de execução. 

## O que é OpenMP?

OpenMP é uma API para **paralelismo em memória compartilhada** que permite transformar um código sequencial em paralelo com pequenas mudanças no código.

* Um processo
* Várias threads
* Todas compartilham o mesmo espaço de memória


Para usar OpenMP é necessário compilar com suporte a API, usando `-fopenmp`:

```bash
g++ -O3 -fopenmp codigo.cpp -o paralelo
```

Sem `-fopenmp`, os pragmas são ignorados.

### Estrutura mínima

```cpp
#include <omp.h>      // Biblioteca do OpenMP 
#include <iostream>   

int main() {

    // Início de uma região paralela.
    #pragma omp parallel
    {
        // Retorna o ID da thread atual.
        int id = omp_get_thread_num();

        // Retorna o número total de threads.
        int n  = omp_get_num_threads();

        // Cada thread executa este trecho, como std::cout é compartilhado, a ordem das mensagens
        // pode variar entre execuções.
        std::cout << "Thread " << id
                  << " de " << n << "\n";
    }
    // Ao final da região paralela existe uma barreira implícita:
    // todas as threads precisam terminar antes do código continuar.

    return 0;
}
```

### Configurando corretamente o SLURM 


**Rodar no cluster com SLURM** definindo o número de threads:

```bash
srun --partition=normal --cpus-per-task=4 ./meu_binario
```

ou

```bash
#!/bin/bash
#SBATCH --job-name=omp_test   # nome do job
#SBATCH --output=output_omp.out  # arquivo de saída
#SBATCH --cpus-per-task=4               # 4 threads para usar
#SBATCH --time=00:05:00                 # tempo máximo de execução
#SBATCH --mem=2G                        # Memória 

# garante que o OpenMP use exatamente os recursos alocados pelo SLURM
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# executa o binário
./meu_binario

```

### Principais Pragmas


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


## Scheduling no OpenMP

Quando um laço é paralelizado, é preciso definir **como as iterações serão distribuídas entre as threads**. Essa estratégia é chamada de **schedule** (escalonamento).

* **static** → divide as iterações em blocos fixos, atribuídos antecipadamente a cada thread.
* **dynamic** → as iterações são distribuídas em blocos sob demanda, conforme as threads terminam suas tarefas.
* **guided** → inicia com blocos maiores e reduz gradualmente o tamanho, equilibrando a carga de trabalho.
* **auto** → delega ao compilador a escolha da estratégia.
* **runtime** → a estratégia é definida em tempo de execução pela variável de ambiente `OMP_SCHEDULE`.

A escolha do escalonamento não altera o resultado final da computação, mas impacta diretamente o desempenho e o balanceamento da carga de trabalho.


O programa `omp_schedulers.cpp` é um código exemplo para **visualizar a distribuição das iterações de um laço entre threads** em diferentes estratégias de `schedule`.

Observe a saída do código:
`omp_schedulers.cpp`
```cpp
#include <iostream>   
#include <string>     
#include <vector>     
#include <algorithm>  
#include <omp.h>      

// -----------------------------------------------------------------------------
//   Imprime a distribuição para visualizar o efeito do 'schedule'.
// -----------------------------------------------------------------------------
void print_iterations(const std::string& description,
                      const std::vector< std::vector<int> >& vectors,
                      const int n)
{
    std::vector<std::string> strings(4, std::string()); // 4 linhas de saída, uma por thread
    for (int i = 0; i != n; i++)                        // varre todas as iterações 0..n-1
    {
        for (int j = 0; j != 4; j++)                   // para cada "thread" (0..3)
        {
            const auto& vector = vectors[j];           // vetor com as iterações que a thread j executou
            auto it = std::find(vector.begin(), vector.end(), i); // procura o i dentro do vetor da thread j
            if (it != vector.end())
            {
                strings[j] += "*";                     // se a thread j executou a iteração i, marca '*'
            }
            else
            { 
                strings[j] += " ";                     // caso contrário, espaço em branco
            }
        }
    }
    std::cout << description << std::endl;             
    for (auto& s : strings)                            // imprime as 4 linhas (uma por thread)
    {
        std::cout << s << "\n";
    }
    std::cout << std::endl;
}

// -----------------------------------------------------------------------------
//   - Função que recebe a política de agendamento,
//     a descrição e 'n'.
//   - Aloca 'vectors' (4 vetores: um por thread) e chama 'function' para preenchê-los.
//   - Depois imprime o resultado com print_iterations.
// -----------------------------------------------------------------------------
template <typename T>
void schedule(T function, 
              const std::string& description, 
              const int n)
{
    std::vector<std::vector<int>> vectors(4, std::vector<int>()); // 4 threads simuladas
    function(vectors, n);                                         // executa a política (preenche vectors)
    print_iterations(description, vectors, n);                    // visualiza distribuição
}

// -----------------------------------------------------------------------------
// Cada função 'scheduleXYZ' abaixo:
//   - Abre uma região paralela com 4 threads (num_threads(4))
//   - Faz um for paralelo sobre i = 0..n-1
//   - Cada thread registra a iteração 'i' que executou em vectors[tid]
// -----------------------------------------------------------------------------

// Default: sem especificar 'schedule' explicitamente (deixa o runtime decidir)
void scheduleDefault(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i); // registra a iteração i sob a thread atual
        }
    }
}

// schedule(static): divide as iterações em blocos fixos, um por thread 
void scheduleStatic(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(static, 4): blocos fixos de 4 iterações por vez
void scheduleStatic4(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(static, 4)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(static, 8): blocos fixos de 8 iterações por vez
void scheduleStatic8(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(static, 8)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(dynamic): threads pegam blocos sob demanda 
void scheduleDynamic(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(dynamic, 1): blocos dinâmicos de 1 iteração 
void scheduleDynamic1(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(dynamic, 4): blocos dinâmicos de 4 iterações
void scheduleDynamic4(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(dynamic, 8): blocos dinâmicos de 8 iterações
void scheduleDynamic8(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(guided): blocos começam grandes e vão diminuindo 
void scheduleGuided(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(guided)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(guided, 2): guided com bloco mínimo de 2
void scheduleGuided2(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(guided, 2)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(guided, 4): guided com bloco mínimo de 4
void scheduleGuided4(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(guided, 4)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(guided, 8): guided com bloco mínimo de 8
void scheduleGuided8(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(guided, 8)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(auto): deixa o runtime escolher o melhor 
void scheduleAuto(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(auto)
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

// schedule(auto): deixa o compilador escolher a melhor estratégia
void scheduleRuntime(std::vector<std::vector<int>>& vectors, int n)
{
    #pragma omp parallel num_threads(4) shared(vectors, n)
    {    
        #pragma omp for schedule(auto) 
        for (int i = 0; i < n; i++)
        {
            vectors[omp_get_thread_num()].push_back(i);
        }
    }
}

int main()
{
    const int n = 64; // número de iterações do laço a serem distribuídas entre 4 threads

    // Executa cada política de agendamento e imprime a “faixa” de iterações por thread.
    schedule(scheduleDefault,  "default:               ", n);
    schedule(scheduleStatic,   "schedule(static):      ", n);
    schedule(scheduleStatic4,  "schedule(static, 4):   ", n);
    schedule(scheduleStatic8,  "schedule(static, 8):   ", n);
    schedule(scheduleDynamic,  "schedule(dynamic):     ", n);
    schedule(scheduleDynamic1, "schedule(dynamic, 1):  ", n);
    schedule(scheduleDynamic4, "schedule(dynamic, 4):  ", n);
    schedule(scheduleDynamic8, "schedule(dynamic, 8):  ", n);
    schedule(scheduleGuided,   "schedule(guided):      ", n);
    schedule(scheduleGuided2,  "schedule(guided, 2):   ", n);
    schedule(scheduleGuided4,  "schedule(guided, 4):   ", n);
    schedule(scheduleGuided8,  "schedule(guided, 8):   ", n);
    schedule(scheduleAuto,     "schedule(auto):        ", n);
    schedule(scheduleRuntime,  "schedule(runtime):     ", n);

    return 0;
}

```


**Compilar o código com OpenMP**

```bash
g++ -fopenmp omp_schedulers.cpp -o omp_schedulers
```

**Rodar no cluster com SLURM** definindo o número de threads:

```bash
srun --partition=normal --cpus-per-task=4 ./omp_schedulers
```

ou

```bash
#!/bin/bash
#SBATCH --job-name=omp_scheduler_test   # nome do job
#SBATCH --output=output_omp_schedulers.out  # arquivo de saída
#SBATCH --cpus-per-task=4               # 4 threads para usar
#SBATCH --time=00:05:00                 # tempo máximo de execução
#SBATCH --mem=2G                        # Memória 

# garante que o OpenMP use exatamente os recursos alocados pelo SLURM
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# executa o binário
./omp_schedulers


```

### Analisando os Schedulers no OpenMP

Cada scheduler do OpenMP se comporta de maneira diferente, e você deve observar o impacto de cada um:

   **static**: As iterações são divididas igualmente entre as threads.
   
   **dynamic**: As threads pegam blocos de iterações conforme terminam o trabalho.
   
   **guided**: Distribui blocos maiores no início e menores no final, equilibrando a carga.
   
   **auto**: Deixa o compilador escolher a melhor estratégia.
   
   **runtime**: Usa a estratégia definida em tempo de execução.


## Sua vez de paralelizar com OpenMP

Código base:

`paralelo.cpp`
```cpp
// Compile: g++ -FlagDeOtimização -fopenmp paralelo.cpp -o paralelo
// Execute: ./paralelo [N]  (N padrão = 10'000'000)

#include <iostream>
#include <vector>
#include <random>
#include <algorithm> 
#include <omp.h>

int main(int argc, char** argv) {
    // ------------------------------
    // Parâmetros e dados de entrada
    // ------------------------------
    const int N = (argc >= 2 ? std::stoi(argv[1]) : 10'000'000);
    std::cout << "N = " << N << "\n";

    // Vetor base (valores aleatórios em [0,1))
    std::vector<float> a(N);
    {
        std::mt19937 rng(123);                // seed fixa só p/ reprodutibilidade
        std::uniform_real_distribution<> U(0.0, 1.0);
        for (int i = 0; i < N; ++i) a[i] = static_cast<float>(U(rng));
    }

    // =========================================================
    // Transformação elemento-a-elemento 
    // =========================================================
    const float alpha = 2.0f, beta = 35.5f;
    std::vector<float> c(N);

    double t0 = omp_get_wtime();

    for (int i = 0; i < N; ++i) {
        c[i] = alpha * a[i] + beta;
    }

    double t1 = omp_get_wtime();
    std::cout << "[A] tempo = " << (t1 - t0) << " s\n";
    int idx = N/2; // pega o elemento do meio do vetor
    std::cout << "[A] c[" << idx << "] = " << c[idx] << "\n";

    // =========================================================
    // TAREFA B: Soma 
    // =========================================================
    t0 = omp_get_wtime();

    double soma = 0.0;
    for (int i = 0; i < N; ++i) {
        soma += static_cast<double>(c[i]) * static_cast<double>(c[i]);
    }

    t1 = omp_get_wtime();
    std::cout << "[B] tempo = " << (t1 - t0) << " s | soma  = " << soma << "\n";
    return 0;
}

```

??? note "Gabarito"
    ```cpp
    // Compile: g++ -O3 -fopenmp paralelo.cpp -o paralelo
    // Execute: ./paralelo [N]

    #include <iostream>
    #include <vector>
    #include <random>
    #include <algorithm>
    #include <omp.h>

    int main(int argc, char** argv) {

        // ------------------------------
        // Parâmetros e dados de entrada
        // ------------------------------
        // Se o usuário passar N na linha de comando, usamos.
        // Caso contrário, usamos 10 milhões como padrão.
        const int N = (argc >= 2 ? std::stoi(argv[1]) : 10'000'000);
        std::cout << "N = " << N << "\n";

        // Mostra quantas threads estão disponíveis
        std::cout << "Threads disponíveis = "
                << omp_get_max_threads() << "\n";

        // ---------------------------------------------------------
        // Geração do vetor base 
        // ---------------------------------------------------------
        std::vector<float> a(N);
        {
            std::mt19937 rng(123);  // seed fixa para reprodutibilidade
            std::uniform_real_distribution<> U(0.0, 1.0);

            for (int i = 0; i < N; ++i)
                a[i] = static_cast<float>(U(rng));
        }

        // =========================================================
        // TAREFA A: Transformação elemento-a-elemento
        // =========================================================
        // Este é um caso clássico de paralelismo de dados:
        // Cada iteração é independente.
        const float alpha = 2.0f, beta = 35.5f;
        std::vector<float> c(N);

        double t0 = omp_get_wtime();

        // parallel for:
        // - Cria múltiplas threads
        // - Divide automaticamente o intervalo [0, N)
        // - Cada thread processa um bloco do vetor
        // schedule(static) divide igualmente
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            c[i] = alpha * a[i] + beta;
        }

        double t1 = omp_get_wtime();

        std::cout << "[A] tempo = " << (t1 - t0) << " s\n";

        // Apenas para verificar correção
        int idx = N/2;
        std::cout << "[A] c[" << idx << "] = " << c[idx] << "\n";

        // =========================================================
        // TAREFA B: Soma dos quadrados
        // =========================================================
        // Aqui existe dependência de escrita na variável 'soma'.
        // Sem tratamento adequado haveria race condition.
        t0 = omp_get_wtime();

        double soma = 0.0;

        // reduction(+:soma):
        // - Cada thread recebe uma cópia privada de 'soma'
        // - Faz a acumulação localmente
        // - Ao final, OpenMP combina todos os resultados
        // Isso evita race condition e é mais eficiente que atomic/critical.
        #pragma omp parallel for reduction(+:soma) schedule(static)
        for (int i = 0; i < N; ++i) {
            soma += static_cast<double>(c[i]) *
                    static_cast<double>(c[i]);
        }

        t1 = omp_get_wtime();

        std::cout << "[B] tempo = " << (t1 - t0)
                << " s | soma = " << soma << "\n";

        return 0;
    }

    ```

Para Compilar:

```bash
g++ -FlagDeOtimização -fopenmp paralelo.cpp -o paralelo
```

Para Executar:
```bash
#!/bin/bash
#SBATCH --job-name=paralelo_todo      # Nome do job
#SBATCH --output=paralelo.txt         # nome do arquivo de saida
#SBATCH --cpus-per-task=4             # Esse processo tem 4 threads disponíveis para usar
#SBATCH --mem=2G                      # Memória solicitada
#SBATCH --time=00:05:00               # Tempo solicitado (hh:mm:ss)
#SBATCH --partition=normal            # fila

# ------------------------------
# Configurações OpenMP
# ------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}   # OpenMP abre 4 threads e distribui o trabalho entre elas

# Troque aqui o Schedule do seu teste
export OMP_SCHEDULE=static

# Prints bonitinhos
echo "==== Configuração de Execução ===="
echo "Job ID          : $SLURM_JOB_ID"
echo "CPUs-per-task   : $SLURM_CPUS_PER_TASK"
echo "Memória total   : $SLURM_MEM_PER_NODE MB"
echo "OMP_NUM_THREADS : $OMP_NUM_THREADS"
echo "OMP_SCHEDULE    : $OMP_SCHEDULE"
echo "=================================="

# ------------------------------
# Executa o programa
# ------------------------------
./paralelo 

```

### Para entender se você entendeu!!

O objetivo desta atividade é praticar a paralelização de laços com OpenMP e analisar como diferentes políticas de escalonamento (`schedule`) impactam o desempenho.

Você deverá modificar o arquivo `paralelo.cpp`, paralelizando os laços apropriados com OpenMP. Em seguida, execute o programa utilizando os três tipos de escalonamento: `static`, `dynamic` e `guided`.

Varie o número de threads utilizando `OMP_NUM_THREADS` nos valores {1, 2, 4, 8, 16}.

Observe se houve ganho de desempenho ao aumentar o número de threads. O speedup ocorreu de forma proporcional? Até que ponto o aumento de threads trouxe benefício?

Compare também os três tipos de escalonamento (`static`, `dynamic` e `guided`). Algum deles apresentou desempenho superior? As diferenças foram significativas ou praticamente irrelevantes para este tipo de problema?



# Esta atividade não precisa ser entregue. Boa semana!

