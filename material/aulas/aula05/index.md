# Paralelismo em CPU com OpenMP

## Objetivo

* **Paralelismo em CPU**: como dividir o trabalho entre múltiplos *cores*.
* **Threads**: cada thread executa uma parte do trabalho.
* **OpenMP**: diretivas em OpenMP para paralelizar loops.
* **Scheduling**: forma como as iterações do loop são distribuídas entre threads (`static`, `dynamic`, `guided`).

Os processadores atuais possuem múltiplos núcleos de execução (*cores*). Para aproveitar essa capacidade, podemos dividir o trabalho em partes menores que possam ser executadas simultaneamente. Essa divisão é feita por meio de **threads**, onde cada thread executa uma fração das instruções.

No caso de **laços de repetição**, o paralelismo é obtido ao repartir as iterações entre várias threads. Em vez de uma única thread percorrer todo o laço, cada thread recebe um subconjunto de iterações, reduzindo o tempo total de execução. O OpenMP facilita esse processo por meio de diretivas como `#pragma omp parallel for`.

!!! tip
    Para saber mais, veja o material disponível em [Guia de Pragmas OpnMP](../../teoria/openmp.md) 
    
## Scheduling no OpenMP

Quando um laço é paralelizado, é preciso definir **como as iterações serão distribuídas entre as threads**. Essa estratégia é chamada de **schedule** (escalonamento).

* **static** → divide as iterações em blocos fixos, atribuídos antecipadamente a cada thread.
* **dynamic** → as iterações são distribuídas em blocos sob demanda, conforme as threads terminam suas tarefas.
* **guided** → inicia com blocos maiores e reduz gradualmente o tamanho, equilibrando a carga de trabalho.
* **auto** → delega ao compilador a escolha da estratégia.
* **runtime** → a estratégia é definida em tempo de execução pela variável de ambiente `OMP_SCHEDULE`.

A escolha do escalonamento não altera o resultado final da computação, mas impacta diretamente o desempenho e o balanceamento da carga de trabalho.


O programa `omp_schedulers.cpp` é um código exemplo para **visualizar a distribuição das iterações de um laço entre threads** em diferentes estratégias de `schedule`.

Para cada política de escalonamento, o programa registra quais iterações foram executadas por cada thread e exibe uma saída gráfica com `*`, indicando a distribuição.

Dessa forma, é possível observar o comportamento de cada política:

`omp_schedulers.cpp`
```cpp
#include <iostream>   
#include <string>     
#include <vector>     
#include <algorithm>  
#include <omp.h>      // OpenMP (omp_get_thread_num, diretivas)

// -----------------------------------------------------------------------------
// print_iterations:
//   - Recebe uma descrição, um vetor de vetores 'vectors' (4 vetores, um por thread),
//     e 'n' (número total de iterações do laço).
//   - Constrói 4 strings com '*' indicando quais iterações cada thread executou.
//   - Imprime a distribuição para visualizar o efeito do 'schedule'.
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
    std::cout << description << std::endl;             // título/descrição da experiência
    for (auto& s : strings)                            // imprime as 4 linhas (uma por thread)
    {
        std::cout << s << "\n";
    }
    std::cout << std::endl;
}

// -----------------------------------------------------------------------------
// schedule (template):
//   - Função "driver" que recebe outra função 'function' (uma política de agendamento),
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
//   - Faz um for paralelo '#pragma omp for' sobre i = 0..n-1
//   - Cada thread registra a iteração 'i' que executou em vectors[tid]
//   Observação didática: push_back em 'vectors[tid]' é aceitável aqui para fins
//   de visualização (em geral, acessos concorrentes a containers exigem cuidado).
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

// schedule(static): divide as iterações em blocos fixos, um por thread (tamanho auto)
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

// schedule(dynamic): threads pegam blocos sob demanda (tamanho padrão do runtime)
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

// schedule(dynamic, 1): blocos dinâmicos de 1 iteração (alto overhead, ótimo balanceamento)
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

// schedule(guided): blocos começam grandes e vão diminuindo (bom p/ carga irregular)
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

// schedule(auto): deixa o runtime escolher o melhor esquema
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
srun --partition=normal --ntasks=1 --cpus-per-task=4 ./omp_schedulers
```

ou

```bash
#!/bin/bash
#SBATCH --job-name=omp_scheduler_test   # nome do job
#SBATCH --output=output_omp_schedulers.out  # arquivo de saída
#SBATCH --ntasks=1                      # 1 processo (1 task MPI)
#SBATCH --cpus-per-task=4               # 4 CPUs para essa task → 4 threads OMP
#SBATCH --time=00:05:00                 # tempo máximo de execução
#SBATCH --mem=2G                        # Memória total do job (ex.: 2 GB)

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


## Atividade 03 - Paralelismo com OpenMP

Código base para a atividade 03:

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
    // TAREFA A: Transformação elemento-a-elemento (map)
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
    // TAREFA B: Soma (redução) da norma L2 parcial
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
Para Compilar:

```bash
g++ -FlagDeOtimização -fopenmp paralelo.cpp -o paralelo
```

Para Executar:
```bash
#!/bin/bash
#SBATCH --job-name=paralelo_todo      # Nome do job
#SBATCH --output=paralelo.txt         # nome do arquivo de saida
#SBATCH --ntasks=1                    # Sempre 1 processo (o programa roda só uma vez)
#SBATCH --cpus-per-task=4             # Esse processo tem 4 CPUs disponíveis para usar
#SBATCH --mem=2G                      # Memória solicitada
#SBATCH --time=00:05:00               # Tempo solicitado (hh:mm:ss)
#SBATCH --partition=normal            # fila

# ------------------------------
# Configurações OpenMP
# ------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}   # OpenMP abre 4 threads e distribui o trabalho entre elas
export OMP_PLACES=cores                         # Fixa threads em cores
export OMP_PROC_BIND=close                      # Coloca threads próximas (melhor cache)

# Troque aqui o Schedule do seu teste
export OMP_SCHEDULE=static

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
# paralelo (4 threads, por ex.)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}   # OpenMP abre 4 threads e distribui o trabalho entre elas
./paralelo 

```


### Objetivo

Paralelizar laços com OpenMP, comparar o efeito de `schedule` no desempenho.

### Tarefa A - Map: transformação elemento-a-elemento

**Solicitações de Implementação**

1. Paralelize o código correspondente a Tarefa A.
2. Registre **tempo** e **valor da conta** em 3 execuções para cada `OMP_SCHEDULE = static`, `dynamic`, `guided`.
3. O que está sendo paralelizado nesse for? O que está sendo distribuido entre as threads?

---

### Tarefa B - Redução ingênua: soma de quadrados

**Solicitações de Implementação**

1. Paralelize o código correspondente a Tarefa B.
2. Registre **tempo** e **valor da soma** em 3 execuções para cada `OMP_SCHEDULE = static`, `dynamic`, `guided`.
3. Compare com a execução **sequencial** (threads=1).
4. O que está sendo paralelizado nesse for? O que está sendo distribuido entre as threads?

---

### Coleta de Resultados (mínimo)

* **Tabela tempos (Parte 1)**: `scheduler`, `execução`, `tempo (s)`.
* **Tabela tempos (Parte 2)**: `schedule`, `threads`, `média (s)`, `desvio`.
* **Tabela soma (Parte 3)**: `schedule`, `tempo (s)`, `soma obtida`.

### Parâmetros de Execução

* Varie `OMP_NUM_THREADS` em {1, 2, 4, 8} (quando solicitado).
* Mantenha os **mesmos N** (tamanho do problema) em todas as comparações do mesmo grupo.


**Perguntas de Análise**

* Houve **speedup** com mais threads? Até onde?
* `static` vs `dynamic` vs `guided`: quem foi melhor? Alguma diferença relevante?
* Alguma mudança no resultado das contas? 

**Faça um relatório com as suas análises e entregue até as 23h59 de 28/08 pelo [GitHub Classroom](https://classroom.github.com/a/kMFUt-E8)** 

