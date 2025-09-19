

## Exercícios para estudar para AI

A **Ninja Laís Freitas** trabalhou na elaboração de  exercícios para que vocês se ambientem com o estilo de prova e exercitem os seus conhecimentos, os Exercícios estão separados por aula, você pode acessa-los [clicando aqui](../../Exercicios/aula01/aula01.md) ou em **"Exercícios"** ao lado de **"Suporte"**;

## **Simulado AI**

Além dos exercícios preparados pela Ninja Laís, também disponibilizo um simulado. A prova contará com questões teóricas no Blackboard e questões práticas no Classroom, elaboradas no mesmo nível de dificuldade do simulado disponibilizado. 

??? warning ""É fundamental que você realize os testes necessários para garantir que o SMOWL esteja funcionando corretamente em seu computador antes da prova. A responsabilidade pela infraestrutura adequada é inteiramente do aluno. Caso a ferramenta não esteja disponível, a sua prova será anulada. Em situações de dificuldade técnica, procure o Helpdesk."

## Questões Teóricas

??? warning "ATENÇÃO: As respostas estão gigantes para que você tenha material para estudar, ao responder na hora da prova, você pode ser mais direto."


**1. Tiling**

Explique por que a técnica de tiling melhora o desempenho de programas paralelos.

* Relacione com hierarquia de memória (L1, L2, L3, RAM).
* Diga por que, na prática, costuma-se usar o tamanho da **cache L2** como referência para o tamanho dos blocos.


??? note "Ver Resposta"

        A técnica de tiling consiste em dividir um problema grande (matriz, vetor, dados) 
        em sub-blocos menores, chamados de tiles, que são processados um de cada vez.

        O processador tem múltiplos níveis de memória cache:

        - L1: muito rápida, mas muito pequena. 
        - L2: maior que L1, ainda rápida.  
        - L3: bem maior, mas mais lenta que L1/L2 e é compartilhada entre os cores.
        - RAM: muito maior, mas muito lenta comparada às caches.
        
        Se os dados acessados couberem na cache, evitamos o custo de buscar na RAM.

        Trabalhar em blocos significa utilizar dados que já estão na cache antes que sejam descartados. 

        Por que usar L2 como referência para o tamanho dos blocos?
        O tiling melhora o desempenho porque organiza o acesso à memória de forma que os dados 
        “fiquem mais tempo nas caches rápidas” e menos tempo sendo buscados na RAM.
        Usar o tamanho da L2 é a prática comum porque ela oferece
        o melhor equilíbrio entre capacidade e latência.


**2. Balanceamento de carga — OpenMP**

Considere um laço paralelizado com OpenMP em que cada iteração tem custo variável.

* Compare `schedule(static)` e `schedule(dynamic)`.
* Explique em qual cenário cada estratégia é mais vantajosa.
* Cite uma situação em que `schedule(guided)` pode ser melhor.

??? note "Ver Resposta"

    Quando usamos OpenMP, a forma como as iterações de um laço são distribuídas entre as threads pode impactar o desempenho. O static divide o espaço de iterações em blocos fixos e atribui cada bloco a uma thread antes da execução começar. Essa estratégia tem a vantagem de ter overhead muito baixo, porque a divisão é feita apenas uma vez, e mantém uma boa localidade de memória, já que cada thread tende a percorrer regiões contíguas do vetor ou da matriz. Por isso, o `schedule(static)` é bastante eficiente quando o custo por iteração é previsível e relativamente uniforme.

    O dynamic funciona de forma adaptativa: as threads recebem blocos de iterações sob demanda e, assim que terminam, pegam novos blocos disponíveis. Esse modelo introduz mais overhead, porque é necessário sincronizar a fila de tarefas várias vezes, e pode prejudicar um pouco a localidade de cache, já que as iterações não necessariamente ficam contíguas para cada thread. A contrapartida é que essa abordagem consegue lidar melhor com laços em que o custo das iterações é irregular. Se algumas iterações demandam muito mais trabalho do que outras, o `dynamic` evita que uma thread fique sobrecarregada enquanto outras ficam ociosas, promovendo melhor balanceamento da carga de trabalho.

    O guided, é uma variação do dynamic. Os blocos começam grandes e vão diminuindo de tamanho conforme o laço avança. Assim, o início da execução tem menos overhead de gerenciamento, enquanto o final se beneficia de blocos menores que ajudam a equilibrar a carga restante. Essa estratégia é útil quando temos um número muito grande de iterações com custos variáveis, pois combina a eficiência inicial de blocos maiores com a adaptabilidade final de blocos pequenos.

    Na prática, escolhemos `static` para problemas regulares e previsíveis, `dynamic` para casos irregulares ou imprevisíveis, e `guided` quando queremos um meio-termo entre overhead baixo e balanceamento eficiente, quando temos laços longos e diferentes complexidades computacionais.


**3. MPI — Comunicação**

Explique a diferença entre comunicação ponto-a-ponto e coletiva no MPI. Dê um exemplo de uso para cada categoria.

??? note "Ver Resposta"
    A comunicação ponto-a-ponto ocorre quando dois processos trocam mensagens diretamente entre si, de forma explícita. Normalmente, isso é feito com funções como `MPI_Send` e `MPI_Recv`. Nesse modelo, um processo envia uma mensagem contendo dados e outro processo, identificado pelo seu rank, recebe essa mensagem. Esse tipo de comunicação é útil quando queremos ter controle fino sobre quem fala com quem e em que momento, como em um pipeline onde cada processo realiza uma etapa do cálculo e passa o resultado para o próximo.

    Já a comunicação coletiva envolve um grupo inteiro de processos dentro de um comunicador, coordenando a troca de informações de forma padronizada. Exemplos típicos são `MPI_Bcast`, que envia uma mensagem de um processo para todos os outros, `MPI_Reduce`, que combina dados de todos os processos em um único resultado, e `MPI_Gather`, que junta dados espalhados em um único processo. Esse tipo de comunicação é mais conveniente quando todos os processos precisam participar da mesma operação, como calcular a soma global de resultados parciais de um vetor distribuído entre diferentes processos.

    Assim, enquanto a comunicação ponto-a-ponto é flexível e granular, permitindo desenhar padrões específicos de interação, a comunicação coletiva simplifica operações envolvendo todos os processos.


**4. Códigos híbridos MPI+OpenMP**

Quais são as vantagens de combinar MPI e OpenMP em um cluster de HPC?

??? note "Ver Resposta"
    A principal vantagem de combinar MPI e OpenMP em um cluster de HPC é aproveitar dois níveis de paralelismo ao mesmo tempo, MPI divide o trabalho entre os diferentes nós do cluster (memória distribuída). O OpenMP divide o trabalho entre os núcleos de CPU dentro de cada nó usando thread(memória compartilhada).

    Isso reduz o número de processos MPI, diminui o tráfego de mensagens, e melhora o uso da cache dos nós do cluster.




**5. Passagem por valor, referência e ponteiro**

a) Explique as diferenças entre passagem por valor, referência e ponteiro em C++. 
b) Em termos de cópia de dados e eficiência, qual é a diferença prática?

??? note "Ver Resposta"
    Em C++, quando passamos um parâmetro por valor, o compilador cria uma cópia independente da variável original. Isso garante segurança (o original nunca é alterado), mas pode ser custoso se o objeto for grande, já que toda a cópia precisa ser feita.

    Na passagem por referência, não há cópia: a função recebe um “apelido” para a variável original. Assim, qualquer modificação feita dentro da função afeta o valor fora dela. Essa forma é eficiente porque evita cópias desnecessárias, mas exige cuidado para não alterar dados de forma indesejada.

    Já na passagem por ponteiro, a função recebe o endereço da variável. A eficiência é semelhante à da referência (não há cópia), mas o código fica mais verboso e exige atenção extra, pois ponteiros podem ser nulos ou apontar para locais inválidos na memória.

    Na prática, a diferença está no custo de cópia: valores grandes (como vetores ou objetos complexos) são muito mais eficientes quando passados por referência ou ponteiro. A passagem por valor só é recomendada para tipos pequenos e triviais (como `int` ou `bool`).


**6. Escalonamento e desempenho**

a) Por que a escolha do escalonamento `schedule` pode afetar o desempenho de um programa paralelo com OpenMP?

b) O que isso tem a ver com tiling (divisão de blocos de dados) para encaixar na cache?

??? note "Ver Resposta"
    a) A escolha do `schedule` em OpenMP afeta diretamente como as iterações de um loop são distribuídas entre as threads, e isso tem impacto no equilíbrio de carga e no uso eficiente da memória. Se usamos `schedule(static)`, cada thread recebe blocos fixos de iterações, o que funciona bem quando todas as iterações têm custo parecido. Mas, se o custo variar, algumas threads podem terminar mais cedo e ficar ociosas, desperdiçando desempenho. Já `schedule(dynamic)` e `guided` permitem redistribuir iterações conforme as threads vão terminando, equilibrando melhor a carga, mas com um pouco mais de overhead de gerenciamento.

    b) Esse conceito se conecta ao tiling porque dividir as os dados em blocos também é uma forma de escalonamento, mas voltada para a hierarquia de memória. Ao quebrar os dados em blocos do tamanho adequado para caber na cache, garantimos que cada thread trabalhe em um conjunto de dados contíguos e reutilizáveis. Tanto o `schedule` quanto o `tiling` lidam com a divisão do trabalho, o primeiro com balanceamento de carga entre nós, o segundo com otimização da memória cache.


## Questões Práticas

**7. Tiling em produto vetorial**

Implemente a multiplicação de dois vetores grandes em blocos `tiling`, comparando o desempenho com a versão ingênua.

* Use `Bsize` definido para caber na cache L2.
* Meça o tempo de execução com e sem tiling.

```cpp
// q07.cpp
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::atoi(argv[1]) : 1000000);
    int Bsize = (argc > 2 ? std::atoi(argv[2]) : 32768); // ajuste pensando na L2

    std::vector<float> A(N, 1.1f), B(N, 2.2f), C_naive(N, 0.0f), C_tile(N, 0.0f);

    // ===== Versão ingênua =====
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    // TODO: percorrer i=0..N-1 e preencher C_naive[i] = A[i] * B[i]
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // ===== Versão com tiling =====
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    // TODO: laço de blocos
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

    double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double tile_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "N=" << N << " Bsize=" << Bsize
              << " naive_ms=" << naive_ms
              << " tile_ms="  << tile_ms << "\n";

    // TODO: validar (comparar C_naive vs C_tile)
    return 0;
}
```

??? note "Ver Resposta"
        #include <iostream>
        #include <vector>
        #include <chrono>

        // Objetivo: comparar a versão ingênua (varre o vetor todo) com a versão em BLOCOS (tiling).
        // Melhor aproveitamento da cache: processamos blocos contíguos que "cabem" na L2.

        int main(int argc, char** argv) {
            int N = (argc > 1 ? std::atoi(argv[1]) : 1000000);
            int Bsize = (argc > 2 ? std::atoi(argv[2]) : 32768); // ajuste pensando na L2 (50–75% da L2/sizeof(float))

            std::vector<float> A(N, 1.1f), B(N, 2.2f), C_naive(N, 0.0f), C_tile(N, 0.0f);

            // ===== Versão ingênua (baseline) =====
            std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < N; i++) {
                C_naive[i] = A[i] * B[i];
            }
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

            // ===== Versão com tiling =====
            // Percorre o vetor em janelas [start, end) contíguas → melhor localidade espacial.
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            for (int start = 0; start < N; start += Bsize) {
                int end = (start + Bsize < N) ? (start + Bsize) : N;
                for (int i = start; i < end; i++) {
                    C_tile[i] = A[i] * B[i];
                }
            }
            std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

            double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double tile_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();

            // Validação simples
            int erros = 0;
            for (int i = 0; i < N; i++) {
                if (C_naive[i] != C_tile[i]) { erros++; break; }
            }

            std::cout << "N=" << N << " Bsize=" << Bsize
                    << " naive_ms=" << naive_ms
                    << " tile_ms="  << tile_ms
                    << " ok=" << (erros == 0) << "\n";
            return 0;
        }


**8. Balanceamento de carga em OpenMP**

Implemente um programa em C++ que conta quantos números primos existem em um vetor de inteiros grandes.

* Paralelize com `#pragma omp parallel for`.
* Encontre qual é o menor custo de hardware para o melhor benefício de otimização

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

static bool is_prime(unsigned x) {
    // TODO: implementar teste de primalidade simples (ou copiar do material de exercícios)
    return false; 
}

int main(int argc, char** argv) {
    int N = (argc > 1 ? std::atoi(argv[1]) : 500000);
    unsigned seed = (argc > 2 ? (unsigned)std::atoi(argv[2]) : 123u);

    std::vector<unsigned> v(N);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<unsigned> U(1u, 5000000u);
    for (int i = 0; i < N; i++) v[i] = U(rng);

    long long total_primos = 0;
    double t0 = omp_get_wtime();


    //TODO paralelize adequadamente esse loop
    for (int i = 0; i < N; i++) {
        // TODO: se for primo, incrementa a lista de primos
    }

    double t1 = omp_get_wtime();
    std::cout << "N=" << N << " primos=" << total_primos
              << " tempo_s=" << (t1 - t0) << "\n";
    return 0;
}
```

??? note "Ver Resposta"

        #include <iostream>
        #include <vector>
        #include <random>
        #include <omp.h>

        // Função para verificar se um número é primo
        static bool eh_primo(unsigned int numero) {
            if (numero < 2) return false;                // menores que 2 não são primos
            if (numero % 2 == 0) return numero == 2;     // múltiplos de 2 só são primos se forem o próprio 2

            // Testa apenas divisores ímpares até a raiz quadrada do número
            unsigned int divisor = 3;
            while (divisor * divisor <= numero) {
                if (numero % divisor == 0) return false; // achou divisor → não é primo
                divisor += 2;                            // incrementa de 2 em 2 → só ímpares
            }
            return true;
        }

        int main(int argc, char** argv) {
            // Quantos números vamos testar (padrão = 500.000)
            int quantidade_numeros = (argc > 1 ? std::atoi(argv[1]) : 500000);

            // Semente para o gerador de números aleatórios (padrão = 123)
            unsigned int semente = (argc > 2 ? (unsigned int)std::atoi(argv[2]) : 123);

            // Vetor que guarda os números aleatórios
            std::vector<unsigned int> numeros(quantidade_numeros);

            // Gerador pseudoaleatório
            std::mt19937 gerador(semente);
            std::uniform_int_distribution<unsigned int> distribuicao(1, 5000000);

            // Preenche o vetor com números aleatórios entre 1 e 5.000.000
            for (int i = 0; i < quantidade_numeros; i++) {
                numeros[i] = distribuicao(gerador);
            }

            long long contador_primos = 0;  // contador de primos encontrados

            // Marca tempo inicial
            double tempo_inicio = omp_get_wtime();

            // Loop paralelo com OpenMP
            // - Cada thread pega um pedaço do vetor
            // - reduction(+:contador_primos) → soma local de cada thread é acumulada corretamente
            // - schedule(static,4) → podemos testar com OMP_SCHEDULE=dynamic,4 ou guided,4
            #pragma omp parallel for reduction(+:contador_primos) schedule(static,4)
            for (int i = 0; i < quantidade_numeros; i++) {
                if (eh_primo(numeros[i])) {
                    contador_primos += 1;
                }
            }

            // Marca tempo final
            double tempo_fim = omp_get_wtime();

            // Exibe resultados
            std::cout << "Total de numeros testados = " << quantidade_numeros << "\n";
            std::cout << "Quantidade de primos encontrados = " << contador_primos << "\n";
            std::cout << "Tempo de execucao = " << (tempo_fim - tempo_inicio) << " segundos\n";

            return 0;
        }


---

**9. MPI — Somas distribuídas**

Escreva um programa MPI que:

* Inicializa um vetor de tamanho `N` no `rank 0`.
* Divide o vetor com `MPI_Scatter`.
* Cada processo calcula a soma parcial.
* Usa `MPI_Reduce` para calcular a soma total no `rank 0`.

```cpp
// q09_mpi_soma.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1 ? std::atoi(argv[1]) : (1 << 20));
    std::vector<int> v;
    if (rank == 0) {
        v.resize(N);
        for (int i = 0; i < N; i++) v[i] = i % 10;
    }

    // Assumimos N % size == 0 neste esqueleto
    int chunk = N / size;
    std::vector<int> local(chunk);

    // TODO dividir o vetor com  MPI scatter

    long long soma_local = 0;
    // TODO: somar todos os elementos do vetor local[0..chunk-1] em soma_local

    long long soma_total = 0;
    
    // TODO: implementar MPI Reduce
    if (rank == 0) {
        std::cout << "Soma total = " << soma_total << "\n";
    }

    MPI_Finalize();
    return 0;
}

```

??? note "Ver Resposta"

        #include <mpi.h>
        #include <iostream>
        #include <vector>
        #include <cstdlib>

        // Padrão: assume N % size == 0 para simplicidade.

        int main(int argc, char** argv) {
            MPI_Init(&argc, &argv);

            int rank = 0, size = 1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            int N = (argc > 1 ? std::atoi(argv[1]) : (1 << 20)); // ~1M
            std::vector<int> v;
            if (rank == 0) {
                v.resize(N);
                for (int i = 0; i < N; i++) v[i] = i % 10; // dados simples
            }

            int chunk = N / size;
            std::vector<int> local(chunk);

            // Divide o vetor v em pedaços contíguos para todos os ranks
            MPI_Scatter(v.data(), chunk, MPI_INT,
                        local.data(), chunk, MPI_INT,
                        0, MPI_COMM_WORLD);

            // Soma local do pedaço recebido
            long long soma_local = 0;
            for (int i = 0; i < chunk; i++) soma_local += local[i];

            // Reduz todas as somas locais para o rank 0
            long long soma_total = 0;
            MPI_Reduce(&soma_local, &soma_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::cout << "Soma total = " << soma_total << "\n";
            }

            MPI_Finalize();
            return 0;
        }


---

**10. Código híbrido MPI+OpenMP**

Implemente um programa que calcula o produto escalar entre dois vetores grandes:

* Distribua os vetores entre os processos com `MPI_Scatter`.
* Cada processo calcula a soma parcial com OpenMP reduction.
* Combine os resultados no `rank 0` com `MPI_Reduce`.

```cpp
// q10_hibrido_dot.cpp
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1 ? std::atoi(argv[1]) : (1 << 20));
    std::vector<float> v, w;
    if (rank == 0) {
        v.assign(N, 1.0f);
        w.assign(N, 2.0f);
    }

    int chunk = N / size; // esqueleto simples (N múltiplo de size)
    std::vector<float> vl(chunk), wl(chunk);
    //TODO: implementar o MPI_scatter para o vetor v

    //TODO: implementar o MPI_scatter para o vetor w
    
    double soma_local = 0.0;

    // TODO: paralelizar com OpenMP  e calcular soma_local 

    double soma_total = 0.0;
    // TODO implementar MPI_Reduce 

    if (rank == 0) {
        std::cout << "dot = " << soma_total << "\n";
    }

    MPI_Finalize();
    return 0;
}

```

??? note "Ver respostas"

        #include <mpi.h>
        #include <omp.h>
        #include <iostream>
        #include <vector>
        #include <cstdlib>

        // Estratégia: Scatter v e w, cada processo calcula produto parcial com OpenMP (reduction),
        // depois Reduce (soma) no rank 0.

        int main(int argc, char** argv) {
            MPI_Init(&argc, &argv);

            int rank = 0, size = 1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            int N = (argc > 1 ? std::atoi(argv[1]) : (1 << 20));
            std::vector<float> v, w;
            if (rank == 0) {
                v.assign(N, 1.0f);
                w.assign(N, 2.0f);
            }

            int chunk = N / size; // simples: N múltiplo de size
            std::vector<float> vl(chunk), wl(chunk);

            MPI_Scatter(v.data(), chunk, MPI_FLOAT, vl.data(), chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Scatter(w.data(), chunk, MPI_FLOAT, wl.data(), chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

            // Redução em double reduz erro de arredondamento
            double soma_local = 0.0;

            // Paralelismo intra-nó com OpenMP
            #pragma omp parallel for reduction(+:soma_local) schedule(static)
            for (int i = 0; i < chunk; i++) {
                soma_local += (double)vl[i] * (double)wl[i];
            }

            double soma_total = 0.0;
            MPI_Reduce(&soma_local, &soma_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::cout << "dot = " << soma_total << "\n";
            }

            MPI_Finalize();
            return 0;
        }


---

**11. Otimização com passagem por referência**

Implemente duas versões de uma função que calcula a média móvel de um vetor de `double`:

* Versão (a) recebe os dados **por valor**.
* Versão (b) recebe os dados **por referência constante**.
* Compare os tempos de execução para vetores de 10⁶ elementos e indique qual foi a versão mais eficiente e justifique porque.

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// (a) Média móvel passando POR VALOR (copia 'dados')
std::vector<double> media_movel_por_valor(std::vector<double> dados, std::size_t janela_K) {
    // TODO: validar janela_K (0 < janela_K <= dados.size())
    // TODO: somar os 'janela_K' primeiros elementos
    // TODO: empurrar a primeira média para o vetor de saída
    // TODO: janela deslizante: para i = janela_K..N-1
    //       soma += dados[i] - dados[i - janela_K];
    //       empurrar nova média
    return std::vector<double>();
}

// (b) TODO passe os dados POR REFERÊNCIA 
std::vector<double> media_movel_por_referencia(std::vector<double> dados, std::size_t janela_K){
    // TODO: mesma lógica da versão por valor, mas SEM copiar 'dados'
    return std::vector<double>();
}

int main(int argc, char** argv) {
    // Parâmetros de entrada
    std::size_t tamanho_N = (argc > 1 ? (std::size_t)std::atoll(argv[1]) : 1000000);
    std::size_t janela_K  = (argc > 2 ? (std::size_t)std::atoll(argv[2]) : 128);

    // Gera dados aleatórios em [0,1)
    std::vector<double> vetor_dados(tamanho_N);
    std::mt19937_64 gerador(42u);
    std::uniform_real_distribution<double> distribuicao(0.0, 1.0);
    for (std::size_t i = 0; i < tamanho_N; i++) {
        vetor_dados[i] = distribuicao(gerador);
    }

    // Mede tempo da versão por VALOR
    std::chrono::high_resolution_clock::time_point inicio_valor = std::chrono::high_resolution_clock::now();
    std::vector<double> medias_valor = media_movel_por_valor(vetor_dados, janela_K);
    std::chrono::high_resolution_clock::time_point fim_valor = std::chrono::high_resolution_clock::now();

    // Mede tempo da versão por REFERÊNCIA
    std::chrono::high_resolution_clock::time_point inicio_referencia = std::chrono::high_resolution_clock::now();
    std::vector<double> medias_referencia = media_movel_por_referencia(vetor_dados, janela_K);
    std::chrono::high_resolution_clock::time_point fim_referencia = std::chrono::high_resolution_clock::now();

    // Cálculo de tempos (ms)
    double tempo_valor_ms      = std::chrono::duration<double, std::milli>(fim_valor - inicio_valor).count();
    double tempo_referencia_ms = std::chrono::duration<double, std::milli>(fim_referencia - inicio_referencia).count();

    // Validação simples (tolerância numérica)
    bool resultados_iguais = (medias_valor.size() == medias_referencia.size());
    for (std::size_t i = 0; resultados_iguais && i < medias_valor.size(); i++) {
        if (std::abs(medias_valor[i] - medias_referencia[i]) > 1e-12) {
            resultados_iguais = false;
        }
    }

    // Saída
    std::cout << "tempo_valor_ms=" << tempo_valor_ms
              << " tempo_referencia_ms=" << tempo_referencia_ms
              << " iguais=" << (resultados_iguais ? 1 : 0) << "\n";

    return 0;
}

```

??? note "Ver resposta"
        #include <iostream>
        #include <vector>
        #include <random>
        #include <chrono>
        #include <cmath>

        // (a) Média móvel passando POR VALOR (copia 'dados')
        std::vector<double> media_movel_por_valor(std::vector<double> dados, std::size_t janela_K) {
            std::vector<double> medias;
            if (janela_K == 0 || janela_K > dados.size()) return medias;

            // soma inicial dos K primeiros
            double soma = 0.0;
            for (std::size_t i = 0; i < janela_K; i++) {
                soma += dados[i];
            }
            medias.push_back(soma / (double)janela_K);

            // janela deslizante
            for (std::size_t i = janela_K; i < dados.size(); i++) {
                soma += dados[i];                 // entra o novo
                soma -= dados[i - janela_K];     // sai o antigo
                medias.push_back(soma / (double)janela_K);
            }
            return medias;
        }

        // (b) Média móvel passando POR REFERÊNCIA 
        std::vector<double> media_movel_por_referencia(const std::vector<double>& dados, std::size_t janela_K) {
            std::vector<double> medias;
            if (janela_K == 0 || janela_K > dados.size()) return medias;

            double soma = 0.0;
            for (std::size_t i = 0; i < janela_K; i++) {
                soma += dados[i];
            }
            medias.push_back(soma / (double)janela_K);

            for (std::size_t i = janela_K; i < dados.size(); i++) {
                soma += dados[i];
                soma -= dados[i - janela_K];
                medias.push_back(soma / (double)janela_K);
            }
            return medias;
        }

        int main(int argc, char** argv) {
            // Parâmetros de entrada
            std::size_t tamanho_N = (argc > 1 ? (std::size_t)std::atoll(argv[1]) : 1000000);
            std::size_t janela_K  = (argc > 2 ? (std::size_t)std::atoll(argv[2]) : 128);

            // Gera dados aleatórios em [0,1)
            std::vector<double> vetor_dados(tamanho_N);
            std::mt19937_64 gerador(42u);
            std::uniform_real_distribution<double> distribuicao(0.0, 1.0);
            for (std::size_t i = 0; i < tamanho_N; i++) {
                vetor_dados[i] = distribuicao(gerador);
            }

            // Mede tempo da versão por VALOR
            std::chrono::high_resolution_clock::time_point inicio_valor = std::chrono::high_resolution_clock::now();
            std::vector<double> medias_valor = media_movel_por_valor(vetor_dados, janela_K);
            std::chrono::high_resolution_clock::time_point fim_valor = std::chrono::high_resolution_clock::now();

            // Mede tempo da versão por REFERÊNCIA
            std::chrono::high_resolution_clock::time_point inicio_referencia = std::chrono::high_resolution_clock::now();
            std::vector<double> medias_referencia = media_movel_por_referencia(vetor_dados, janela_K);
            std::chrono::high_resolution_clock::time_point fim_referencia = std::chrono::high_resolution_clock::now();

            // Cálculo de tempos (ms)
            double tempo_valor_ms      = std::chrono::duration<double, std::milli>(fim_valor - inicio_valor).count();
            double tempo_referencia_ms = std::chrono::duration<double, std::milli>(fim_referencia - inicio_referencia).count();

            // Validação simples (tolerância numérica)
            bool resultados_iguais = (medias_valor.size() == medias_referencia.size());
            for (std::size_t i = 0; resultados_iguais && i < medias_valor.size(); i++) {
                if (std::abs(medias_valor[i] - medias_referencia[i]) > 1e-12) {
                    resultados_iguais = false;
                }
            }

            // Saída
            std::cout << "tempo_valor_ms=" << tempo_valor_ms
                    << " tempo_referencia_ms=" << tempo_referencia_ms
                    << " iguais=" << (resultados_iguais ? 1 : 0) << "\n";

            return 0;
        }


---

**12. Otimização com ponteiros**

Implemente uma função em C++ que calcula a média móvel de um vetor usando **aritmética de ponteiros** em vez de índices.

* Compare o desempenho com a versão que usa índices tradicionais.
* Mostre que `*(ptr + i)` é equivalente a `ptr[i]`.

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

std::vector<double> media_movel_ptr(const double* ptr, std::size_t N, std::size_t K) {
    // TODO: validar ponteiro e K
    // TODO: somar os K primeiros usando *(ptr + i) e empurrar a média
    // TODO: janela deslizante: soma += *(ptr + i) - *(ptr + i - K); push média
    return std::vector<double>();
}

int main(int argc, char** argv) {
    std::size_t N = (argc > 1 ? (std::size_t)std::atoll(argv[1]) : 1000000);
    std::size_t K = (argc > 2 ? (std::size_t)std::atoll(argv[2]) : 256);

    std::vector<double> v(N);
    std::mt19937_64 rng(123u);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (std::size_t i = 0; i < N; i++) v[i] = U(rng);

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> mv = media_movel_ptr(v.data(), v.size(), K);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::cout << "ptr_ms=" << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " out_size=" << mv.size() << "\n";

    // TODO: comparar com versão por referência/índices
    return 0;
}
```

??? note "Ver Resposta"

        #include <iostream>
        #include <vector>
        #include <random>
        #include <chrono>
        #include <cmath>

        // Função: calcula a média móvel de tamanho janela_K usando ponteiros
        std::vector<double> media_movel_ptr(const double* dados_ptr, size_t tamanho_N, size_t janela_K) {
            // Validação de entrada
            if (dados_ptr == nullptr) return std::vector<double>();
            if (janela_K == 0 || janela_K > tamanho_N) return std::vector<double>();

            std::vector<double> medias;  // vetor de saída
            double soma = 0.0;           // acumulador da janela

            // Soma inicial dos primeiros K elementos
            for (size_t i = 0; i < janela_K; i++) {
                soma += *(dados_ptr + i);   // equivalente a dados_ptr[i]
            }
            medias.push_back(soma / (double)janela_K);

            // Janela deslizante: entra um elemento novo, sai o mais antigo
            for (size_t i = janela_K; i < tamanho_N; i++) {
                soma += *(dados_ptr + i);           // adiciona o novo
                soma -= *(dados_ptr + i - janela_K); // remove o antigo
                medias.push_back(soma / (double)janela_K);
            }

            return medias;
        }

        int main(int argc, char** argv) {
            // Tamanho do vetor e janela definidos pela linha de comando ou valores padrão
            size_t tamanho_N = (argc > 1 ? (size_t)std::atoll(argv[1]) : 1000000);
            size_t janela_K  = (argc > 2 ? (size_t)std::atoll(argv[2]) : 256);

            // Cria vetor de dados aleatórios no intervalo [0, 1)
            std::vector<double> vetor_dados(tamanho_N);
            std::mt19937 gerador(123);  // gerador pseudoaleatório com semente fixa
            std::uniform_real_distribution<double> distribuicao(0.0, 1.0);

            for (size_t i = 0; i < tamanho_N; i++) {
                vetor_dados[i] = distribuicao(gerador);
            }

            // Mede tempo da versão com ponteiros
            auto inicio = std::chrono::high_resolution_clock::now();
            std::vector<double> medias = media_movel_ptr(vetor_dados.data(), vetor_dados.size(), janela_K);
            auto fim = std::chrono::high_resolution_clock::now();

            double tempo_ms = std::chrono::duration<double, std::milli>(fim - inicio).count();
            std::cout << "tempo_ptr_ms=" << tempo_ms
                    << " tamanho_saida=" << medias.size() << "\n";

            return 0;
        }
