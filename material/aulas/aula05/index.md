# Paralelismo em CPU com OpenMP

## Objetivo

* **Paralelismo em CPU**: como dividir o trabalho entre múltiplos *cores*.
* **Threads**: cada thread executa uma parte do trabalho.
* **OpenMP**: diretivas simples em C++ para paralelizar loops e seções de código.
* **Scheduling**: forma como as iterações do loop são distribuídas entre threads (`static`, `dynamic`, `guided`).

## Esqueleto do Código `miner_omp.cpp`

```cpp
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <functional>
#include <climits>
#include <omp.h> // OpenMP

// =============================================================
// Hash: converte string -> 64 bits -> hex
// =============================================================
static std::string hash_simples_hex(const std::string& input) {
    std::hash<std::string> hasher;
    unsigned long long v = static_cast<unsigned long long>(hasher(input));
    std::ostringstream os;
    os << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << v;
    return os.str(); // ~16 hex chars (64 bits)
}

// =============================================================
// Critério de dificuldade: hash deve começar com N zeros
// =============================================================
static bool validaHash(const std::string& h, int dificuldade) {
    if (dificuldade <= 0) return true;
    if ((int)h.size() < dificuldade) return false;
    for (int i = 0; i < dificuldade; ++i) if (h[i] != '0') return false;
    return true;
}

// =============================================================
// Baseline SEQUENCIAL (linear): testa nonce = 0..limite-1
// =============================================================
static void minerar_linear_seq(const std::string& bloco,
                               int dificuldade,
                               unsigned long long limite) {
    auto t0 = std::chrono::high_resolution_clock::now();

    unsigned long long vencedor_nonce = 0;
    std::string vencedor_hash;
    bool found = false;

    for (unsigned long long nonce = 0; nonce < limite; ++nonce) {
        const std::string tentativa = bloco + std::to_string(nonce);
        const std::string h = hash_simples_hex(tentativa);
        if (validaHash(h, dificuldade)) {
            found = true;
            vencedor_nonce = nonce;
            vencedor_hash  = h;
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[SEQ-LINEAR] dif=" << dificuldade
              << " | limite=" << limite
              << " | tempo=" << secs << "s";
    if (found) {
        std::cout << " | nonce=" << vencedor_nonce
                  << " | hash=" << vencedor_hash << "\n";
    } else {
        std::cout << " | (nao encontrou)\n";
    }
}

// =============================================================
// ESQUELETO PARALELO (linear) — OpenMP
// Objetivo: paralelizar o for abaixo.
// =============================================================
static void minerar_linear_omp_skel(const std::string& bloco,
                                    int dificuldade,
                                    unsigned long long limite) {
    auto t0 = std::chrono::high_resolution_clock::now();

    bool found = false;                // compartilhado
    unsigned long long vencedor_nonce = 0; // compartilhado
    std::string vencedor_hash;         // compartilhado

    // TODO: inserir diretiva OpenMP aqui:
    for (unsigned long long nonce = 0; nonce < limite; ++nonce) {
        if (found) continue; 

        // variáveis locais (privadas por padrão no loop)
        const std::string tentativa = bloco + std::to_string(nonce);
        const std::string h = hash_simples_hex(tentativa);

        if (validaHash(h, dificuldade)) {
            // registrar um possível vencedor 
            found = true;
            vencedor_nonce = nonce;
            vencedor_hash  = h;
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[OMP-LINEAR]  dif=" << dificuldade
              << " | limite=" << limite
              << " | tempo=" << secs << "s";
    if (found) {
        std::cout << " | nonce=" << vencedor_nonce
                  << " | hash=" << vencedor_hash << "\n";
    } else {
        std::cout << " | (nao encontrou)\n";
    }
}

// =============================================================
// Baseline SEQUENCIAL (random+heurística):
// - Gera nonce aleatório 64-bit por tentativa (uniforme)
// - Pré-filtro simples: exige primeiro char '0' antes de validar tudo
// =============================================================
static void minerar_random_heuristica_seq(const std::string& bloco,
                                          int dificuldade,
                                          unsigned long long tentativas) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> distrib(0, ULLONG_MAX);

    bool found = false;
    unsigned long long vencedor_nonce = 0;
    std::string vencedor_hash;

    for (unsigned long long i = 0; i < tentativas; ++i) {
        const unsigned long long nonce = distrib(gen);
        const std::string tentativa = bloco + std::to_string(nonce);
        const std::string h = hash_simples_hex(tentativa);

        // Heurística barata (pré-filtro):
        if (h[0] != '0') continue;

        if (validaHash(h, dificuldade)) {
            found = true;
            vencedor_nonce = nonce;
            vencedor_hash  = h;
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[SEQ-RANDH]  dif=" << dificuldade
              << " | tent=" << tentativas
              << " | tempo=" << secs << "s";
    if (found) {
        std::cout << " | nonce=" << vencedor_nonce
                  << " | hash=" << vencedor_hash << "\n";
    } else {
        std::cout << " | (nao encontrou)\n";
    }
}

// =============================================================
// ESQUELETO PARALELO (random+heurística) — OpenMP
// Objetivo: paralelizar o "lote de tentativas" por thread.
// =============================================================
static void minerar_random_heuristica_omp_skel(const std::string& bloco,
                                               int dificuldade,
                                               unsigned long long tentativas_total) {
    auto t0 = std::chrono::high_resolution_clock::now();

    bool found = false;                // compartilhado
    unsigned long long vencedor_nonce = 0; // compartilhado
    std::string vencedor_hash;         // compartilhado

    int T = omp_get_max_threads();     
    unsigned long long quota = (tentativas_total / (T > 0 ? T : 1));

    // TODO: criar região paralela OpenMP
        // TODO: obter id da thread 
        // TODO: RNG por thread (seed distinta; ex.: rd() ^ (constante * tid))
        // TODO: cada thread executa sua quota de tentativas

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[OMP-RANDH]  dif=" << dificuldade
              << " | tent(TOTAL)=" << tentativas_total
              << " | tempo=" << secs << "s";
    if (found) {
        std::cout << " | nonce=" << vencedor_nonce
                  << " | hash=" << vencedor_hash << "\n";
    } else {
        std::cout << " | (nao encontrou)\n";
    }
}

// =============================================================
// argv[1]=dificuldade | argv[2]=limiteLinear | argv[3]=tentativasRandom
// =============================================================
int main(int argc, char** argv) {
    const std::string bloco = "transacao_simples";
    const int dificuldade = (argc >= 2 ? std::stoi(argv[1]) : 5);
    const unsigned long long limiteLinear = (argc >= 3 ? std::stoull(argv[2]) : 500000ULL);
    const unsigned long long tentativasRandom = (argc >= 4 ? std::stoull(argv[3]) : 500000ULL);

    std::cout << "=== Exercício OpenMP (Skeleton) | dif=" << dificuldade << " ===\n\n";

    // Baselines sequenciais
    minerar_linear_seq(bloco, dificuldade, limiteLinear);
    minerar_random_heuristica_seq(bloco, dificuldade, tentativasRandom);

    // Skeletons paralelos (para os alunos completarem)
    minerar_linear_omp_skel(bloco, dificuldade, limiteLinear);
    minerar_random_heuristica_omp_skel(bloco, dificuldade, tentativasRandom);

    return 0;
}
```


## Atividade

1. **Compilar o código com OpenMP**

   ```bash
   g++ -fopenmp miner_omp.cpp -o miner_omp
   ```

2. **Rodar no cluster com SLURM** definindo o número de threads:

   ```bash
   srun -c 4 ./miner_parallel
   ```

   ou

   ```bash
   OMP_NUM_THREADS=8 ./miner_parallel
   ```

3. **Explorar os seguintes pontos**:

   * O que acontece quando aumentamos o número de threads?
   * Qual diferença entre `schedule(static)` e `schedule(dynamic)`?
   * O tempo de execução cai de forma linear com mais threads? Por quê?
   * Como a aleatoriedade influencia os resultados de cada execução?

---

## 5. Objetivo do exercício

Com essa prática, o aluno deve:

* Entender **como aplicar OpenMP** em um problema real.
* Comparar desempenho entre versão sequencial e paralela.
* Refletir sobre os **ganhos e limitações** do paralelismo em CPU.
* Perceber que a mineração de nonces é um exemplo claro de problema "paralelizável", mas que a aleatoriedade e a sincronização (variáveis compartilhadas) afetam o resultado.

