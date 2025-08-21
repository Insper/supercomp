
## Objetivo

Ao final desta atividade, você será capaz de:

* Analisar **heurísticas** com aleatoriedade para reduzir o espaço de busca e fugir de mínimos locais.
* Usar **aleatoriedade** para guiar a busca de soluções.


## Um pouco de teoria: O que é Nonce?

Um **nonce** é um número que só pode ser usado uma única vez dentro de um determinado contexto. A palavra vem da expressão inglesa *“number used once”*. O nonce tem um papel fundamental dentro do mecanismo conhecido como **Proof of Work**.

No sistema de Proof of Work, que é usado por diversas criptomoedas como o Bitcoin, o objetivo principal é garantir que novos blocos de transações só sejam adicionados à blockchain mediante a realização de um trabalho computacional significativo. Esse trabalho é feito pelos mineradores, que tentam encontrar um valor de nonce que, quando combinado com os dados de um bloco e passado por uma função hash (o SHA-256), gere um resultado que satisfaça uma condição específica de dificuldade. Essa condição normalmente exige que o hash gerado comece com um certo número de zeros, **quanto mais zeros, mais difícil o problema**.

Para encontrar esse nonce, o minerador precisa testar diferentes valores, um a um, recalculando o hash a cada tentativa. Como a função hash é determinística, mas seu resultado parece aleatório mesmo com pequenas mudanças nos dados de entrada, não há como prever qual nonce gerará um hash válido. Isso significa que a única forma de resolver o problema é por tentativa e erro, **o que exige muito poder computacional e tempo.**

---
!!! tip 
    Quer saber mais? Assista [esse vídeo sobre nonce](https://www.youtube.com/watch?v=diwHGOA1_c4&t=6s)


##  Por que usar heurísticas com aleatoriedade?

Em contextos como Proof of Work, usar heurísticas com aleatoriedade é interessante porque o espaço de busca é enorme e imprevisível. A função hash se comporta como uma caixa-preta: pequenas mudanças no nonce geram resultados totalmente diferentes, sem padrão aparente. Isso torna estratégias determinísticas (como testar de 0 em diante) ineficientes e vulneráveis a colisões entre mineradores.

A aleatoriedade permite explorar regiões diferentes do espaço de busca, reduzindo repetição de esforços e aumentando a chance de sucesso. Além disso, torna o processo menos previsível, dificultando ataques ou manipulações. Em um problema onde não há como saber onde está a solução, tentar caminhos variados aleatoriamente é uma boa forma de encontrar uma resposta mais rápido.



Analise o código exemplo `mineracao.cpp`

```cpp
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <functional>
#include <climits>

// -------------------------------------------------------------
// - Retorna string hex com 16 bytes (64 bits) -> 16 dígitos hex em ambientes 64 bits.
// -------------------------------------------------------------
std::string sha256(const std::string& input) {
    std::hash<std::string> hasher;     // Functor de hash da STL
    size_t h = hasher(input);          // Aplica o hash de 64 bits à string
    unsigned long long v = static_cast<unsigned long long>(h); // Normaliza para 64 bits

    // Converte o valor inteiro para string hexadecimal com padding à esquerda
    // Para 64 bits -> 16 dígitos hex (2 por byte * 8 bytes)
    std::ostringstream os;
    os << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << v;
    return os.str(); // Ex.: "00af3c..."; comprimento típico: 16 chars em 64 bits
}

// -------------------------------------------------------------
// Verifica se o "hash" (string hex) começa com 'dificuldade' zeros.
// -------------------------------------------------------------
bool validaHash(const std::string& hash, int dificuldade) {
    if (dificuldade <= 0) return true;               // dificuldade 0 sempre passa
    if (dificuldade > static_cast<int>(hash.size())) // não pode exigir mais zeros do que o tamanho do hash
        return false;
    return hash.rfind(std::string(dificuldade, '0'), 0) == 0; // true se começa com zeros
}

// -------------------------------------------------------------
// Estratégia 1: Busca linear — testa nonces em ordem: 0, 1, 2, ...
// -------------------------------------------------------------
void minerar_linear(const std::string& bloco, int dificuldade) {
    auto start = std::chrono::high_resolution_clock::now(); // Marca início do cronômetro

    unsigned long long nonce = 0;        // Começa do 0
    unsigned long long tentativas = 0;   // Contador de tentativas

    while (true) {
        // Monta a entrada "bloco || nonce"
        std::string tentativa = bloco + std::to_string(nonce);
        // Calcula o "hash" da tentativa
        std::string hash = sha256(tentativa);
        ++tentativas;

        // Se atendeu à dificuldade (começa com N zeros), reporta e encerra
        if (validaHash(hash, dificuldade)) {
            auto end = std::chrono::high_resolution_clock::now();
            double tempo = std::chrono::duration<double>(end - start).count();
            std::cout << "[LINEAR] Nonce: " << nonce << "\nHash: " << hash
                      << "\nTentativas: " << tentativas << "\nTempo: " << tempo << "s\n\n";
            break;
        }
        ++nonce; // Tenta o próximo nonce
    }
}

// -------------------------------------------------------------
// Estratégia 2: Busca aleatória com heurística simples
// - Gera nonces aleatórios em 64 bits
// - Heurística: só segue para validação completa se o hash começar com '0'
// - maxTentativas: evita loop "infinito" 
// -------------------------------------------------------------
void minerar_random_heuristica(const std::string& bloco, int dificuldade, unsigned long long maxTentativas) {
    auto start = std::chrono::high_resolution_clock::now(); // Cronômetro

    // Preparação do gerador aleatório:
    std::random_device rd;                          // Fonte de entropia (seed)
    std::mt19937_64 gen(rd());                      // Gerador de aleatórios
    std::uniform_int_distribution<unsigned long long> distrib(0, ULLONG_MAX); // Uniforme em [0, 2^64-1]

    unsigned long long tentativas = 0; // Contador de tentativas

    while (tentativas < maxTentativas) {
        // Sorteia um nonce (exploração aleatória do espaço de busca)
        unsigned long long nonce = distrib(gen);

        // Monta a tentativa e calcula o hash simulado
        std::string tentativa = bloco + std::to_string(nonce);
        std::string hash = sha256(tentativa);
        ++tentativas;

        // Heurística: se o primeiro dígito não é '0', provavelmente não atende a dificuldades maiores
        if (hash[0] != '0') continue;

        // Verificação completa do critério de dificuldade
        if (validaHash(hash, dificuldade)) {
            auto end = std::chrono::high_resolution_clock::now();
            double tempo = std::chrono::duration<double>(end - start).count();
            std::cout << "[HEURISTICA-RANDOM] Nonce: " << nonce << "\nHash: " << hash
                      << "\nTentativas: " << tentativas << "\nTempo: " << tempo << "s\n\n";
            return; // Sucesso: encerra a função
        }
    }

    // Se não encontrou dentro do limite de tentativas, informa
    std::cout << "[HEURISTICA-RANDOM] Não encontrou nonce válido em " << maxTentativas << " tentativas.\n\n";
}

// -------------------------------------------------------------
// main: executa as duas estratégias para comparação 
// -------------------------------------------------------------
int main() {
    std::string bloco = "transacao_simples"; // Simulação de transação
    int dificuldade = 5;                     // Nº de zeros à esquerda no hash simulado
    unsigned long long limite_random = 500000; // Limite de tentativas para a estratégia aleatória

    std::cout << "=== Mineração  | dificuldade = " << dificuldade << " ===\n\n";

    // Estratégia linear 
    minerar_linear(bloco, dificuldade);

    // Estratégia aleatória + heurística
    minerar_random_heuristica(bloco, dificuldade, limite_random);

    return 0; 
}

```


## Desafio!

**Objetivo:** Analisar e aprimorar a heurística exemplo.

**Tarefa:**

1. Execute o código acima 5 vezes.
2. Compare:

   * O número de tentativas e o tempo da busca linear.
   * O número de tentativas e o tempo da busca aleatória com heurística.
   * Qual das duas abordagem acerta mais?

3. Interprete os resultados:

   * A heurística sempre é mais rápida?
   * Em qual cenário a heuristica aleatória pode ser pior?
   * O que fazer para que a busca aleatória com heurística encontre o nonce com mais frequência?
   * Por que usar aleatoriedade e filtros simples (como descartar hashes que não começam com '0') pode acelerar a busca por um hash válido?

**Pergunta para reflexão:**

> *Quais melhorias poderiam ser implementadas neste algoritmo para ter uma heuristica mais eficiente?*


## **Esta atividade não tem entrega, bom final de semana!**