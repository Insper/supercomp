#include <vector>
#include <random>

int busca_com_filtro(const std::vector<int>& v, int alvo, unsigned long long maxTentativas) {
    if (v.empty()) return -1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, v.size() - 1);

    // Heurística: só verifica índices pares
    for (unsigned long long t = 0; t < maxTentativas; t++) {
        size_t idx = dist(gen);
        if (idx % 2 != 0) continue; // só posições pares
        if (v[idx] == alvo) return (int)idx;
    }

    // fallback: busca linear completa
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] == alvo) return (int)i;
    }
    return -1; // não encontrado
}
