#include <vector>
#include <random>

int busca_hibrida(const std::vector<int>& v, int alvo, int K, unsigned long long maxTentativas) {
    // Parte 1: busca sequencial nos primeiros K elementos
    for (int i = 0; i < K && i < (int)v.size(); i++) {
        if (v[i] == alvo) return i;
    }

    // Parte 2: busca aleatória no restante
    if (v.empty()) return -1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, v.size() - 1);

    for (unsigned long long t = 0; t < maxTentativas; t++) {
        size_t idx = dist(gen);
        if (v[idx] == alvo) return (int)idx;
    }
    return -1; // não encontrado
}
