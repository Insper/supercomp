#include <vector>
#include <random>

int busca_linear(const std::vector<int>& v, int alvo) {
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] == alvo) return (int)i;
    }
    return -1; // não encontrado
}

int busca_aleatoria(const std::vector<int>& v, int alvo, unsigned long long maxTentativas) {
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
