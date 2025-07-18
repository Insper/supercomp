#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

// Constantes
const size_t N = 100'000'000;
const size_t K = 10;

// Gera vetor com valores entre 12.0 e 189.98
vector<double> gerar_leituras(size_t tamanho) {
    vector<double> dados(tamanho);
    default_random_engine gerador(42);  // Seed fixa
    uniform_real_distribution<double> distribuicao(12.0, 189.98);
    for (auto& valor : dados) {
        valor = distribuicao(gerador);
    }
    return dados;
}

// Média móvel - passagem por valor
vector<double> media_movel_por_valor(vector<double> dados, size_t k) {
    vector<double> resultado;
    resultado.reserve(dados.size() - k + 1);
    double soma = 0.0;

    for (size_t i = 0; i < k; ++i) soma += dados[i];
    resultado.push_back(soma / k);

    for (size_t i = 1; i <= dados.size() - k; ++i) {
        soma = soma - dados[i - 1] + dados[i + k - 1];
        resultado.push_back(soma / k);
    }
    return resultado;
}

// Média móvel - passagem por referência
vector<double> media_movel_por_referencia(const vector<double>& dados, size_t k) {
    vector<double> resultado;
    resultado.reserve(dados.size() - k + 1);
    double soma = 0.0;

    for (size_t i = 0; i < k; ++i) soma += dados[i];
    resultado.push_back(soma / k);

    for (size_t i = 1; i <= dados.size() - k; ++i) {
        soma = soma - dados[i - 1] + dados[i + k - 1];
        resultado.push_back(soma / k);
    }
    return resultado;
}

// Média móvel - passagem por ponteiro
vector<double> media_movel_por_ponteiro(const double* dados, size_t tamanho, size_t k) {
    vector<double> resultado;
    resultado.reserve(tamanho - k + 1);
    double soma = 0.0;

    for (size_t i = 0; i < k; ++i) soma += dados[i];
    resultado.push_back(soma / k);

    for (size_t i = 1; i <= tamanho - k; ++i) {
        soma = soma - dados[i - 1] + dados[i + k - 1];
        resultado.push_back(soma / k);
    }
    return resultado;
}

// Medição de tempo
template <typename Func, typename... Args>
double medir_tempo(Func funcao, Args&&... args) {
    auto inicio = chrono::high_resolution_clock::now();
    funcao(forward<Args>(args)...);
    auto fim = chrono::high_resolution_clock::now();
    chrono::duration<double> duracao = fim - inicio;
    return duracao.count();
}

int main() {
    cout << "Gerando dados..." << endl;
    vector<double> leituras = gerar_leituras(N);

    cout << "\n[1] Media movel por valor:" << endl;
    double t_valor = medir_tempo(media_movel_por_valor, leituras, K);
    cout << "Tempo: " << t_valor << " s" << endl;

    cout << "\n[2] MMedia movel por referencia:" << endl;
    double t_ref = medir_tempo(media_movel_por_referencia, leituras, K);
    cout << "Tempo: " << t_ref << " s" << endl;

    cout << "\n[3] MMedia movel por ponteiro:" << endl;
    const double* ptr = leituras.data();
    double t_ptr = medir_tempo(media_movel_por_ponteiro, ptr, N, K);
    cout << "Tempo: " << t_ptr << " s" << endl;

    return 0;
}
