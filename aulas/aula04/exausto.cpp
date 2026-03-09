#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <chrono>

using namespace std;

const int CAPACIDADE_MOTO = 5;

struct Ponto {
    double x;
    double y;
};

double distancia(const Ponto& a, const Ponto& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

/*
Agora o custo considera que:
- A cada 5 entregas a moto deve voltar na coleta.
*/
double calcularCusto(const Ponto& motorista,
                     const Ponto& coleta,
                     const vector<Ponto>& entregas,
                     const vector<int>& rota) {

    double custo = 0.0;

    // motorista -> coleta
    custo += distancia(motorista, coleta);

    int cargaAtual = CAPACIDADE_MOTO;
    Ponto posicaoAtual = coleta;

    for (size_t i = 0; i < rota.size(); i++) {

        // Se acabou a carga, volta para coleta
        if (cargaAtual == 0) {
            custo += distancia(posicaoAtual, coleta);
            posicaoAtual = coleta;
            cargaAtual = CAPACIDADE_MOTO;
        }

        // Vai até próxima entrega
        custo += distancia(posicaoAtual, entregas[rota[i]]);
        posicaoAtual = entregas[rota[i]];
        cargaAtual--;
    }

    // Após última entrega, volta para motorista
    custo += distancia(posicaoAtual, motorista);

    return custo;
}

void permutar(const Ponto& motorista,
              const Ponto& coleta,
              const vector<Ponto>& entregas,
              vector<int>& rota,
              int inicio,
              double& melhorCusto,
              vector<int>& melhorRota) {

    if (inicio == rota.size()) {

        double custo = calcularCusto(motorista,
                                     coleta,
                                     entregas,
                                     rota);

        if (custo < melhorCusto) {
            melhorCusto = custo;
            melhorRota = rota;
        }

        return;
    }

    for (size_t i = inicio; i < rota.size(); i++) {

        swap(rota[inicio], rota[i]);

        permutar(motorista,
                 coleta,
                 entregas,
                 rota,
                 inicio + 1,
                 melhorCusto,
                 melhorRota);

        swap(rota[inicio], rota[i]);
    }
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        cout << "Uso: ./exausto <numero_de_entregas>\n";
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 0 || n > 20) {
        cout << "Escolha um numero entre 1 e 20.\n";
        return 1;
    }

    Ponto motorista{0,0};
    Ponto coleta{5,5};

    vector<Ponto> todos = {
        {10,10}, {20,10}, {30,10}, {40,10}, {50,10},
        {10,20}, {20,20}, {30,20}, {40,20}, {50,20},
        {10,30}, {20,30}, {30,30}, {40,30}, {50,30},
        {10,40}, {20,40}, {30,40}, {40,40}, {50,40}
    };

    vector<Ponto> entregas(todos.begin(), todos.begin() + n);

    vector<int> rota(n);
    vector<int> melhorRota;

    for (int i = 0; i < n; i++)
        rota[i] = i;

    double melhorCusto = numeric_limits<double>::max();

    auto inicio = chrono::high_resolution_clock::now();

    permutar(motorista,
             coleta,
             entregas,
             rota,
             0,
             melhorCusto,
             melhorRota);

    auto fim = chrono::high_resolution_clock::now();
    chrono::duration<double> tempo = fim - inicio;

    cout << "\nMelhor rota encontrada:\n";
    cout << "Motorista(0,0) -> ";

    int cargaAtual = CAPACIDADE_MOTO;
    cout << "Coleta(5,5) -> ";

    for (size_t i = 0; i < melhorRota.size(); i++) {

        if (cargaAtual == 0) {
            cout << "Coleta(5,5) -> ";
            cargaAtual = CAPACIDADE_MOTO;
        }

        int idx = melhorRota[i];
        cout << "P" << idx
             << "(" << entregas[idx].x
             << "," << entregas[idx].y << ") -> ";

        cargaAtual--;
    }

    cout << "Motorista(0,0)\n";

    cout << "\nCusto total: " << melhorCusto << endl;
    cout << "Tempo de execucao: "
         << tempo.count()
         << " segundos\n";

    return 0;
}