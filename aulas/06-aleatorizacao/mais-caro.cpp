#include <iostream>
#include <algorithm>
#include <vector>
#include <random>


struct objeto {
    int id;
    int peso;
    int valor;
};

double numero_aleatorio() {
    char * a = getenv("SEED");
    double seed = atof(a);
    static std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> distribution(0, 1.0);
    return distribution(eng);
}

int main() {
    int N, W;
    std::cin >> N >> W;
    std::vector<objeto> objetos(N);
    
    for (int i = 0; i < N; i++) {
        objetos[i].id = i;
        std::cin >> objetos[i].peso >> objetos[i].valor;
    }

    int peso = 0;
    int valor = 0;
    std::vector<int> resposta;
    resposta.reserve(N);

    std::sort(objetos.begin(), objetos.end(), [](objeto &a, objeto &b) {
        return a.valor > b.valor;
    } );

    
    
    for (int i = 0; i < N; i++) {
        double r = numero_aleatorio();
        std::cout << r << "\n";
        if (objetos[i].peso + peso <= W &&
            r <= 0.75) {
            resposta.push_back(objetos[i].id);
            valor += objetos[i].valor;
            peso += objetos[i].peso;
        }
    }

    std::sort(resposta.begin(), resposta.end());
    std::cout << peso << " " << valor << " 0\n";
    for (int id : resposta) {
        std::cout << id << " ";
    }
    std::cout << "\n";

    return 0;
}
