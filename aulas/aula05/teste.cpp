#include <iostream>     // Entrada e saída (cout, cin)
#include <vector>       
#include <cmath>        // Função sqrt para cálculo de distância
#include <limits>       // numeric_limits (usado para infinito)
#include <cstdlib>      // atoi (converter string para inteiro)
#include <chrono>       // Medição de tempo de execução

using namespace std;

/*
Capacidade máxima da moto.
Ela só pode transportar 5 itens por viagem.
Quando zera, precisa voltar ao ponto de coleta.
*/
const int CAPACIDADE_MOTO = 5;

/*
Estrutura que representa um ponto no plano cartesiano.
Cada ponto tem coordenadas (x, y).
*/
struct Ponto {
    double x;
    double y;
};

/*
Função que calcula a distância euclidiana entre dois pontos.
*/
double distancia(const Ponto& a, const Ponto& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

/*
Função que calcula o custo total de uma rota completa.

- O motorista sai de casa.
- Vai até o ponto de coleta.
- Realiza as entregas na ordem definida em "rota".
- A cada 5 entregas precisa voltar na coleta.
- Ao final de todas as entregas, retorna para casa.
*/
double calcularCusto(const Ponto& motorista,
                    const Ponto& coleta,
                    const vector<Ponto>& entregas,
                    const vector<int>& rota) {

    double custo = 0.0;

    // 1) Motorista vai até o ponto de coleta
    custo += distancia(motorista, coleta);

    int cargaAtual = CAPACIDADE_MOTO;   // Quantos itens ainda cabem na moto
    Ponto posicaoAtual = coleta;        // Começa na coleta

    // 2) Percorre todas as entregas na ordem do vetor
    for (size_t i = 0; i < rota.size(); i++) {

        // Se a carga acabou, volta para coleta para reabastecer
        if (cargaAtual == 0) {
            custo += distancia(posicaoAtual, coleta);
            posicaoAtual = coleta;
            cargaAtual = CAPACIDADE_MOTO;
        }

        // Vai até o próximo ponto de entrega
        custo += distancia(posicaoAtual, entregas[rota[i]]);
        posicaoAtual = entregas[rota[i]];
        cargaAtual--; // Um item foi entregue
    }

    // 3) Após a última entrega, retorna para casa
    custo += distancia(posicaoAtual, motorista);

    return custo;
}

/*
Função recursiva que gera TODAS as possibilidades possíveis
(busca exaustiva).

- rota: vetor que representa a ordem atual das entregas
- inicio: posição atual da recursão
- melhorCusto: melhor custo encontrado até agora
- melhorRota: rota correspondente ao melhor custo
*/
void permutar(const Ponto& motorista,
            const Ponto& coleta,
            const vector<Ponto>& entregas,
            vector<int>& rota,
            int inicio,
            double& melhorCusto,
            vector<int>& melhorRota) {

    // Caso base:
    // Se "inicio" chegou ao final, significa que temos o vetor preenchido com os pontos
    if (inicio == rota.size()) {

        // Calcula o custo da rota completa
        double custo = calcularCusto(motorista,
                                    coleta,
                                    entregas,
                                    rota);

        // Se essa rota for melhor que a atual melhor, atualiza
        if (custo < melhorCusto) {
            melhorCusto = custo;
            melhorRota = rota;
        }

        return;
    }

    /*
    Geração das permutações:
    Fixamos o elemento da posição "inicio"
    e permutamos os elementos seguintes.
    */
    for (size_t i = inicio; i < rota.size(); i++) {

        // Troca o elemento atual
        swap(rota[inicio], rota[i]);

        // Chamada recursiva para próxima posição
        permutar(motorista,
                coleta,
                entregas,
                rota,
                inicio + 1,
                melhorCusto,
                melhorRota);

        // Desfaz a troca 
        swap(rota[inicio], rota[i]);
    }
}

int main(int argc, char* argv[]) {

    /*
    O programa recebe como parâmetro
    a quantidade de pontos de entrega.

    Exemplo:
    ./exausto 8
    */

    int n = atoi(argv[1]);

    if (n <= 0 || n > 20) {
        cout << "Escolha um numero entre 1 e 20.\n";
        return 1;
    }

    // Posição fixa do motorista (casa)
    Ponto motorista{0,0};

    // Posição fixa do ponto de coleta
    Ponto coleta{5,5};

    /*
    Lista dos pontos de entrega
    */
    vector<Ponto> todos = {
        {10,10}, {20,10}, {30,10}, {40,10}, {50,10},
        {10,20}, {20,20}, {30,20}, {40,20}, {50,20},
        {10,30}, {20,30}, {30,30}, {40,30}, {50,30},
        {10,40}, {20,40}, {30,40}, {40,40}, {50,40}
    };

    // Seleciona apenas os pontos até o N escolhido
    vector<Ponto> entregas(todos.begin(), todos.begin() + n);

    // Vetor da rota das entregas
    vector<int> rota(n);

    // Vetor da melhor rota 
    vector<int> melhorRota;


    // Inicializa o índice dos pontos na ordem [1,2,3 .. N]
    for (int i = 0; i < n; i++)
        rota[i] = i;

    // Inicializa melhor custo como infinito
    double melhorCusto = numeric_limits<double>::max();

    // Início da medição de tempo
    auto inicio = chrono::high_resolution_clock::now();

    // Executa busca exaustiva
    permutar(motorista,
            coleta,
            entregas,
            rota,
            0,
            melhorCusto,
            melhorRota);

    // Fim da medição
    auto fim = chrono::high_resolution_clock::now();
    chrono::duration<double> tempo = fim - inicio;

    // Impressão da melhor rota encontrada
    cout << "\nMelhor rota encontrada:\n";
    cout << "Motorista(0,0) -> ";

    int cargaAtual = CAPACIDADE_MOTO;
    cout << "Coleta(5,5) -> ";

    for (size_t i = 0; i < melhorRota.size(); i++) {

        // Se zerar a carga, volta na coleta
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

    // Exibe custo e tempo total
    cout << "\nCusto total: " << melhorCusto << endl;
    cout << "Tempo de execucao: "
        << tempo.count()
        << " segundos\n";

    return 0;
}