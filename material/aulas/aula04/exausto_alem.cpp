#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// 1. calcula custo Branchless
// 2. precalculated matrix

using namespace std;

const int CAPACIDADE_MOTO = 5;

struct Ponto {
  double x;
  double y;
};

inline double distancia_pre(int a, int b){

    alignas(64) static const double DISTS[484] = {
    0.0000, 7.0711, 14.1421, 22.3607, 31.6228, 41.2311, 50.9902, 22.3607, 28.2843, 36.0555, 44.7214, 53.8516, 31.6228, 36.0555, 42.4264, 50.0000, 58.3095, 41.2311, 44.7214, 50.0000, 56.5685, 64.0312, // Row 0
    7.0711, 0.0000, 7.0711, 15.8114, 25.4951, 35.3553, 45.2769, 15.8114, 21.2132, 29.1548, 38.0789, 47.4342, 25.4951, 29.1548, 35.3553, 43.0116, 51.4782, 35.3553, 38.0789, 43.0116, 49.4975, 57.0088, // Row 1
    14.1421, 7.0711, 0.0000, 10.0000, 20.0000, 30.0000, 40.0000, 10.0000, 14.1421, 22.3607, 31.6228, 41.2311, 20.0000, 22.3607, 28.2843, 36.0555, 44.7214, 30.0000, 31.6228, 36.0555, 42.4264, 50.0000, // Row 2
    22.3607, 15.8114, 10.0000, 0.0000, 10.0000, 20.0000, 30.0000, 14.1421, 10.0000, 14.1421, 22.3607, 31.6228, 22.3607, 20.0000, 22.3607, 28.2843, 36.0555, 31.6228, 30.0000, 31.6228, 36.0555, 42.4264, // Row 3
    31.6228, 25.4951, 20.0000, 10.0000, 0.0000, 10.0000, 20.0000, 22.3607, 14.1421, 10.0000, 14.1421, 22.3607, 28.2843, 22.3607, 20.0000, 22.3607, 28.2843, 36.0555, 31.6228, 30.0000, 31.6228, 36.0555, // Row 4
    41.2311, 35.3553, 30.0000, 20.0000, 10.0000, 0.0000, 10.0000, 31.6228, 22.3607, 14.1421, 10.0000, 14.1421, 36.0555, 28.2843, 22.3607, 20.0000, 22.3607, 42.4264, 36.0555, 31.6228, 30.0000, 31.6228, // Row 5
    50.9902, 45.2769, 40.0000, 30.0000, 20.0000, 10.0000, 0.0000, 41.2311, 31.6228, 22.3607, 14.1421, 10.0000, 44.7214, 36.0555, 28.2843, 22.3607, 20.0000, 50.0000, 42.4264, 36.0555, 31.6228, 30.0000, // Row 6
    22.3607, 15.8114, 10.0000, 14.1421, 22.3607, 31.6228, 41.2311, 0.0000, 10.0000, 20.0000, 30.0000, 40.0000, 10.0000, 14.1421, 22.3607, 31.6228, 41.2311, 20.0000, 22.3607, 28.2843, 36.0555, 44.7214, // Row 7
    28.2843, 21.2132, 14.1421, 10.0000, 14.1421, 22.3607, 31.6228, 10.0000, 0.0000, 10.0000, 20.0000, 30.0000, 14.1421, 10.0000, 14.1421, 22.3607, 31.6228, 22.3607, 20.0000, 22.3607, 28.2843, 36.0555, // Row 8
    36.0555, 29.1548, 22.3607, 14.1421, 10.0000, 14.1421, 22.3607, 20.0000, 10.0000, 0.0000, 10.0000, 20.0000, 22.3607, 14.1421, 10.0000, 14.1421, 22.3607, 28.2843, 22.3607, 20.0000, 22.3607, 28.2843, // Row 9
    44.7214, 38.0789, 31.6228, 22.3607, 14.1421, 10.0000, 14.1421, 30.0000, 20.0000, 10.0000, 0.0000, 10.0000, 31.6228, 22.3607, 14.1421, 10.0000, 14.1421, 36.0555, 28.2843, 22.3607, 20.0000, 22.3607, // Row 10
    53.8516, 47.4342, 41.2311, 31.6228, 22.3607, 14.1421, 10.0000, 40.0000, 30.0000, 20.0000, 10.0000, 0.0000, 41.2311, 31.6228, 22.3607, 14.1421, 10.0000, 44.7214, 36.0555, 28.2843, 22.3607, 20.0000, // Row 11
    31.6228, 25.4951, 20.0000, 22.3607, 28.2843, 36.0555, 44.7214, 10.0000, 14.1421, 22.3607, 31.6228, 41.2311, 0.0000, 10.0000, 20.0000, 30.0000, 40.0000, 10.0000, 14.1421, 22.3607, 31.6228, 41.2311, // Row 12
    36.0555, 29.1548, 22.3607, 20.0000, 22.3607, 28.2843, 36.0555, 14.1421, 10.0000, 14.1421, 22.3607, 31.6228, 10.0000, 0.0000, 10.0000, 20.0000, 30.0000, 14.1421, 10.0000, 14.1421, 22.3607, 31.6228, // Row 13
    42.4264, 35.3553, 28.2843, 22.3607, 20.0000, 22.3607, 28.2843, 22.3607, 14.1421, 10.0000, 14.1421, 22.3607, 20.0000, 10.0000, 0.0000, 10.0000, 20.0000, 22.3607, 14.1421, 10.0000, 14.1421, 22.3607, // Row 14
    50.0000, 43.0116, 36.0555, 28.2843, 22.3607, 20.0000, 22.3607, 31.6228, 22.3607, 14.1421, 10.0000, 14.1421, 30.0000, 20.0000, 10.0000, 0.0000, 10.0000, 31.6228, 22.3607, 14.1421, 10.0000, 14.1421, // Row 15
    58.3095, 51.4782, 44.7214, 36.0555, 28.2843, 22.3607, 20.0000, 41.2311, 31.6228, 22.3607, 14.1421, 10.0000, 40.0000, 30.0000, 20.0000, 10.0000, 0.0000, 41.2311, 31.6228, 22.3607, 14.1421, 10.0000, // Row 16
    41.2311, 35.3553, 30.0000, 31.6228, 36.0555, 42.4264, 50.0000, 20.0000, 22.3607, 28.2843, 36.0555, 44.7214, 10.0000, 14.1421, 22.3607, 31.6228, 41.2311, 0.0000, 10.0000, 20.0000, 30.0000, 40.0000, // Row 17
    44.7214, 38.0789, 31.6228, 30.0000, 31.6228, 36.0555, 42.4264, 22.3607, 20.0000, 22.3607, 28.2843, 36.0555, 14.1421, 10.0000, 14.1421, 22.3607, 31.6228, 10.0000, 0.0000, 10.0000, 20.0000, 30.0000, // Row 18
    50.0000, 43.0116, 36.0555, 31.6228, 30.0000, 31.6228, 36.0555, 28.2843, 22.3607, 20.0000, 22.3607, 28.2843, 22.3607, 14.1421, 10.0000, 14.1421, 22.3607, 20.0000, 10.0000, 0.0000, 10.0000, 20.0000, // Row 19
    56.5685, 49.4975, 42.4264, 36.0555, 31.6228, 30.0000, 31.6228, 36.0555, 28.2843, 22.3607, 20.0000, 22.3607, 31.6228, 22.3607, 14.1421, 10.0000, 14.1421, 30.0000, 20.0000, 10.0000, 0.0000, 10.0000, // Row 20
    64.0312, 57.0088, 50.0000, 42.4264, 36.0555, 31.6228, 30.0000, 44.7214, 36.0555, 28.2843, 22.3607, 20.0000, 41.2311, 31.6228, 22.3607, 14.1421, 10.0000, 40.0000, 30.0000, 20.0000, 10.0000, 0.0000 // Row 21
};

    return DISTS[a * 22 + b];
}

/* Removi a referência pois o struct é muito leve e é mais rapido a cópia do que
 * a dereferênca.
 */
inline double distancia(Ponto a, Ponto b) {

  double u = a.x - b.x;
  double v = a.y - b.y;

  return sqrt(u * u + v * v);
}

static double calcularCusto(const int motorista, const int coleta, vector<Ponto> & __restrict entregas,
                     vector<int>& __restrict rota) {

  double custo = 0.0;
  custo += distancia_pre(motorista, coleta);
  int carga = CAPACIDADE_MOTO;
  int atual = coleta;

  for (int i = 0; i < rota.size(); i++) {

    bool vazia = (carga == 0);

    int p_entrega = rota[i];

    custo += vazia ? distancia_pre(atual, coleta) +  distancia_pre(coleta, p_entrega): distancia_pre(atual, p_entrega);
    atual  = p_entrega;
    carga  = vazia ? CAPACIDADE_MOTO - 1 : carga-1;

  }

  custo += distancia_pre(atual, motorista);

  return custo;
}

/*
https://en.wikipedia.org/wiki/Heap%27s_algorithm
*/
static void permutarItter(const int motorista, const int coleta, vector<Ponto>& __restrict entregas,
                   vector<int>& rota, int inicio, double &melhorCusto,
                   vector<int>& melhorRota) {

  vector<int> stack_state(rota.size(), 0);

  double custo = calcularCusto(motorista, coleta, entregas, rota);

  if (custo < melhorCusto) {
    melhorCusto = custo;
    melhorRota = rota;
  }

  // Estudar std::next_iteration
  int i = 0;
  for (; i < rota.size();) {
    if (stack_state[i] < i) {
      if (i % 2 == 0) {
        swap(rota[0], rota[i]);
      } else {
        swap(rota[stack_state[i]], rota[i]);
      }

      {
        custo = calcularCusto(motorista, coleta, entregas, rota);

        if (custo < melhorCusto) {
          melhorCusto = custo;
          melhorRota = rota;
        }
      }
      stack_state[i]++;
      i = 0;
    }
    else{
        stack_state[i] = 0;
        i++;
    }
  }
}

int main(int argc, char *argv[]) {

    if(argc < 2){
        cout << "Uso invalido, faltou numero\n";
        return 1;
    }

  int n = atoi(argv[1]);

  if (n <= 0) {
    cout << "Numero invalido\n";
    return 1;
  }

  Ponto motorista{0, 0};
  Ponto coleta{5, 5};

  vector<Ponto> pontos = { motorista, coleta,
                          {10, 10}, {20, 10}, {30, 10}, {40, 10}, {50, 10},
                          {10, 20}, {20, 20}, {30, 20}, {40, 20}, {50, 20},
                          {10, 30}, {20, 30}, {30, 30}, {40, 30}, {50, 30},
                          {10, 40}, {20, 40}, {30, 40}, {40, 40}, {50, 40}};

  // Copia elemento por elemento
  // Se for sempre assim, posso só passar um limitante para a permutar
  vector<Ponto> entregas;
  for (int i = 0; i < n+2; i++) {
    entregas.push_back(pontos[i]);
  }

  vector<int> rota;
  for (int i = 2; i < n+2; i++) {
    rota.push_back(i);
  }

  vector<int> melhorRota;
  double melhorCusto = numeric_limits<double>::max();

  auto inicioTempo = chrono::high_resolution_clock::now();
    
  permutarItter(0, 1, entregas, rota, 0, melhorCusto, melhorRota);

  auto fimTempo = chrono::high_resolution_clock::now();
  chrono::duration<double> tempo = fimTempo - inicioTempo;

  cout << "Melhor custo: " << melhorCusto << endl;
  cout << "Tempo: " << tempo.count() << " segundos\n";

  return 0;
}
