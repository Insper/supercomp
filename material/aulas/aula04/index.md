# O que é busca exaustiva

Busca exaustiva é a estratégia que testa todas as soluções possíveis e escolher a melhor.

Considere um serviço de entregas de comércio eletrônico, como as realizadas pelo Mercado Livre. Um motorista inicia a rota da sua casa, vai até um ponto de coleta para retirar as encomendas e, a partir daí, deve realizar entregas em diversos pontos distribuídos pela cidade. Após concluir todas as entregas, ele retorna ao local de origem.

O objetivo é encontrar a melhor rota com o menor custo total de deslocamento, considerando a localização inicial do motorista, o ponto de coleta e todos os pontos de entrega. Cada ponto de entrega deve ser visitado uma única vez. O custo da rota será definido como a distância total percorrida.


No nosso caso, o problema é encontrar a melhor ordem de entrega dos pontos. Se existem `n` entregas, então existem `n!` possíveis permutações. O algoritmo de busca exaustiva vai:

1. Gera todas as combinações possíveis.
2. Calcula o custo de cada uma.
3. Guardar a melhor.

Essa heuristica sempre encontra a solução ótima, porém, é extremamente cara computacionalmente e não escala.

Analise o código  `exausto.cpp`

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <chrono>
#include <algorithm>

using namespace std;

const int CAPACIDADE_MOTO = 5;

struct Ponto {
    double x;
    double y;
};

/*
POLEMICA:
Passagem por valor + uso de pow (muito mais lento que multiplicação direta)
*/
double distancia(Ponto a, Ponto b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

/*
POLEMICA:
Função cria matriz de distâncias TODA VEZ que é chamada
*/
vector<vector<double>> criarMatrizDistancias(vector<Ponto> pontos) {
    int n = pontos.size();
    vector<vector<double>> matriz(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matriz[i][j] = distancia(pontos[i], pontos[j]);
        }
    }

    return matriz;
}

/*
POLEMICA
Recebe vetores por valor (cópia os dados a cada chamada)
*/
double calcularCusto(Ponto motorista,
                     Ponto coleta,
                     vector<Ponto> entregas,
                     vector<int> rota) {

    double custo = 0.0;

    // POLEMICA:
    // Cria vetor auxiliar desnecessário
    vector<Ponto> todosPontos = entregas;
    todosPontos.push_back(coleta);

    // POLEMICA:
    // Recria matriz inteira a cada cálculo
    vector<vector<double>> matriz = criarMatrizDistancias(todosPontos);

    custo += distancia(motorista, coleta);

    int carga = CAPACIDADE_MOTO;
    Ponto atual = coleta;

    for (int i = 0; i < rota.size(); i++) {

        if (carga == 0) {
            custo += distancia(atual, coleta);
            atual = coleta;
            carga = CAPACIDADE_MOTO;
        }

        // POLEMICA:
        // Acesso indireto ruim para cache
        Ponto destino = entregas.at(rota.at(i));

        custo += distancia(atual, destino);

        // POLEMICA:
        // Ordenação inútil dentro do loop
        sort(entregas.begin(), entregas.end(),
            [](Ponto a, Ponto b) {
                return a.x < b.x;
            });

        atual = destino;
        carga--;
    }

    custo += distancia(atual, motorista);

    // POLEMICA:
    // Loop inútil que não altera nada
    for (int i = 0; i < 1000; i++) {
        custo += 0;
    }

    return custo;
}

/*
POLEMICA:
Recursão recebe tudo por valor
melhorCusto também por valor 
*/
void permutar(Ponto motorista,
              Ponto coleta,
              vector<Ponto> entregas,
              vector<int> rota,
              int inicio,
              double melhorCusto,
              vector<int> melhorRota) {

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

    for (int i = inicio; i < rota.size(); i++) {

        swap(rota[inicio], rota[i]);

        // POLEMICA:
        // Aloca vetor temporário inútil
        vector<int> lixo(rota.begin(), rota.end());

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

    int n = atoi(argv[1]);

    if (n <= 0) {
        cout << "Numero invalido\n";
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

    // Copia elemento por elemento
    vector<Ponto> entregas;
    for (int i = 0; i < n; i++) {
        entregas.push_back(todos[i]);
    }

    vector<int> rota;
    for (int i = 0; i < n; i++) {
        rota.push_back(i);
    }

    vector<int> melhorRota;
    double melhorCusto = numeric_limits<double>::max();

    auto inicioTempo = chrono::high_resolution_clock::now();

    permutar(motorista,
             coleta,
             entregas,
             rota,
             0,
             melhorCusto,
             melhorRota);

    auto fimTempo = chrono::high_resolution_clock::now();
    chrono::duration<double> tempo = fimTempo - inicioTempo;

    cout << "Melhor custo: " << melhorCusto << endl;
    cout << "Tempo: " << tempo.count() << " segundos\n";

    return 0;
}
```

## Desafio!


O código está cheio de problemas, identifique os problemas e otimize o código;


Execute o seu algorítimo otimizado **N = 13 no Cluster Franky** e responda:

1 - Explique de forma breve e objetiva quais otimizações você aplicou no código.

2 - Qual flag de otimização teve o melhor desempenho?

3 - Qual fila foi mais rápida? O que você acha que influenciou esse resultado?

4 - Compare o código base fornecido com a versão otimizada desenvolvida por você. **Execute ambos para N = 9, 10, 11 e 12** e gere um gráfico comparativo. 


[Submeta a sua entrega pelo Classroom diponível neste link até 27/02 ás 23h59](https://classroom.github.com/a/lBx_S1nd)

A entrega deve conter o seu algorítmo e as suas análises (pode deixar as análises no README.md) 

















