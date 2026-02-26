## Busca Exaustiva
Na aula passada você deveria ter sumido com os erros do código fornecido, e a ideia era que chegasse em algo parecido com isso:
??? note "Gabarito - Busca Exaustiva"

    O código  `exausto.cpp` implementa a busca exaustiva bonitinho:

    ```cpp
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
    ```

Na busca exaustiva, o algoritmo investe tempo explorando caminhos que já dá pra saber que não levarão a uma solução melhor.

É aqui que entra o conceito de poda.

## Aplicando a poda (Branch and Bound)

Poda é uma técnica usada em algoritmos de busca para evitar explorar soluções que não melhoram o resultado atual.

A ideia central é simples:

Se uma solução parcial já tem custo maior que o melhor custo encontrado até agora, então não faz sentido continuar expandindo esse caminho.

Você interrompe essa ramificação da recursão.

O algoritmo continua correto (ainda encontra o ótimo), mas evita explorar regiões inúteis do espaço de busca.

## Desafio!

**Objetivo:** Analisar e aprimorar a heurística exemplo.

Melhore a heuristica implementando a poda. Você precisará modificar três coisas principais:

### 1) Passar custo parcial na recursão

Atualmente o código calcula o custo só no final.

Você deve:

* Passar um parâmetro `custoParcial`
* Atualizar esse custo a cada nível da recursão


### 2) Fazer o teste de poda antes da chamada recursiva

Logo após calcular o novo custo parcial, inserir algo como:

se (custoParcial >= melhorCusto)
retorna;

Para impedir a expansão da ramificação inutil.

## 6. Estrutura da nova recursão

Fluxo lógico:

1. Se custo parcial ≥ melhor custo → poda
2. Se completou todas as entregas → atualiza melhor custo
3. Para cada próxima entrega possível:

   * calcula novo custo parcial
   * atualiza estado (posição, carga)
   * chama recursivamente



Teste o seu algorítimo pelo menos 3 vezes para **para N = 13** e responda:

* A poda (Branch and Bound) sempre melhora o desempenho em relação à busca exaustiva? 


## Hill Climbing

Hill Climbing, ou Subida da Encosta, é um algoritmo de busca local utilizado para resolver problemas de otimização em que o espaço de soluções é muito grande. A ideia central é simples: a cada passo, tenta-se melhorar a solução explorando pequenas modificações. Sempre que uma solução vizinha apresenta custo menor, o algoritmo se move para ela. Esse processo continua até que nenhuma melhoria seja encontrada. Nesse ponto, o algoritmo para, assumindo que atingiu um ponto de máximo ou mínimo local.

A principal vantagem dessa abordagem é que ela evita a explosão combinatória típica da busca exaustiva. Enquanto a busca completa avalia todas as possíveis combinações, que cresce exponencialmente com o número de entregas, Hill Climbing explora apenas uma pequena parte do espaço de busca, focando em melhorar progressivamente uma solução. Isso torna o método mais escalável para valores maiores de N, onde a busca exaustiva se torna inviável.


```cpp
// Hill Climbing
vector<int> hillClimbing(const Ponto& motorista,
                         const Ponto& coleta,
                         const vector<Ponto>& entregas) {

    int n = entregas.size();

    // Solução inicial sequencial
    vector<int> atual(n);
    for (int i = 0; i < n; i++)
        atual[i] = i;

    double melhorCusto = calcularCusto(motorista, coleta, entregas, atual);

    bool melhorou = true;

    while (melhorou) {

        melhorou = false;
        vector<int> melhorVizinho = atual;

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {

                vector<int> vizinho = atual;
                swap(vizinho[i], vizinho[j]);

                double custoVizinho = calcularCusto(motorista, coleta, entregas, vizinho);

                if (custoVizinho < melhorCusto) {
                    melhorCusto = custoVizinho;
                    melhorVizinho = vizinho;
                    melhorou = true;
                }
            }
        }

        atual = melhorVizinho;
    }

    return atual;
}
```

## Aleatoriedade
Se utilizarmos aleatoriedade para escolher as rotas iniciais antes de aplicar a heurística poderiamos encontrar as soluções melhores mais rapidamente?


Quando usamos Hill Climbing “puro”, ele é determinístico: dada a mesma solução inicial e a mesma regra de vizinhança, ele sempre percorre o mesmo caminho e para no mesmo ponto. Isso é polêmico, o algorítimo pode sempre cair nos mesmos minimos locais

A aleatoriedade entra justamente para quebrar esse comportamento rígido.

Em vez de sempre começar da mesma solução ou explorar vizinhos em uma ordem fixa, colocamos elementos aleatórios para diversificar a exploração do espaço de busca. Isso ajuda o algoritmo a escapar de mínimos locais, explorar regiões diferentes do espaço e, estatisticamente, encontrar soluções melhores.


A aleatoriedade pode ajudar nesses pontos:

Primeiro, na solução inicial. Em vez de começar sempre com a ordem 0,1,2,...,N-1, você pode gerar uma permutação aleatória. Isso faz com que cada execução comece em um ponto diferente. Esse é o conceito de Multi-Start: executar Hill Climbing várias vezes com soluções iniciais diferentes e guardar a melhor solução encontrada.

Segundo, na escolha da vizinhança. Em vez de testar todos os vizinhos possíveis e escolher o melhor, você pode sortear dois índices aleatórios e testar apenas essa troca. Se melhorar, aceita. Caso contrário, tenta outra. Isso reduz o custo por iteração e torna a trajetória imprevisível.

## Desafio!!!

**1 — Tornar a solução inicial aleatória**
Em vez de preencher o vetor sequencialmente, pense em como embaralhar esse vetor usando uma função de randomização pronta do C++.


**2 — Alterar a geração de vizinhos**
Em vez de dois loops aninhados testando todas as trocas possíveis (i,j), tente:

    sortear dois índices distintos

    trocar temporariamente

    calcular o custo

    decidir se aceita

Pergunta importante: como evitar recalcular o custo completo da rota a cada pequena troca? Existe alguma forma incremental?

**4 — Implementar múltiplos recomeços**
Crie um laço externo que execute Hill Climbing várias vezes.
A cada execução:
    
    gere uma solução inicial aleatória
    
    execute o Hill Climbing
    
    compare com a melhor solução global

Pergunta: como você deve armazenar a melhor solução global sem fazer cópias desnecessárias a cada iteração?

**5 — Definir critério de parada**
Cuidado para não ficar preso eternamente nos loops, um critério de parada é fundamental. Qual critério faz mais sentido neste contexto?

A ideia central é que aleatoriedade não é “bagunça”. Ela é um mecanismo controlado para aumentar diversidade na exploração do espaço de soluções. Em problemas combinatórios grandes, isso quase sempre melhora a qualidade média das soluções encontradas, mesmo que não garanta a ótima global.



## Para analizar as implementações:

Faça os testes para **N = 10, 11, 12 e 13**

1- Comparando Busca Exaustiva otimizada, Branch and Bound, Hill Climbing puro e O Hill Climbing aleatório, qual apresentou melhor escalabilidade conforme N aumentou? 


2 - O Hill Climbing puro encontrou a mesma solução que a busca exaustiva? O que será que aconteceu?


3 - Usando O Hill Climbing aleatório, melhorou a qualidade média das soluções? Houve aumento significativo no tempo total? Analise o custo-benefício.


4 Se você tivesse que resolver o problema para N = 100, qual abordagem escolheria e por quê? Considere tempo, qualidade da solução e escalabilidade.



[Submeta a sua entrega pelo Classroom diponível neste link até 06/03 ás 14h00](https://classroom.github.com/a/XI5jzrP-)

A entrega deve conter o seu algorítmo e as suas análises (pode deixar as análises no README.md) 



# Código completo com o Hill Climbing


```cpp
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

// Distância Euclidiana
double distancia(const Ponto& a, const Ponto& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Calcula custo total da rota
double calcularCusto(const Ponto& motorista,
                     const Ponto& coleta,
                     const vector<Ponto>& entregas,
                     const vector<int>& rota) {

    double custo = 0.0;

    custo += distancia(motorista, coleta);

    int carga = CAPACIDADE_MOTO;
    Ponto atual = coleta;

    for (int i = 0; i < rota.size(); i++) {

        if (carga == 0) {
            custo += distancia(atual, coleta);
            atual = coleta;
            carga = CAPACIDADE_MOTO;
        }

        const Ponto& destino = entregas[rota[i]];
        custo += distancia(atual, destino);
        atual = destino;
        carga--;
    }

    custo += distancia(atual, motorista);

    return custo;
}

// Hill Climbing
vector<int> hillClimbing(const Ponto& motorista,
                         const Ponto& coleta,
                         const vector<Ponto>& entregas) {

    int n = entregas.size();

    // Solução inicial sequencial
    vector<int> atual(n);
    for (int i = 0; i < n; i++)
        atual[i] = i;

    double melhorCusto = calcularCusto(motorista, coleta, entregas, atual);

    bool melhorou = true;

    while (melhorou) {

        melhorou = false;
        vector<int> melhorVizinho = atual;

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {

                vector<int> vizinho = atual;
                swap(vizinho[i], vizinho[j]);

                double custoVizinho = calcularCusto(motorista, coleta, entregas, vizinho);

                if (custoVizinho < melhorCusto) {
                    melhorCusto = custoVizinho;
                    melhorVizinho = vizinho;
                    melhorou = true;
                }
            }
        }

        atual = melhorVizinho;
    }

    return atual;
}

int main(int argc, char* argv[]) {

    if (argc < 2) {
        cout << "Uso: ./hill N\n";
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

    auto inicio = chrono::high_resolution_clock::now();

    vector<int> melhorRota = hillClimbing(motorista, coleta, entregas);

    double melhorCusto = calcularCusto(motorista, coleta, entregas, melhorRota);

    auto fim = chrono::high_resolution_clock::now();
    chrono::duration<double> tempo = fim - inicio;

    // Impressão da rota
    cout << "\nRota encontrada:\n";
    cout << "Motorista(0,0) -> Coleta(5,5) -> ";

    int carga = CAPACIDADE_MOTO;

    for (int i = 0; i < melhorRota.size(); i++) {

        if (carga == 0) {
            cout << "Coleta(5,5) -> ";
            carga = CAPACIDADE_MOTO;
        }

        int idx = melhorRota[i];
        cout << "P" << idx
             << "(" << entregas[idx].x
             << "," << entregas[idx].y << ") -> ";

        carga--;
    }

    cout << "Motorista(0,0)\n";

    cout << "\nCusto total: " << melhorCusto << endl;
    cout << "Tempo de execucao: "
         << tempo.count()
         << " segundos\n";

    return 0;
}

```
