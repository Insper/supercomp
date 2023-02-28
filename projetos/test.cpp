#include <chrono>
#include <random>
#include <fstream>
#include <boost/random.hpp>

using namespace std;

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    ofstream inputFile;
    inputFile.open("input.txt");
    inputFile << n << " " << m << endl;

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);

    // Definindo distribuição normal com média de 3 e desvio padrão de 1
    normal_distribution<double> distribution_dif(3, 1.0);

    uniform_int_distribution<int> distribution_hr(0, 23);
    uniform_int_distribution<int> distribution_cat(1, m);

    vector<int> maxFilmes(m); // Vetor para armazenar o máximo de filmes por categoria
    for (int i = 0; i < m; i++) {
        maxFilmes[i] = distribution_cat(generator); // Gerando o máximo de filmes para cada categoria
        inputFile << maxFilmes[i] << " "; // Escrevendo o valor no arquivo de entrada
    }
    inputFile << endl;

    for (int i = 0; i < n; i++) {
        int hora_inicio = distribution_hr(generator);
        double dif_media = distribution_dif(generator);
        int hora_fim = ((int)hora_inicio + (int)round(dif_media)) % 24;
        int categoria = distribution_cat(generator);

        inputFile << hora_inicio << " " << hora_fim << " " << categoria << endl;
    }

   
    inputFile.close();
    return 0;
}