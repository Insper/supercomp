#include <iostream>
#include <algorithm>
#include <fstream> //Biblioteca pra trabalhar com arquivos
#include <vector>
#include <list>

using namespace std;
vector<vector<int>> LerGrafo(const string &nomeArquivo, int &numVertices);
vector<int> EncontrarCliqueMaximo(vector<vector<int>> &grafo, int numVertices);

int main()
{
  int numVertices = 0;
  string nomeArquivo;
  
  vector<vector<int>> grafo = LerGrafo(nomeArquivo, numVertices);
  vector<int> cliqueMaxima = EncontrarCliqueMaximo(grafo, numVertices);
  cout << "[";
  for (auto elemento : cliqueMaxima)
  {
    cout << elemento << " ";
  }
  cout << "]" << endl;
  return 0;
}

// Função para ler o grafo a partir do arquivo de entrada
vector<vector<int>> LerGrafo(const string &nomeArquivo, int &numVertices)
{
  //ifstream arquivo(nomeArquivo);
  int numArestas;
  //arquivo >> numVertices >> numArestas;

  cin >> numVertices >> numArestas;
  cout << "Num Arestas: " << numArestas << endl;
  vector<vector<int>> grafo(numVertices, vector<int>(numVertices, 0));

  for (int i = 0; i < (numArestas - 1); ++i)
  {
    int u, v;
    cin >> u >> v;
    grafo[u - 1][v - 1] = 1;
    grafo[v - 1][u - 1] = 1; // O grafo é não direcionado
  }

  //arquivo.close();

  return grafo;
}
vector<int> EncontrarCliqueMaximo(vector<vector<int>> &grafo, int numVertices)
{
  vector<int> cliqueMaxima, candidatos, novosCandidatos;
  int v = 0;
  bool podeAdicionar;

  cout << "Candidatos: ";
  for (int i = 0; i < (numVertices - 1); i++)
  {
    int vertice = i+1;
    candidatos.push_back(vertice);
    cout << candidatos[i] << " ";
  }
    cout << endl;

  while (candidatos.size() > 0)
  {
    v = candidatos.back();
    candidatos.pop_back();
    podeAdicionar = true;

    for (int u : cliqueMaxima)
    { 
      cout << "Grafo[" << u << "][" << v <<"]: "
           << grafo[u][v] << endl;
      if (grafo[u][v] == 0)
      {
        podeAdicionar = false;
        break;
      }
    }

    if (podeAdicionar == true)
    {
      cliqueMaxima.push_back(v);
      vector<int> novosCandidatos;

      for (int u : candidatos)
      {
        bool adjacenteATodos = true;

        for (int c : cliqueMaxima)
        {
          if (grafo[u][c] == 0)
          {
            adjacenteATodos = false;
            break;
          }
        }
        if (adjacenteATodos == true)
        {
          novosCandidatos.push_back(u);
        }
      }
      candidatos = novosCandidatos;
    }
  }
  return cliqueMaxima;
}
