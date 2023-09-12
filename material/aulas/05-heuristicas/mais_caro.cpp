#include <iostream>
#include <algorithm>
#include <fstream> //Biblioteca pra trabalhar com arquivos
#include <vector>

using namespace std;

struct Item
{
  int id;
  int peso;
  int valor;
};

void carregaDados(int& N, int& W, vector<Item>& itens);
bool comparaValor(Item &a, Item &b) { return a.valor > b.valor; };
bool sortSaida(int& a, int& b) { return a < b ;};

int main()
{
  int n_lines, max_w;
  int count = 0;   // Contador de controle do loop
  int w_atual = 0; // Peso da mochila no momento
  int v_atual = 0; // Valor da mochila no momento
  int q_itens = 0; // Quantidade de itens dentro da mochila
  int opt; // 0 se for usada uma heurística ou busca local
           // 1 se a solução for ótimo global

  vector<int> saida;  // Vetor de saída do problema
  vector<Item> itens; // Vetor que armazena os itens da mochila

  string linha; // Guarda os vlores lidos na linha do arquivo

  carregaDados(n_lines, max_w, itens);

  // Ordena os valores dentro do vetor de itens
  sort(itens.begin(), itens.end(), comparaValor);

  // Seleciona e ordena os itens que ficam dentro da mochila
  for (int i = 0; i < n_lines; i++)
  {
    if (w_atual + itens[i].peso <= max_w)
    {
      saida.push_back(itens[i].id);
      w_atual += itens[i].peso;
      v_atual += itens[i].valor;
      q_itens++;
    }
  }
  opt = 0; 
  sort(saida.begin(), saida.end(), sortSaida);
  cout << w_atual << " " << v_atual << " " << opt << "\n";
  for (int i = 0; i < q_itens; i++)
  {
    cout << saida[i] << " ";
  }
  cout << "\n";
  return 0;
}

void carregaDados(int &N, int &W, vector<Item> &itens)
{
  double w, v;
  struct Item temp; // Estrutura temporária de itens
  cin >> N >> W;
  for (int i = 0; i < N; i++)
  {
    cin >> w >> v;
    temp.id = i;
    temp.peso = w;
    temp.valor = v;
    itens.push_back(temp);
  }
}