#include <iostream>
#include <vector>

using namespace std;

int main() {

    vector<int> dados;

    cout << "=== ESTADO INICIAL ===" << endl;
    cout << "Tamanho: " << dados.size() << endl;
    cout << "Capacidade: " << dados.capacity() << endl;

    size_t capacidade_anterior = dados.capacity();

    cout << "\n=== INSERINDO ELEMENTOS ===\n" << endl;
    int dado = 1;
    for (int i = 0; i < 23; ++i) {
        dado = i * 5; 
        cout << "Inserindo dado: " << dado << endl;
        dados.push_back(dado);

        if (dados.capacity() != capacidade_anterior) {

            cout << ">> Capacidade mudou para: "
                 << dados.capacity() << endl << endl;

            capacidade_anterior = dados.capacity();
        }
    }

    cout << "\n=== ESTADO FINAL ===" << endl;
    cout << "Tamanho final: " << dados.size() << endl;
    cout << "Capacidade final: " << dados.capacity() << endl;

    cout << "\n=== ALOCACAO DOS DADOS ===" << endl;

    cout << "| ";
    for (size_t i = 0; i < dados.size(); ++i) {
        cout << dados[i] << " | ";
    }
    cout << endl;

    return 0;
}
