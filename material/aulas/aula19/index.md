# Simulando a  Prova Final


## Exercício 1

Implemente um programa que recebe um vetor de floats e realiza **duas etapas de processamento em GPU**:

1. Calcule o prefix sum (soma cumulativa) de todos os elementos.
   Exemplo: `[2, 1, 3, 4] → [2, 3, 6, 10]`.

2. Use o último elemento (a soma total) para normalizar todos os valores:
   `[2, 3, 6, 10] → [0.2, 0.3, 0.6, 1.0]`.

| Critério                                   | Descrição                                                                                                                                       | Peso     |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| Compilação sem erros   | O código compila corretamente com `nvcc`.  | **0.2**  |
| Implementação em GPU Síncrona         | Programação paralela em GPU de forma síncrona. | **+0.6** |
| Implementação em GPU Assíncrona        | Utiliza streams, otimiza o uso de CPU e GPU   | **+0.9** |
| Uso correto do Slurm no Cluster Franky |Configurou corretamente o ambiente HPC (via `srun` ou `sbatch`), com parâmetros adequados de GPU. | **+0.3** |
| **Total**                                  |                                                                                                                                                 | **2.0**  |


```cpp
#include <iostream>
#include <vector>
#include <chrono>

// SCAN + NORMALIZAÇÃO 

int main() {
    const size_t N = 87654;
    std::vector<float> v(N);
    std::vector<float> prefix(N);

    auto inicio = std::chrono::high_resolution_clock::now();

    // Gera o vetor
    for (size_t i = 0; i < N; ++i)
        v[i] = static_cast<float>((i + 1) * 2);

    // prefix sum
    prefix[0] = v[0];
    for (size_t i = 1; i < N; ++i)
        prefix[i] = prefix[i - 1] + v[i];

    // Normalização
    float total = prefix.back();  // soma total
    for (size_t i = 0; i < N; ++i)
        prefix[i] /= total;

    auto fim = std::chrono::high_resolution_clock::now();
    double tempo = std::chrono::duration<double, std::milli>(fim - inicio).count();

    // RESULTADOS
    std::cout << "\nÚltimos 10 valores do prefix sum:\n[ ";
    for (size_t i = N - 10; i < N; ++i)
        std::cout << prefix[i] * total << " ";
    std::cout << "]\n";

    std::cout << "Últimos 10 valores normalizados:\n[ ";
    for (size_t i = N - 10; i < N; ++i)
        std::cout << prefix[i] << " ";
    std::cout << "]\n";

    std::cout << "\nTempo CPU: " << tempo << " ms\n";
}

```


## Exercício 

Paralelize esse código que calcula a média harmônica dos elementos de um vetor em GPU:

### **Rubrica**
| Critério                                                | Descrição                                                                                                                                                     | Peso     |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | 
| **Compilação sem erros** | O código compila corretamente usando `nvcc`, sem erros | **0.2**  |
| **Implementação em GPU Síncrona**  | O código é paralelizado corretamente em GPU de forma assíncrona. | **+0.6** |
| **Implementação em GPU Assíncrona** | O código utiliza *streams*, com sobreposição de operações entre GPU e CPU. | **+0.9** | 
|**Uso correto do SLURM no Cluster Franky**  | Configurou corretamente o ambiente HPC (via `srun` ou `sbatch`), com parâmetros adequados de GPU.| **+0.3** | 
| **Total**  |                    | **2.0**  | 



```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>


struct Resultado {
    std::vector<double> valores;
    long long soma;
    double tempo_ms;
};

Resultado gerar_valores_e_somar(int N) {
    Resultado r;
    r.soma = 0;
    r.valores.resize(N);

    auto inicio = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        r.valores[i] = i + 1;
        r.soma += i + 1;
    }

    auto fim = std::chrono::high_resolution_clock::now();
    r.tempo_ms = std::chrono::duration<double, std::milli>(fim - inicio).count();

    return r;
}


int main() {
    const int N = 89878;
    double soma_inversos = 0.0;

    std::cout << "Calculando resultados para N = " << N << "...\n\n";

    // Gera vetor e calcula somatória (tudo em uma função)
    Resultado dados = gerar_valores_e_somar(N);

    // MÉDIA HARMÔNICA
    auto inicio_h = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < dados.valores.size(); ++i) {
        double x = dados.valores[i];
        if (x != 0.0)
            soma_inversos += 1.0 / x;
        else
            soma_inversos += 0.0;
    }

    double H = 0.0;
    if (soma_inversos != 0.0)
        H = dados.valores.size() / soma_inversos;

    auto fim_h = std::chrono::high_resolution_clock::now();
    double tempo_h = std::chrono::duration<double, std::milli>(fim_h - inicio_h).count();

    // =========================================================
    // RESULTADOS
    // =========================================================
    std::cout << "Somatória de 1 até " << N << ": " << dados.soma << "\n";
    std::cout << "Média harmônica de 1 até " << N << ": " << H << "\n\n";
    std::cout << "Tempo da geração e somatória: " << dados.tempo_ms << " ms\n";
    std::cout << "Tempo da média harmônica: " << tempo_h << " ms\n";

    return 0;
}

```
