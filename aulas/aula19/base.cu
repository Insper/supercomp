#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include "base.hpp"

//----------------------------------------------------------------------------
// KERNEL find2dMean Calcula a média de cada coluna na GPU
//----------------------------------------------------------------------------
__global__ void find2dMeanKernel(float* d_matrix, float* d_avg, int numLoops, int timeSteps)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < timeSteps)
    {
        float sum = 0.0f;
        for (int row = 0; row < numLoops; row++)
        {
            // Acesso à matriz linearizada [linha * total_colunas + coluna]
            sum += d_matrix[row * timeSteps + col];
        }
        d_avg[col] = sum / numLoops;
    }
}

//----------------------------------------------------------------------------
// FUNÇÃO que gerencia os dados da CPU para GPU 
//----------------------------------------------------------------------------
float* find2dMeanGPU(float* h_matrix, int numLoops, int timeSteps)
{
    size_t matrixSize = numLoops * timeSteps * sizeof(float);
    size_t avgSize = timeSteps * sizeof(float);
    float *d_matrix, *d_avg;
    float* h_avg = new float[timeSteps];

    // 1. Alocar memória na GPU
    cudaMalloc(&d_matrix, matrixSize);
    cudaMalloc(&d_avg, avgSize);

    // 2. Copiar dados da CPU para a GPU
    cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);

    // 4. Executar o kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (timeSteps + threadsPerBlock - 1) / threadsPerBlock;
    find2dMeanKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_avg, numLoops, timeSteps);

    // 5. Copiar o resultado da GPU para a CPU
    cudaMemcpy(h_avg, d_avg, avgSize, cudaMemcpyDeviceToHost);

    // 6. Liberar a memória da GPU
    cudaFree(d_matrix);
    cudaFree(d_avg);

    return h_avg;
}


//----------------------------------------------------------------------------
// Calcula a volatilidade a partir do arquivo data.csv
//----------------------------------------------------------------------------
float calculateVolatility(float spotPrice, int32_t timeSteps)
{
    // Abre o arquivo data.csv em modo de leitura, encerra em caso de falha
    std::ifstream filePtr;
    filePtr.open("nvidia_stock.txt", std::ifstream::in);

    if (!filePtr.is_open())
    {
        std::cerr << "Não foi possível abrir o arquivo! Encerrando..\n";
        exit(EXIT_FAILURE);
    }

    int32_t i = 0;
    int32_t maxLen = timeSteps - 1;
    std::unique_ptr<float[]> priceArr = std::make_unique<float[]>(maxLen);
    std::string line;

    while( std::getline(filePtr, line) && (i < maxLen) ){
        if(line.empty()) continue;

        try {
                priceArr[i] = std::stof(line);
                i++;

        } catch (const std::exception& e){
                std::cerr << "Erro ao converter o valor da linha " << i <<"\n";
                continue;
        }

    }
    filePtr.close();

    float sum = spotPrice;
    // Encontra a média dos preços estimados no final de cada minuto
    for (i = 0; i < maxLen; i++){
        sum += priceArr[i];
    }
    float meanPrice = sum / (maxLen + 1);

    // Calcula a volatilidade do mercado como o desvio padrão
    sum = std::pow((spotPrice - meanPrice), 2.0f);
    for (i = 0; i < maxLen; i++){
        sum += std::pow((priceArr[i] - meanPrice), 2.0f);
    }

    float stdDev = std::sqrt(sum / maxLen); // Nota: Fórmula do desvio padrão amostral requer divisão pelo tamanho

    // Retorna como porcentagem
    return stdDev / 100.0f;
}


/** ---------------------------------------------------------------------------
    Gera um número aleatório semeado pelo relógio do sistema baseado na
    distribuição normal padrão assumindo média 0.0 e desvio padrão 1.0
----------------------------------------------------------------------------*/
float genRand(float mean, float stdDev)
{
    const auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(static_cast<uint32_t>(seed));
    std::normal_distribution<float> distribution(mean, stdDev);
    return distribution(generator);
}

//----------------------------------------------------------------------------
// Simula o modelo de Black-Scholes
//----------------------------------------------------------------------------
float* runBlackScholesModel(float spotPrice, int32_t timeSteps, float riskRate, float volatility)
{
    // Média e desvio padrão
    static constexpr float  mean = 0.0f, stdDev = 1.0f;
    // Intervalo de tempo (Timestep)
    float  deltaT = 1.0f / timeSteps;
    std::unique_ptr<float[]> normRand = std::make_unique<float[]>(timeSteps - 1); // Array de números aleatórios distribuídos normalmente
    float* stockPrice = new float[timeSteps];                                     // Array do preço da ação em diferentes tempos
    stockPrice[0] = spotPrice;                                                    // O preço da ação em t=0 é o preço à vista (spot price)

    // Preenche o array com números aleatórios
    for (int32_t i = 0; i < timeSteps - 1; i++)
        normRand[i] = genRand(mean, stdDev);

    // Aplica a equação de Black-Scholes para calcular o preço da ação no próximo passo de tempo
    for (int32_t i = 0; i < timeSteps - 1; i++)
        stockPrice[i + 1] = stockPrice[i] * exp(((riskRate - (std::pow(volatility, 2.0f) / 2.0f)) * deltaT) + (volatility * normRand[i] * std::sqrt(deltaT)));

    return stockPrice;
}


void displayHelp(char* programName) {
    std::cout << "\n========================================================================\n";
    std::cout << "   SIMULADOR DE MONTE CARLO - MODELO BLACK-SCHOLES (NVIDIA PREDICT)\n";
    std::cout << "========================================================================\n\n";
    std::cout << "Uso: " << programName << " [opcionais]\n\n";
    std::cout << "Sintaxe dos Argumentos:\n";
    std::cout << "  1. [inLoops]           Iterações internas para redução de variância (Padrão: 100)\n";
    std::cout << "  2. [outLoops]          Total de simulações de Monte Carlo (Padrão: 10000)\n";
    std::cout << "  3. [timeStepsHistory]  Janela de dados históricos para cálculo de volatilidade (Padrão: 180)\n";
    std::cout << "  4. [timeStepsForecast] Horizonte de projeção da série temporal (Padrão: 180)\n";
    std::cout << "  5. [spotPrice]         Preço atual do ativo (S_0) no tempo zero (Padrão: 0.50)\n";
    std::cout << "  6. [riskRate]          Taxa livre de risco anualizada (Risk-free rate) (Padrão: 0.5)\n\n";

    std::cout << "Descrição Técnica:\n";
    std::cout << "  Este software realiza uma simulação estocástica baseada no Movimento \n";
    std::cout << "  Browniano Geométrico. Ele processa dados históricos de fechamento para \n";
    std::cout << "  derivar a volatilidade implícita e projeta N caminhos possíveis, \n";
    std::cout << "  consolidando-os em um resultado estatisticamente convergente (opt.csv).\n\n";

    std::cout << "Exemplo de execução avançada:\n";
    std::cout << "  " << programName << " 500 50000 252 30 145.20 0.05\n";
    std::cout << "  (Simula 30 dias com 25 milhões de trajetórias totais)\n";
    std::cout << "========================================================================\n\n";
}


/* ============================================================================
   APLICAÇÃO DO MÉTODO DE MONTE CARLO
   ----------------------------------
   O Método de Monte Carlo utiliza amostragens aleatórias massivas para modelar
   sistemas complexos (que não possuem uma solução exata simples) e estimar os
   seus resultados através de probabilidade.

   Neste código, ele é aplicado da seguinte forma:
   1. Aleatoriedade (Caminhos): O modelo gera múltiplos cenários ou "caminhos"
      possíveis para o preço da ação ao longo do tempo usando a equação estocástica
      de Black-Scholes (movimento browniano geométrico), impulsionada pela
      função de números aleatórios 'genRand'.
   2. Repetição: Executamos milhares dessas simulações independentes através de
      laços de repetição aninhados (controlados por 'inLoops' e 'outLoops').
   3. Agregação: Ao final, calculamos a média (find2dMean) de todos esses caminhos
      simulados. Pela Lei dos Grandes Números, a média destas simulações converge
      para o valor esperado ou o "resultado mais provável" do comportamento da
      ação ao longo do tempo em um mercado volátil.
============================================================================ */

//----------------------------------------------------------------------------
// Função Principal
//----------------------------------------------------------------------------


int32_t run(int32_t inLoops, int32_t outLoops, int32_t timeStepsHistory,
            int32_t timeStepsForecast, float spotPrice, float riskRate)
{
    // Agora usamos memória linear para facilitar a transferência para GPU
    float* h_stockFlat = new float[inLoops * timeStepsForecast];
    float* h_avgStockFlat = new float[outLoops * timeStepsForecast];

    const float volatility = calculateVolatility(spotPrice, timeStepsHistory);

    std::cout << "Usando volatilidade de mercado: " << volatility << "\n";
    std::cout << "Executando aceleração via GPU (Kernel find2dMean)\n\n";

    for (int32_t i = 0; i < outLoops; i++)
    {
        for (int32_t j = 0; j < inLoops; j++)
        {
            float* path = runBlackScholesModel(spotPrice, timeStepsForecast, riskRate, volatility);
            // Copia o caminho para a matriz linearizada
            for(int k=0; k < timeStepsForecast; k++) {
                h_stockFlat[j * timeStepsForecast + k] = path[k];
            }
            delete[] path;
        }

        // CHAMA A GPU para processar as médias do bloco interno
        float* avgResult = find2dMeanGPU(h_stockFlat, inLoops, timeStepsForecast);
        
        // Armazena na matriz de médias global
        for(int k=0; k < timeStepsForecast; k++) {
            h_avgStockFlat[i * timeStepsForecast + k] = avgResult[k];
        }
        delete[] avgResult;
    }

    // Calcula o resultado final (Média das Médias) também na GPU
    float* optStock = find2dMeanGPU(h_avgStockFlat, outLoops, timeStepsForecast);

    // Grava o resultado
    std::ofstream filePtr("opt.csv");
    if (filePtr.is_open()) {
        for (int32_t i = 0; i < timeStepsForecast; i++)
            filePtr << optStock[i] << "\n";
        filePtr.close();
    }

    delete[] h_stockFlat;
    delete[] h_avgStockFlat;
    delete[] optStock;

    return EXIT_SUCCESS;
}

int32_t timed_run(int32_t inLoops, int32_t outLoops, int32_t timeStepsHistory,
         int32_t timeStepsForecast, float_t spotPrice, float_t riskRate){
    const auto beginTime = std::chrono::steady_clock::now();

    if(run(inLoops, outLoops, timeStepsHistory, timeStepsForecast, spotPrice, riskRate)){
        return -1;
    }

    const auto endTime = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime).count();
    return elapsed;
}


int main(int argc, char** argv)
{
    static int inLoops = 100;           // Iterações do laço interno
    static int outLoops = 100;        // Iterações do laço externo
    static int timeStepsHistory = 100;  // Intervalos de tempo do mercado de ações (dias)
    static int timeStepsForecast = 100; // Intervalos de tempo do mercado de ações (dias)
    static float spotPrice = 0.50f;       // Preço à vista (spot price em t = 0)
    static float riskRate = 0.5;          // Taxa de juros livre de risco (%)

    if ( (argc != 1 ) && ( argc != 7 )){
        displayHelp(argv[0]);
        return 0;
    }

    inLoops = std::stoi(argv[1]);
    outLoops = std::stoi(argv[2]);
    timeStepsHistory = std::stoi(argv[3]);
    timeStepsForecast = std::stoi(argv[4]);
    spotPrice = std::stof(argv[5]);
    riskRate = std::stof(argv[6]);

    std::cout << "--- Configurações da Simulação ---" << "\n";
    std::cout << "Laços Internos: " << inLoops << "\n";
    std::cout << "Laços Externos: " << outLoops << "\n";
    std::cout << "Histórico (dias): " << timeStepsHistory << "\n";
    std::cout << "Previsão (dias): " << timeStepsForecast << "\n";
    std::cout << "Preço Inicial: " << spotPrice << "\n";
    std::cout << "Taxa de Risco: " << riskRate << "\n";
    std::cout << "----------------------------------" << "\n";


    int elapsed = timed_run(inLoops, outLoops, timeStepsHistory, timeStepsForecast, spotPrice, riskRate);

    std::cout << "Executado em " << elapsed << " s\n";


    return 0;
}
