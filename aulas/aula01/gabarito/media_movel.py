import time
import random

# N = 100 milhões de leituras
N = 100_000_000

# Janela K = 10
K = 10

# Gera os dados
print("Gerando dados aleatórios...")
start_gen = time.time()
dados = [random.uniform(12.0, 189.98) for _ in range(N)]
print(f"Tempo para gerar os dados: {(time.time() - start_gen):.2f} segundos")

# Calcula média móvel
start_avg = time.time()
media = []
soma = sum(dados[:K])
media.append(soma / K)

for i in range(1, N - K):
    soma = soma - dados[i - 1] + dados[i + K - 1]
    media.append(soma / K)
tempo = f"{time.time() - start_avg:.2f}"

print("Tempo para calcular média móvel:", tempo, "segundos")