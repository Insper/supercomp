# O Laboratório que Ficou Preso em um Loop

Era uma manhã chuvosa no Laboratório Egghead, o centro de pesquisa mais avançado do mundo.
O cientista Edson Vegapunk, trabalhava em um projeto ousado:
**criar Akuma no Mi artificiais**, as lendárias frutas que concedem poderes sobre-humanos.

Para isso, ele precisava calcular o **score de compatibilidade genética** de milhares de amostras de DNA coletadas de marinheiros, piratas e até de alguns animais mutantes.

Seu algoritmo era simples (pelo menos para ele):
para cada amostra, aplicar uma função matemática complexa envolvendo **trigonometria**, **exponenciais** e **raízes quadradas**, um algoritmo que simulava o quanto o DNA era compatível com uma fruta mística.

Mas havia um problema.
O processamento **demorava mais de 4 horas** para rodar uma única leva de dados.

Enquanto o código rodava, Vegapunk olhava para o monitor e murmurava:

> “Se eu continuar nesse ritmo... o Luffy já vai ter comido todas antes de eu terminar o protótipo!”

```python
# score.py
import math, random, time

def score(x):
    return math.sin(x)**2 + math.sqrt(abs(x)) + math.exp(-x**2 / 50)

data = [random.uniform(0, 1000) for _ in range(10_000_000)]
start = time.time()

results = [score(x) for x in data]
end = time.time()
print(f"Score médio: {sum(results)/len(results):.4f}")
print(f"Tempo total: {end - start:.2f}s")
```

---

Edson percebeu algo curioso.
Seu computador tinha **20 núcleos**, mas apenas **um** estava em uso.

Usando seu conhecimento, ele modificou o código para que cada núcleo da CPU processasse uma parte das amostras.

> “Com isso, teremos vários núcleos trabalhando em paralelo!”

```python
#score_paralelo.py
import math
import random
import time
import multiprocessing as mp

# --- Função de cálculo pesado ---
def score(x):
    """Simula o cálculo de um score computacionalmente intensivo."""
    return math.sin(x)**2 + math.sqrt(abs(x)) + math.exp(-x**2 / 50)

# --- Função que executa um teste com N processos ---
def run_test(nproc, N=10_000_00):
    """Executa o cálculo com nproc processos e retorna o tempo e o score médio."""
    data = [random.uniform(0, 1000) for _ in range(N)]
    start = time.time()
    with mp.Pool(processes=nproc) as pool:
        results = pool.map(score, data)
    end = time.time()
    total_time = end - start
    avg_score = sum(results) / len(results)
    return total_time, avg_score

# --- Execução principal ---
if __name__ == "__main__":
    test_procs = [1, 2, 4, 8, 16]
    results = []

    print("\n=== Teste de Speedup ===")
    print(f"[INFO] CPUs detectadas: {mp.cpu_count()}")
    print(f"[INFO] Dataset: 10 milhões de amostras\n")

    # --- Rodar o caso base (1 CPU) ---
    print("[INFO] Medindo tempo base (1 CPU)...")
    base_time, base_score = run_test(1)
    print(f"[BASE] Tempo com 1 CPU: {base_time:.2f}s | Score médio: {base_score:.4f}")
    print("-" * 50)

    # --- Testes com múltiplos processos ---
    for n in test_procs[1:]:  # começa a partir de 2
        t, s = run_test(n)
        speedup = base_time / t
        results.append((n, t, speedup))
        print(f"{n:2d} CPUs → {t:7.2f}s | Speedup={speedup:6.2f}×")

    # --- Resumo final ---
    print("\n=== Resumo ===")
    print(f"{'Nproc':>5} | {'Tempo (s)':>10} | {'Speedup':>8}")
    print("-" * 30)
    print(f"{1:5d} | {base_time:10.2f} | {1.00:8.2f}")
    for n, t, sp in results:
        print(f"{n:5d} | {t:10.2f} | {sp:8.2f}")

   
```

Vamos testar no Cluster Franky para ver o resultado:

```bash
#!/bin/bash
#SBATCH --job-name=score-speedup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --output=logs/score_%j.out

python score_paralelo.py

```

Quando o experimento terminou, o laboratório inteiro vibrou.
Na tela principal, o log do Cluster Franky mostrava:

```
16 CPUs →   14.87s | Speedup= 13.27× | Eficiência= 82.9%
```

Edson abriu um sorriso.

> “Maravilha! O cálculo que levava quatro horas agora termina em minutos.
> O poder das Akuma no Mi artificiais está cada vez mais próximo!”


O experimento de Vegapunk revelou o que é essencial em Computação de Alto Desempenho:

| Conceito                   | Explicação                                                                                      |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| **Paralelismo**            | Dividir uma grande tarefa em partes menores e executá-las simultaneamente em múltiplos núcleos. |
| **Speedup (S = T₁ / Tₙ)**  | Mede o quanto o programa ficou mais rápido com N processadores.                                 |
| **Eficiência (E = S / N)** | Mede o quanto cada CPU contribuiu de fato para o ganho total.                                   |
| **Overhead**               | A comunicação e coordenação entre processos que impede o ganho linear perfeito.                 |

Mesmo com 16 núcleos, a eficiência não chegou a 100%.
Parte do tempo foi gasta na **sincronização e gerenciamento de processos**, dificilmente um código pode ser paralelizado completamente.

![Imagem](edsion_lilith.jpg)