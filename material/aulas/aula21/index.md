### 2. Acesso Irregular à Memória

O melhor exemplo desse problema aparece na função `findComponents()`. Um dos problemas dessa função é que os acessos à memória ocorrem em posições imprevisíveis.

Trechos como:

```cpp
int nidx = ny * w + nx;
```

fazem com que threads diferentes acessem regiões distantes da memória.

Isso prejudica a localidade espacial, o aproveitamento da cache, e a eficiência da GPU.

Nesse caso, precisamos reorganizar os dados, transformar as estruturas em formatos mais lineares e contínuos.

### Compactação de dados

Como a imagem binária possui muitos zeros:

* armazenar apenas pixels relevantes;
* utilizar estruturas compactas;
* aplicar conceitos similares a CSR (*Compressed Sparse Row*).

### Processamento por tiles/regiões

Dividir a imagem em regiões menores para melhorar localidade espacial.

---

## Observação Importante

Essa etapa é excelente para mostrar que nem todo algoritmo é naturalmente eficiente em GPU. O handout pode discutir:

* diferenças entre problemas regulares e irregulares;
* limitações do paralelismo massivo;
* impacto da organização dos dados.

Isso enriquece bastante o material didático.

---

# 3. Divergência de Threads

O código possui vários pontos com condicionais que podem causar divergência.

Um exemplo simples aparece em:

```cpp
if (mag > 255) mag = 255;
```

e principalmente em:

```cpp
if (bin[idx] == 255 && visited[idx] == 0)
```

---

## Problema

Quando threads do mesmo warp seguem caminhos diferentes:

* parte das threads fica ociosa;
* a execução torna-se serializada;
* o paralelismo efetivo diminui.

Esse problema é muito comum em visão computacional.

---

## Otimizações Possíveis

### Reduzir condicionais

Substituir operações simples por versões matemáticas.

Exemplo:

```cpp
mag = min(mag, 255);
```

---

### Separar kernels

Executar diferentes comportamentos em kernels distintos.

---

### Reorganizar dados

Agrupar regiões semelhantes para que threads vizinhas executem o mesmo fluxo.

---

## Resultado Esperado

As melhorias reduzem:

* serialização interna dos warps;
* desperdício de ciclos;
* ociosidade das threads.

---

# 4. Conclusão

Esse código é extremamente adequado para um handout didático porque permite mostrar:

| Problema           | Melhor Exemplo       |
| ------------------ | -------------------- |
| Data Reuse         | Sobel                |
| Shared Memory      | Sobel                |
| Tiling             | Sobel                |
| Divergência        | Threshold / BFS      |
| Acesso Irregular   | BFS                  |
| Sparse Data        | Imagem binária       |
| Gargalo de Memória | Sobel                |
| Limitações da GPU  | Connected Components |

Além disso, o fato de processar aproximadamente 3000 imagens torna os ganhos de desempenho perceptíveis, o que é ótimo para análises experimentais e comparação de benchmarks.
