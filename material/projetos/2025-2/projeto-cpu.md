
# **Projeto 1 - Mineração de Hashes CPU em ambiente de HPC**
 
A mineração surgiu com o **Bitcoin**, criado por *Satoshi Nakamoto* em 2008. A proposta era descentralizar o controle do dinheiro, permitindo que qualquer pessoa pudesse participar da validação das transações em uma rede pública.

A validação é feita por meio de um **algoritmo de consenso** chamado **Proof of Work (PoW)**. A PoW exige que os nós da rede (os *mineradores*) resolvam um problema matemático difícil — e quem resolve primeiro, tem o direito de adicionar um novo bloco à blockchain e receber uma recompensa.

Essa “prova” é feita através de um processo chamado de **hashing**, e no caso do Bitcoin, utiliza o algoritmo **SHA-256**.

##  **O que é SHA-256**

**SHA-256** é parte da família de funções de hash **SHA-2**, desenvolvida pela Agência de Segurança Nacional dos Estados Unidos (NSA) em 2001.  
SHA significa *Secure Hash Algorithm*.

### Características principais:
- Gera uma saída de **256 bits** (64 caracteres hexadecimais)
- É uma função **determinística**: mesma entrada, mesma saída
- É **unidirecional**: não dá para "voltar" da saída para a entrada
- Pequenas mudanças na entrada resultam em mudanças drásticas na saída (efeito avalanche)
- Altamente sensível à **colisões** (duas entradas diferentes que dão a mesma saída são indesejáveis)


##  **Como o SHA-256 funciona?**

SHA-256 funciona em blocos de **512 bits de entrada**, que passam por várias etapas:

1. **Pré-processamento**:
   - *Padding*: a mensagem é estendida até múltiplos de 512 bits.
   - *Parsing*: a mensagem é dividida em blocos de 512 bits.
   - *Inicialização*: 8 variáveis de 32 bits com constantes iniciais (derivadas da raiz quadrada dos primeiros 8 primos).

2. **Função de compressão**:
   - Cada bloco de 512 bits passa por **64 rodadas** de operações bit a bit (AND, OR, XOR, ROTR, etc).
   - Usa uma tabela de **64 constantes** (derivadas da raiz cúbica dos primeiros 64 primos).
   - A cada rodada, as variáveis são atualizadas com funções não lineares.

3. **Concatenação final**:
   - Após todos os blocos processados, os 8 valores de 32 bits são concatenados e formam o **hash final de 256 bits**.


##  **Mineração com SHA-256: o que acontece?**

O minerador tenta encontrar um valor chamado **nonce** que, quando combinado com o cabeçalho do bloco e passado pelo SHA-256 duas vezes (**double SHA-256**), resulta em um hash menor que o alvo definido pela **dificuldade da rede**.

Ou seja, o minerador está basicamente **tentando encontrar um número (nonce)** que leve o hash a começar com **N zeros**.

## **Por que é um problema de HPC?**

A mineração é um problema de **busca exaustiva**. Os mineradores testam **bilhões de nonces por segundo**. A performance da mineração depende da capacidade de realizar **SHA-256 o mais rápido possível**.

Por trás da idéia de "minerar bitcoins", existe um problema computacional intensivo, cuja solução depende de capacidade de processamento, eficiência do código e uso inteligente dos recursos de hardware.

Então, o que a mineração tem a ver com HPC?

Simples: a mineração de criptomoedas é um problema clássico de HPC moderno, por envolver:

    Busca exaustiva de soluções (nonces)

    Processamento paralelo de dados

    Uso de algoritmos de hash otimizados (SHA-256)

    Aproveitamento de CPU, GPU e clusters

para entender com mais detalhes como SHA-256 se relaciona com um sistema de criptomoedas, [assista o vídeo](https://www.youtube.com/watch?v=bBC-nXj3Ng4). Vamos ao projeto...

# **Projeto 1 - Mineração de Hashes em ambiente de HPC**

## **Grupos de no máximo 3 alunos, data de entrega 29/Setembro**

[Acesse o repositório do projeto aqui](
https://classroom.github.com/a/2374xoSF)

Neste projeto, seu grupo deverá **diagnosticar e otimizar um algoritmo de mineração de criptomoedas implementado em C++**. O código inicial foi propositalmente escrito de forma ineficiente, apresentando gargalos péssimas práticas de uso de memória.

Espera-se que seu grupo seja capaz de identificar esses problemas, propor hipóteses de melhoria, aplicar técnicas de otimização e mensurar o impacto das mudanças no desempenho da aplicação. Ao final, seu grupo deverá elaborar um relatório técnico, documentando todo o processo de análise e otimização.

A dificuldade da mineração é ajustada pela quantidade de zeros exigida no início do hash. À medida que vocês aumentam essa dificuldade, o desafio computacional cresce, o que demanda boas decisões de otimização e uso eficiente de CPU e memória. Analise adequadamente os recursos disponíveis das filas do Cluster Franky para realizar os seus testes e suas otimizações.

---

### **Objetivo**

Aplicar conhecimentos de:

* Análise de desempenho
* Diagnóstico de gargalos
* Boas práticas de gerenciamento de memória
* Paralelismo com OpenMP
* Distribuição com MPI

## **Rúbrica de Avaliação**

| Conceito | Critérios Técnicos                                                                                                                                                                                                                                                    |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C**    | Executa o minerador sequencial no cluster Franky com dificuldade **6 zeros**, Realiza Passagem de objetos grandes por referência ou ponteiro; Minimização de cópias desnecessárias; Uso eficiente de buffers;  Implementa uma heurísitca eficiente.|
| **B**    | Executa o minerador com dificuldade **7 zeros**, Realiza as otimizações da rúbrica C e aplica **paralelização com OpenMP ou distribuição com MPI** |
| **A**    | Executa o minerador com dificuldade **8 zeros**, Realiza as otimizações da rúbrica C e aplica **paralelização com OpenMP E distribuição com MPI** |

---

## **Entrega**

A entrega deve conter:

0. O nome de cada integrante do grupo

1. Código-fonte funcional

2. Diagnóstico dos gargalos do código base

3. Proposta de otimização e hipótese de melhoria

4. Implementação da hipótese

5. Comparação de desempenho (tempo, speedup, eficiência, etc.)

6. Discussão dos resultados e limitações encontradas

---

## **Bônus por Qualidade do relatório técnico**

| Conceito Base | Com Bônus |
| ------------- | --------- |
| C             | C+        |
| B             | B+        |
| A             | A+        |


### Observações

* Espera-se que o aluno entenda os problemas do código base antes de corrigir.
* O binário `transacoes` simula um ambiente de rede assíncrono com envio de blocos não simultâneos, para que o código minerador seja tolerante a ordem e tempo de chegada.
* A execução dos testes pode demorar minutos. Se planeje bem, escolha adequadamente a fila que será utilizada nos seus testes.

