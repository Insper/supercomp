# **Projeto 1 - Mineração de Hashes CPU em ambiente de HPC**

## Grupos de no máximo 3 alunos, data de entrega 29/Setembro

[Acesse o repositório do projeto aqui](
https://classroom.github.com/a/2374xoSF)

Neste projeto, seu grupo deverá **diagnosticar e otimizar um algoritmo de mineração de criptomoedas implementado em C++**. O código inicial foi propositalmente escrito de forma ineficiente, apresentando gargalos péssimas práticas de uso de memória.

Espera-se que seu grupo seja capaz de identificar esses problemas, propor hipóteses de melhoria, aplicar técnicas de otimização e mensurar o impacto das mudanças no desempenho da aplicação. Ao final, seu grupo deverá elaborar um relatório técnico com perfil, documentando todo o processo de análise e otimização.

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

1. **Código-fonte funcional**

   * Comentado e organizado
   * Inclui versão base e versão otimizada
   * Deve ser entregue via GitHub Classroom 

2. **Relatório Técnico**

   * Diagnóstico dos gargalos do código base
   * Proposta de otimização e hipótese de melhoria
   * Implementação da hipótese
   * Comparação de desempenho (tempo, speedup, eficiência, etc.)
   * Discussão dos resultados e limitações encontradas

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

