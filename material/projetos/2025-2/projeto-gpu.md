
# **Projeto 2 - Mineração de Hashes GPU em ambiente de HPC**
 
## **Individual, data de entrega 14/Novembro**

[Acesse o repositório do projeto aqui](
https://classroom.github.com/a/QGyxrPlE)

Neste projeto, você deverá **diagnosticar e otimizar um algoritmo de mineração de criptomoedas implementado em C++**. 

Espera-se que você seja capaz de identificar os gargalos do código, aplicar técnicas de otimização e mensurar o impacto das mudanças no desempenho da aplicação. Ao final, você deverá elaborar um relatório, documentando todo o processo de análise e otimização.

A dificuldade da mineração é ajustada pela quantidade de zeros exigida no início do hash. À medida que vocês aumentam essa dificuldade, o desafio computacional cresce, o que demanda boas decisões de otimização e uso eficiente de CPU, GPU e memória. Analise adequadamente os recursos disponíveis no sistema de HPC utilizado para realizar os seus testes e suas otimizações.

---

## **Entrega**

A entrega deve incluir um **relatório técnico** descrevendo de forma clara:

* as **estratégias de otimização aplicadas** no código, justificando as escolhas de cada abordagem;
* as **métricas de desempenho** utilizadas (tempo de execução, throughput, etc.);
* os **ganhos de desempenho obtidos**, com evidências experimentais (gráficos, tabelas).

O relatório deve refletir o raciocínio crítico sobre o impacto das otimizações no uso da GPU e a eficiência paralela alcançada.


| **Conceito** | **Critérios**                                                                                                                                                                                                                                                                                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C**        | Executa o **minerador com paralelismo em GPU**, demonstrando domínio de boas práticas como minimização de cópias entre CPU e GPU, uso eficiente de buffers, redução de acessos à memória global e implementação de uma heurística eficiente. O código deve atingir **dificuldade 6 zeros** e realizar a mineração em até **20 minutos** no **Cluster Franky**. |
| **B**        | Executa o minerador com **dificuldade 7 zeros** aplicando **todas as otimizações da rúbrica C**. O experimento deve ser executado no **supercomputador Santos Dumont**, completando a mineração de todos os blocos em no máximo **20 minutos**.                                                                                        |
| **A**        | Executa o minerador com **dificuldade 7 zeros**, aplicando **todas as otimizações da rúbrica C**. O experimento deve ser executado no **supercomputador Santos Dumont**, completando a mineração de todos os blocos em no máximo **10 minutos**.                                                                                        |
| **A+**        | Executa o minerador com **dificuldade 8 zeros**, aplicando **todas as otimizações da rúbrica C**. O experimento deve ser executado no **supercomputador Santos Dumont**, completando a mineração de todos os blocos em no máximo **1 hora e 35 minutos**.     