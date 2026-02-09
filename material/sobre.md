

# Burocracias 

!!! info "Horários"
    **Aulas:**
    
    Segunda -> 16h30 -- 18h30
    
    Sexta -> 14h15 -- 16h15

    **Atendimento -> Início 27/02/26**
    
    Sexta -> 12:30 às 14:00


??? info "Objetivos de Aprendizagem"

    Ao final da disciplina, o estudante será capaz de:

    **Obj 1.** Desenvolver algoritmos usando recursos de computação paralela e distribuída para obter ganhos de desempenho na aplicação final.

    **Obj 2.** Aplicar estruturas lógicas de computação distribuída no desenvolvimento de algoritmos multitarefa.

    **Obj 3.** Usar GPGPU (General-Purpose computing on Graphics Processing Units) para computação numérica e comparar seu desempenho com soluções baseadas em CPU.

    **Obj 4.** Planejar e projetar sistemas de computação de alto desempenho, considerando aspectos de hardware, escalabilidade, e alocação de recursos.

    **Obj 5.** Analisar a complexidade de algoritmos paralelos e a eficiência de implementações específicas, identificando as métricas de desempenho mais adequadas para essa análise.

    **Obj 6.** Aplicar recursos específicos de sistemas operacionais (como escalonadores, controle de threads e gerenciamento de memória) para melhorar o desempenho de algoritmos.

    **Obj 7.** Desenvolver aplicações que utilizam protocolos otimizados para paralelização, como MPI, OpenMP e CUDA.


??? tip "Plano de Aulas - Supercomputação (2026.1)"
    | Data          | Aula | Tópicos Abordados                                                                                                                                 | Atividades                                                                                                   |
    |---------------|------|---------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
    | 09/fev (seg)  | 01   | Introdução à disciplina; conceitos de HPC; plataforma de HPC; acesso remoto (SSH); políticas de uso do Cluster Franky                             | Criação de conta no cluster; configuração do ambiente; primeiro código em C++; uso da IDE                   |
    | 13/fev (sex)  | 02   | Sistemas de HPC; arquitetura de clusters; rede e hardware; filas e jobs; SLURM; clusters no Brasil e no mundo                                     | Atividade prática com SLURM no Cluster Franky; submissão de jobs via SLURM                     |
    | 20/fev (sex)  | 03   | C++ para HPC: vetores; gerenciamento de memória; STL; alocação dinâmica; acesso cache-friendly                                                    | Exercícios de fixação em C++; análise de desempenho                                                          |
    | 23/fev (seg)  | 04   | C++ para HPC: matrizes; layout de memória (row-major); acesso sequencial vs aleatório; custo computacional                                         | Exercícios de fixação; comparação de padrões de acesso à memória                                             |
    | 27/fev (sex)  | 05   | Otimizações em CPU; hierarquia de memória; caches; prefetching; localidade temporal e espacial; loop unrolling                                    | Exercícios de otimização em CPU; medição de desempenho                                                       |
    | 02/mar (seg)  | 06   | Estratégias e algoritmos de otimização; BLAS; Roofline Model; limites computacionais vs memória                                                    | Exercícios de fixação; análise comparativa de desempenho e relatório técnico                                 |
    | 06/mar (sex)  | 07   | Algoritmos avançados de otimização; algoritmo de Strassen; matrizes esparsas; estruturas CSR e CSC                                                | Exercícios de fixação; análise de custo computacional                                                         |
    | 09/mar (seg)  | 08   | Concorrência e paralelismo; modelo fork-join; speedup; eficiência; Leis de Amdahl e Gustafson                                                     | Exercícios de paralelização; análise de escalabilidade                                                       |
    | 13/mar (sex)  | 09   | Otimizações de memória em estratégias paralelas; false sharing; alinhamento; afinidade de memória                                                  | Exercícios de fixação; comparação de desempenho antes e depois da otimização                                 |
    | 16/mar (seg)  | 10   | Gerenciamento de threads; thread pools; sincronização; barreiras; mutexes                                                                         | Exercícios de fixação; implementação correta sem condições de corrida                                        |
    | 20/mar (sex)  | 11   | Aula Estúdio: resolução orientada de problemas reais em HPC                                                                                       | Desenvolvimento incremental da APS 1; acompanhamento em sala                                                |
    | 23/mar (seg)  | 12   | Aula Estúdio: continuação da APS 1                                                                                                                 | Desenvolvimento incremental da APS 1                                                                        |
    | 27/mar (sex)  | 13   | Avaliação Intermediária                                                                                                                             | Avaliação prática e teórica                                                                                 |
    | 30/mar (seg)  | 14   | Avaliação Intermediária                                                                                                                             | Avaliação prática e teórica                                                                                 |
    | 06/abr (seg)  | 15   | Paralelismo com OpenMP; compartilhamento entre threads; efeitos do scheduler; escalonamento estático e dinâmico                                   | Exercícios de fixação com OpenMP                                                                             |
    | 10/abr (sex)  | 16   | Efeitos colaterais do paralelismo; race conditions; dependências de dados; recursão                                                               | Exercícios de fixação; correção de erros concorrentes                                                        |
    | 13/abr (seg)  | 17   | Memória distribuída; MPI ponto a ponto; Send/Recv; latência e topologias de comunicação                                                           | Exercícios de fixação; implementação MPI no cluster                                                          |
    | 17/abr (sex)  | 18   | MPI comunicação coletiva; Broadcast; Scatter; Gather; Reduce                                                                                       | Exercícios de fixação; análise de eficiência de comunicação                                                  |
    | 24/abr (sex)  | 19   | Grupos e comunicadores; programação híbrida MPI + OpenMP                                                                                           | Exercícios de fixação; implementação híbrida                                                                 |
    | 27/abr (seg)  | 20   | Introdução à programação paralela em GPU; arquitetura de GPUs; modelo SIMT; hierarquia de memória                                                 | Exercícios de fixação; primeiro código CUDA                                                                 |
    | 04/mai (seg)  | 21   | Programação em CUDA; kernels; grids e blocks; memória global e compartilhada                                                                       | Exercícios de fixação; implementação de kernels                                                              |
    | 08/mai (sex)  | 22   | CUDA avançado: stencil, tiling e agendamento de threads                                                                                            | Exercícios de fixação; comparação entre versões otimizadas                                                   |
    | 11/mai (seg)  | 23   | Redução e scan em CUDA; algoritmos paralelos clássicos; uso de memória compartilhada                                                              | Exercícios de fixação; medição de speedup                                                                    |
    | 15/mai (sex)  | 24   | Matrizes esparsas em GPU; SpMV; computação assíncrona; streams; sobreposição comunicação–cálculo                                                  | Exercícios de fixação; análise de desempenho                                                                 |
    | 18/mai (seg)  | 25   | Código síncrono vs assíncrono em CUDA; streams; eventos; latência e throughput                                                                    | Exercícios de fixação; comparação quantitativa entre abordagens                                              |
    | 22/mai (sex)  | 26   | Revisão geral do conteúdo; exercícios preparatórios para Avaliação Final                                                                          | Lista de exercícios preparatórios                                                                            |
    | 25/mai (seg)  | 27   | Simulado da Avaliação Final                                                                                                                         | Resolução comentada do simulado                                                                              |
    | 29/mai (sex)  | 28   | Avaliação Final                                                                                                                                     | Avaliação Final                                                                                              |
    | 01/jun (seg)  | 29   | Avaliação Final                                                                                                                                     | Avaliação Final                                                                                              |


??? note "Atividades (Individual) 15%"
    | Percentual de Atividades | Conceito |
    |--------------------------|----------|
    | 50%                      | C        |
    | 70%                      | B        |
    | 90%                      | A        |
    | 100%                     | A +      |




??? note "APS 1  10%"
    Em construção
    

??? note "APS 2 20%"
    Em construção
    