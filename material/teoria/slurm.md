# SLURM

### O que é SLURM?

SLURM (Simple Linux Utility for Resource Management) é um gerenciador de workload open-source amplamente utilizado em clusters de computação de alto desempenho (HPC). Ele é responsável por alocar recursos de computação (como CPUs, memória e GPUs) aos usuários e suas tarefas, gerenciar filas de jobs, monitorar o uso de recursos e agendar a execução de tarefas de forma eficiente.

### Principais Funcionalidades do SLURM

1. Alocação de Recursos: SLURM distribui recursos de computação, como nós e processadores, conforme solicitado pelos usuários. Ele assegura que os recursos são utilizados de maneira eficiente e equitativa.
2. Submissão de Jobs: Usuários podem submeter jobs (tarefas de computação) ao SLURM, que coloca esses jobs em uma fila e os executa quando os recursos necessários estão disponíveis.
3. Monitoramento e Gerenciamento: SLURM monitoriza o estado dos jobs, nós e partições do cluster, fornecendo ferramentas para verificar o status e a utilização de recursos.
4. Políticas de Priorização: Implementa políticas para priorizar jobs com base em vários fatores, como tempo de espera, utilização de recursos e prioridades definidas pelo administrador do cluster.

### Comandos Principais do SLURM

Alguns dos comandos principais usados no SLURM:

1. sbatch: É um comando do SLURM usado para submeter scripts de jobs para execução em um cluster. Esses scripts contêm instruções sobre como os recursos devem ser alocados e quais comandos devem ser executados
2. scancel: Cancela um job pendente ou em execução.
3. scontrol: Ferramenta administrativa usada para visualizar e/ou modificar o estado do SLURM. Muitos comandos `scontrol` só podem ser executados pelo adminstrador do sistema.
4. sinfo: Relata o estado das filas e nós gerenciados pelo SLURM, com várias opções de filtragem, ordenação e formatação.
5. sprio: Exibe uma visão detalhada dos componentes que afetam a prioridade de um job.
6. squeue: Relata o estado dos jobs em execução em ordem de prioridade e depois os jobs pendentes em ordem de prioridade.
7. srun: Submete um job para execução e faz o pedido de alocação dos recursos da maquina.
8. strigger: Define, obtém ou visualiza gatilhos de eventos, como nós caindo ou jobs se aproximando do limite de tempo.

### Exemplos de Uso para Cada Comando SLURM

### 1. sbatch

Submete scripts de jobs para execução em um cluster. Esses scripts contêm instruções sobre como os recursos devem ser alocados e quais comandos devem ser executados.

Exemplo de Script:

```bash
#!/bin/bash
#SBATCH --job-name=meu_job        # Nome do job
#SBATCH --output=meu_job.out      # Arquivo de saída
#SBATCH --error=meu_job.err       # Arquivo de erro
#SBATCH --ntasks=1                # Número de tarefas
#SBATCH --cpus-per-task=4         # Número de CPUs por tarefa
#SBATCH --mem=4G                  # Memória total alocada para o job
#SBATCH --time=00:02:00           # Tempo máximo de execução (hh:mm:ss)
#SBATCH --partition=normal        # Fila do cluster

echo "Iniciando o job"
sleep 60
echo "Job finalizado"

```

Explicação das Flags:

`-job-name=meu_job`: Define o nome do job como `meu_job`.

`-output=meu_job.out`: Especifica o arquivo onde a saída padrão do job será registrada.

`-error=meu_job.err`: Especifica o arquivo onde os erros serão registrados.

`-ntasks=1`: Define o número de tarefas (processos) a serem utilizados pelo job.

`-cpus-per-task=4`: Aloca 4 CPUs para cada tarefa.

`-mem=4G`: Especifica que 4 GB de memória serão alocados para o job.

`-time=00:02:00`: Define o tempo máximo de execução do job como 2 minutos.

`-partition=normal`: Especifica a fila do cluster onde o job será executado.

### 2. scancel

Cancela um job pendente ou em execução.

Exemplo:

```bash
scancel 12345

```

Explicação das Flags:

- `12345`: Especifica o ID do job que deve ser cancelado.

### 3. scontrol

Ferramenta administrativa usada para visualizar e/ou modificar o estado do SLURM. Muitos comandos `scontrol` só podem ser executados pelo administrador do sistema.

Exemplo:

```bash
scontrol show job 12345

```

Explicação das Flags:

`show job 12345`: Exibe informações detalhadas sobre o job com ID `12345`.

### 4. sinfo

Relata o estado das filas e nós gerenciados pelo SLURM.

Exemplo:

```bash
sinfo

```

### 5. sprio

Exibe uma visão detalhada dos componentes que afetam a prioridade de um job.

Exemplo:

```bash
sprio

```

### 6. squeue

Relata o estado dos jobs em execução em ordem de prioridade e depois os jobs pendentes em ordem de prioridade.

Exemplo:

```bash
squeue -u username

```

Explicação das Flags:

`u username`: Filtra a saída para mostrar apenas os jobs do usuário `username`.

### 7. srun

Submete um job para execução e faz o pedido de alocação dos recursos da máquina.

Exemplo:

```bash
srun -N 1 -n 1 --time=00:01:00 ./meu_programa

```

Explicação das Flags:

`N 1`: Especifica que 1 nó deve ser alocado.

`n 1`: Define que 1 tarefa deve ser executadas.

`-time=00:01:00`: Define o tempo máximo de execução do job como 1 minuto.

`./meu_programa`: Especifica o programa a ser executado.


### 9. strigger

 Define, obtém ou visualiza gatilhos de eventos, como nós caindo ou jobs se aproximando do limite de tempo.

Exemplo:
Vamos supor que você queira configurar um gatilho (`strigger`) para monitorar o uso de memória de um job específico e enviar um alerta quando o uso de memória ultrapassar um certo limite. Aqui está um exemplo completo:

```bash
strigger --set --jobid=<JOB_ID> --threshold=500 --action="echo 'Atenção: O uso de memória ultrapassou 500 MB!'"
```
### Explicação do Comando:

`strigger`: Comando para configurar um gatilho no SLURM.

`--set`: Indica que estamos criando um novo gatilho.

`--jobid=<JOB_ID>`: Especifica o ID do job que queremos monitorar. Substitua `<JOB_ID>` pelo ID real do job.

`--threshold=500`: Define o limiar de 500 MB de uso de memória. Quando o job usar mais de 500 MB, o gatilho será ativado.

`--action="echo 'Atenção: O uso de memória ultrapassou 500 MB!'"`: Define a ação que será executada quando o gatilho for ativado. Neste caso, a ação é um simples comando `echo` que imprime uma mensagem de alerta.

Este gatilho será acionado quando o job especificado pelo `<JOB_ID>` ultrapassar 500 MB de uso de memória. A ação definida (`echo`) será executada, e você verá a mensagem "Atenção: O uso de memória ultrapassou 500 MB!" no terminal ou no arquivo de saída do job.

Você pode personalizar o comando `strigger` para outras situações, como monitorar o tempo restante de um job, detectar falhas de nós, ou monitorar o uso de CPU, simplesmente ajustando os parâmetros e as ações conforme necessário.

### Tipos Comuns de Eventos Monitoráveis pelo `strigger`:

Tempo Restante (`TIME_LIMIT`):

   Aciona o gatilho quando um job se aproxima de seu limite de tempo de execução.

   Uso do `--threshold`: Especifica o tempo restante em segundos.

   Exemplo: `strigger --set --jobid=<JOB_ID> --threshold=60 --action="echo 'Job está a 60 segundos do limite de tempo!'"`
  
Uso de Memória (`MEMORY`):

   Aciona o gatilho quando o uso de memória de um job ultrapassa um certo limite.

   Uso do `--threshold`: Especifica a quantidade de memória usada, geralmente em megabytes (MB).

   Exemplo: `strigger --set --jobid=<JOB_ID> --threshold=1024 --action="echo 'Job ultrapassou 1GB de memória!'"`
  
Falha de Nó (`NODE_FAIL`):

   Aciona o gatilho quando um nó falha ou fica indisponível.

   Uso do `--threshold`: Não aplicável diretamente. O gatilho é acionado quando o evento de falha ocorre.

   Exemplo: `strigger --set --node=<NODE_NAME> --event=NODE_FAIL --action="echo 'Nó falhou!'"`
  
Falha de Job (`JOB_FAIL`):

   Aciona o gatilho quando um job falha por qualquer motivo.

   Uso do `--threshold`: Não aplicável diretamente. O gatilho é acionado quando o evento de falha ocorre.

   Exemplo: `strigger --set --jobid=<JOB_ID> --event=JOB_FAIL --action="echo 'Job falhou!'"`
  
Início de Job (`JOB_START`):

   Aciona o gatilho quando um job começa a ser executado.

   Uso do `--threshold`: Não aplicável diretamente. O gatilho é acionado quando o evento de início ocorre.

   Exemplo: `strigger --set --jobid=<JOB_ID> --event=JOB_START --action="echo 'Job iniciou!'"`
  
Finalização de Job (`JOB_END`):

   Aciona o gatilho quando um job termina sua execução, independentemente de ter sido concluído com sucesso ou não.

   Uso do `--threshold`: Não aplicável diretamente. O gatilho é acionado quando o evento de finalização ocorre.

   Exemplo: `strigger --set --jobid=<JOB_ID> --event=JOB_END --action="echo 'Job terminou!'"`
  
Limite de CPU (`CPU_LIMIT`):

   Aciona o gatilho quando o uso de CPU de um job ultrapassa um certo limite.

   Uso do `--threshold`: Especifica o uso de CPU em segundos de CPU ou em porcentagem.

   Exemplo: `strigger --set --jobid=<JOB_ID> --threshold=80 --action="echo 'Job ultrapassou 80% do uso de CPU!'"`
  
Submissão de Job (`JOB_SUBMIT`):

   Aciona o gatilho quando um job é submetido para execução.

   Uso do `--threshold`: Não aplicável diretamente. O gatilho é acionado quando o evento de submissão ocorre.

   Exemplo: `strigger --set --user=<USER_NAME> --event=JOB_SUBMIT --action="echo 'Um job foi submetido!'"`
  
Queda de Partição (`PARTITION_DOWN`):

   Aciona o gatilho quando uma partição inteira do cluster cai ou fica indisponível.

   Uso do `--threshold`: Não aplicável diretamente. O gatilho é acionado quando o evento de queda da partição ocorre.

   Exemplo: `strigger --set --partition=<PARTITION_NAME> --event=PARTITION_DOWN --action="echo 'Partição caiu!'"`
  
Chegada de Job ao Limiar de Tempo (`TIME_LIM_REACHED`):

   Similar ao `TIME_LIMIT`, mas pode ser mais específico para quando o tempo limite é alcançado, não apenas quando está próximo.

   Uso do `--threshold`: Especifica o tempo restante ou o evento do tempo limite.

   Exemplo: `strigger --set --jobid=<JOB_ID> --event=TIME_LIM_REACHED --action="echo 'Tempo limite alcançado!'"`


