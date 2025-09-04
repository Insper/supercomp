

## Comandos SLURM

### Principais recursos que você pode pedir com `srun`

| O que pedir                   | Opção do `srun`                | Exemplo                          |
| ----------------------------- | ------------------------------ | -------------------------------- |
| Número de tarefas (processos) | `--ntasks` ou `-n`             | `--ntasks=2`                     |
| CPUs por tarefa               | `--cpus-per-task`              | `--cpus-per-task=2`              |
| Memória total ou por CPU      | `--mem`, `--mem-per-cpu`       | `--mem=4G` ou `--mem-per-cpu=2G` |
| Tempo de execução             | `--time=DD-HH:MM:SS`            | `--time=01:12:49`               |
| Número de nós                 | `--nodes`                      | `--nodes=2`                      |
| Nó específico                 | `--nodelist`                   | `--nodelist=compute13`           |
| GPUs                          | `--gpus` ou `--gres=gpu:<num>` | `--gpus=1` ou `--gres=gpu:2`     |
| Partição (fila)               | `--partition` ou `-p`          | `--partition=gpu`                |
| Sessão interativa             | `--pty bash`                   | `--pty bash`                     |

**Exemplos**

Pedido simples de execução de tarefa para o SLURM
```bash
srun --nodelist=compute10 --partition=normal --ntasks=1 --pty bash
```
--srun: Comando SLURM para executar tarefas 
--nodelist=compute10: força o SLURM a alocar exatamente esse nó.
--partition=normal: indica a partição à qual o nó pertence.
--ntasks=1: pede uma tarefa.
--pty bash: pede um terminal dentro do nó.


Pedido sem especificar o nó exato:
```bash
srun --partition=normal --ntasks=1 --pty bash
```

Pedido com varias tasks para executar seu programa direto:
```bash
srun --partition=normal --ntasks=4 ./meu_programa_paralelo
```

### Principais recursos que você pode pedir com `sbatch`

| O que pedir                   | Opção do `sbatch` (no script)  | Exemplo dentro do script                     |
| ----------------------------- | ------------------------------ | -------------------------------------------- |
| Nome do job                   | `--job-name`                   | `#SBATCH --job-name=teste%j`                   |
| Número de tarefas (processos) | `--ntasks` ou `-n`             | `#SBATCH --ntasks=2`                         |
| CPUs por tarefa               | `--cpus-per-task`              | `#SBATCH --cpus-per-task=2`                  |
| Memória total ou por CPU      | `--mem`, `--mem-per-cpu`       | `#SBATCH --mem=4G` ou `--mem-per-cpu=2G`     |
| Tempo de execução             | `--time=DD-HH:MM:SS`           | `#SBATCH --time=01:12:49`                    |
| Número de nós                 | `--nodes`                      | `#SBATCH --nodes=2`                          |
| Nó específico                 | `--nodelist`                   | `#SBATCH --nodelist=compute13`               |
| GPUs                          | `--gpus` ou `--gres=gpu:<num>` | `#SBATCH --gpus=1` ou `#SBATCH --gres=gpu:2` |
| Partição (fila)               | `--partition` ou `-p`          | `#SBATCH --partition=gpu`                    |
| Saída padrão (log)            | `--output`                     | `#SBATCH --output=saida%j.txt`                 |
| Log de Erro do sistema        | `--error`                      | `#SBATCH --error=erro%j.txt`                   |

**Exemplos**

**Arquivo:** `job1.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=teste%j
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=saida%j.txt

./meu_programa
```

Submeter com:

```bash
sbatch job1.slurm
```


## Comandos gerais do SLURM

| Finalidade                            | Comando                                      | Exemplo                                           |
| ------------------------------------- | -------------------------------------------- | ------------------------------------------------- |
| Ver status das partições e nós        | `sinfo`                                      | `sinfo -N -l`                                     |
| Ver detalhes de um nó específico      | `scontrol show node`                         | `scontrol show node compute24`                         |
| Ver detalhes de uma partição          | `scontrol show partition`                    | `scontrol show partition normal`                    |
| Ver todos os jobs ativos              | `squeue`                                     | `squeue`                                          |
| Ver seus próprios jobs                | `squeue -u <usuário>`                        | `squeue -u liciascl`                              |
| Ver detalhes de um job                | `scontrol show job`                          | `scontrol show job 12345`                         |
| Cancelar job em execução ou na fila   | `scancel`                                    | `scancel 12345`                                   |
| Cancelar todos os seus jobs           | `scancel -u <usuário>`                       | `scancel -u liciascl`                             |



### Ver todos os nós com status detalhado

```bash
sinfo -N -l
```
Útil para ver quais nós estão **idle, alocados, down ou drain**.

### Ver informações completas do nó `compute24`

```bash
scontrol show node compute24
```
Mostra: memória total e usada, CPUs alocadas, jobs em execução, estado (`IDLE`, `ALLOCATED`, etc.).

### Ver configurações de uma partição especifica

```bash
scontrol show partition normal
```
Mostra: tempo máximo de job, número de nós, limites de memória/CPU, GPUs, estado da fila.

### Ver jobs no sistema

```bash
squeue
```
Mostra todos os jobs na fila e em execução com status `R` (running), `PD` (pending), etc.

### Ver só os jobs da usuária `liciascl`

```bash
squeue -u liciascl
```
Útil para depurar seus próprios jobs (ID, partição, status, tempo, nó, etc.)


### Ver informações completas de um job específico

```bash
scontrol show job 12345
```
Mostra: usuário, partição, CPUs/nós alocados, prioridade, estado, tempo usado, comando enviado.


### Cancelar job com ID `12345`

```bash
scancel 12345
```
Útil se o job travou ou está consumindo recursos indevidamente.

### Cancelar **todos os seus jobs**

```bash
scancel -u $USER
```
Cancela em lote — ótimo em caso de erro em scripts ou submissões mal feitas.


Para mais consulte a documentação oficial em https://slurm.schedmd.com/documentation.html

## Pra verificar Hardware

##  **CPU**

* **lscpu**
  Mostra arquitetura, número de núcleos, threads, caches.

  ```bash
  lscpu
  ```

  Exemplo de saída:

  ```
  Architecture:           x86_64
  CPU(s):                 40
  Thread(s) per core:     2
  Core(s) per socket:     10
  Socket(s):              2
  L1d cache:              32K
  L2 cache:               1M
  L3 cache:               13M
  ```


Lista detalhes por CPU lógico (modelo, MHz, cache).

  ```bash
  cat /proc/cpuinfo 
  ```


Mostra o número de CPUs disponíveis.

  ```bash
  nproc
  ```

## **Memória RAM**

  Mostra uso e total de memória física e swap.

  ```bash
  free -h
  ```

Detalhes avançados de memória (MemTotal, MemFree, Buffers, Cached).

  ```bash
  cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|Swap"
  ```


Estatísticas de memória, processos e CPU.

  ```bash
  vmstat 1 5
  ```


## **Cache**

  Mostra rapidamente o tamanho das caches.

  ```bash
  lscpu | grep cache
  ```

Lista tamanhos de cada nível de cache (por CPU).

  ```bash
  cat /sys/devices/system/cpu/cpu0/cache/index*/size
  ```

Shell script para trazer de forma resumida informações úteis

```bash
echo '=== HOSTNAME ==='; hostname; echo; \
 echo '=== MEMORIA (GB) ==='; \
 cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable|Swap' | \
 awk '{printf \"%s %.2f GB\\n\", \$1, \$2 / 1048576}'; \
 echo; \
 echo '=== CPU INFO ==='; \
 lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\\(s\\)|cache'
 echo '=== GPU INFO ==='; \
 if command -v nvidia-smi &> /dev/null; then nvidia-smi; else echo 'nvidia-smi não disponível'; fi
```
Para executar dentro de um nó de computação:

```bash
srun --partition=normal --ntasks=1 --pty bash -c \
"echo '=== HOSTNAME ==='; hostname; echo; \
 echo '=== MEMORIA (GB) ==='; \
 cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable|Swap' | \
 awk '{printf \"%s %.2f GB\\n\", \$1, \$2 / 1048576}'; \
 echo; \
 echo '=== CPU INFO ==='; \
 lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\\(s\\)|cache'
 echo '=== GPU INFO ==='; \
 if command -v nvidia-smi &> /dev/null; then nvidia-smi; else echo 'nvidia-smi não disponível'; fi"
```

## Bug de arquivos gerados no Windows
Trabalhando em 2 ambientes diferentes é comum ocorrer problemas de fim de
linha entre DOS e UNIX. 

Caso você encontre esse problema aqui vão algumas opções. 


[Video ilustrando o passo a passo](https://www.youtube.com/embed/AiHhyOJ526k)



*Utilizando o editor NANO*

Abra o arquivo utilizando a flag unix:
```sh
nano --unix arquivo.ext
```

feito isso basta salvar com C^O e sair C^X

*Utilizando o editor VI/VIM*

```sh
vi arquivo.ext
```

Com ele aberto é possivel definir o `fileformat` com o seguinte comando:

```vi
:set ff=unix
```

e então sair e salvar com:

```vi
:wq!
```

o `!` irá forçar a sobrescrita. 

*Utilizando o VI/VIM por linha de comando:*

```sh
vi -c "set ff=unix" -c "wq" arquivo.ext
```

*Se dos2unix tiver instalado:*

```sh
dos2unix arquico.ext
```

