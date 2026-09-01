
# Guia rápido — SLURM

## Comandos gerais

| O que quero fazer | Comando |
|---|---|
| Ver partições e nós | `sinfo` |
| Ver meus jobs | `squeue -u $USER` |
| Ver detalhes de um job | `scontrol show job <JOB_ID>` |
| Ver detalhes de um nó | `scontrol show node <NODE>` |
| Ver detalhes de uma partição | `scontrol show partition <PARTITION>` |
| Cancelar um job | `scancel <JOB_ID>` |
| Cancelar todos os meus jobs | `scancel -u $USER` |

---

# Cluster Franky

## Partições

| Recurso | Partição |
|---|---|
| CPU | `merry_cpu` / `sunny_cpu`  / `pluton_cpu` |
| GPU | `merry_gpu` / `sunny_gpu`  / `pluton_gpu` |

## Execução interativa

### Abrir um terminal em um nó

```bash
srun --nodelist=compute10 \
     --partition=merry_cpu \
     --mem=1G \
     --pty bash
````

## Executar programa sequencial

### CPU

```bash
srun --partition=pluton_cpu \
     --mem=1G \
     ./meu_binario
```

### CPU de um nó da fila de GPU

```bash
srun --partition=pluton_gpu \
     --mem=1G \
     ./meu_binario
```

## Executar programa OpenMP

```bash
export OMP_NUM_THREADS=4

srun --partition=merry_cpu \
     --mem=1G \
     --cpus-per-task=4 \
     ./meu_programa_paralelo
```

## Submeter job com `sbatch`

### Programa sequencial

Arquivo `run.slurm`:

```bash
#!/bin/bash

#SBATCH --job-name=teste
#SBATCH --partition=merry_cpu
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=saida_%j.txt

./meu_programa
```

Submeter:

```bash
sbatch run.slurm
```

## Submeter programa OpenMP

```bash
#!/bin/bash

#SBATCH --job-name=openmp
#SBATCH --partition=pluton_cpu
#SBATCH --mem=1G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=saida_%j.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./meu_programa_paralelo
```

---

# Cluster Santos Dumont

## Diretório de trabalho

!!! warning "Importante"
Trabalhe na sua pasta `SCRATCH`.

```bash
cd /scratch/insperhpc/$USER
```

## Ver recursos disponíveis para seu usuário

```bash
sacctmgr list user $USER -s \
format=partition%20,MaxJobs,MaxSubmit,MaxNodes,MaxCPUs,MaxWall
```

## Partições utilizadas

| Recurso | Partição          |
| ------- | ----------------- |
| CPU     | `sequana_cpu_dev` |
| GPU     | `sequana_gpu_dev` |

## Ver nós

### CPU

```bash
sinfo -p sequana_cpu_dev
```

### GPU

```bash
sinfo -p sequana_gpu_dev
```

## Ver jobs

### Meus jobs

```bash
squeue -u $USER
```

### Jobs do projeto Insper

```bash
squeue -A insperhpc
```

## Executar programa sequencial

### CPU

```bash
srun --partition=sequana_cpu_dev \
     --mem=1G \
     ./meu_binario
```

### GPU

No Santos Dumont, ao utilizar a partição de GPU, solicite também uma GPU:

```bash
srun --partition=sequana_gpu_dev \
     --gres=gpu:1 \
     --mem=1G \
     ./meu_binario
```

## Executar programa OpenMP

```bash
export OMP_NUM_THREADS=4

srun --partition=sequana_cpu_dev \
     --mem=1G \
     --cpus-per-task=4 \
     ./meu_programa_paralelo
```

## Submeter programa OpenMP com `sbatch`

Arquivo `run.slurm`:

```bash
#!/bin/bash

#SBATCH --job-name=openmp
#SBATCH --output=saida_%j.txt
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=sequana_cpu_dev
#SBATCH --mem=1G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./meu_binario
```

Submeter:

```bash
sbatch run.slurm
```

## Submeter programa CUDA com `sbatch`

```bash
#!/bin/bash

#SBATCH --job-name=cuda
#SBATCH --output=saida_%j.txt
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=sequana_gpu_dev
#SBATCH --mem=1G

module load cuda/12.6_sequana

./meu_binario
```

Submeter:

```bash
sbatch run.slurm
```

---

# Consultando o hardware do nó

> Execute estes comandos **dentro do nó de computação alocado**.

## CPU

```bash
lscpu
```

Número de CPUs disponíveis:

```bash
nproc
```

## Memória

```bash
free -h
```

## Cache

```bash
lscpu | grep cache
```

## GPU

```bash
nvidia-smi
```

---

# Problemas com arquivos criados no Windows

Se um `.sh` ou `.slurm` criado no Windows apresentar problemas de fim de linha:

```bash
dos2unix arquivo.slurm
```

Ou com `vim`:

```bash
vim arquivo.slurm
```

Dentro do editor:

```vim
:set ff=unix
:wq
```


