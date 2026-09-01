# Cluster Santos Dumont

Após concluir a configuração da VPN, ative a VPN para ser possível acessar o Santos Dumont via SSH.

Em um terminal novo utilize o comando;

```bash
ssh -o MACs=hmac-sha2-256 seu-login-de-acesso@login.sdumont.lncc.br
```
ou 

```bash
ssh seu-login-de-acesso@login.sdumont.lncc.br
```

Note que sua "home" é a pasta de projeto, chamada "/prj/insperhpc/seu-login", não submeta jobs desta pasta. Para executar e testar seus códigos, utilize a pasta `SCRATCH`, que é o ambiente apropriado para processamento e armazenamento temporário de arquivos.

```bash
cd /scratch/insperhpc/seu-login
```

### Ambientação no Santos Dumont

Antes de começar a fazer pedidos de recursos pro SLURM, vamos conhecer as filas que temos acesso e o hardware que temos disponível em cada fila.

```bash
sacctmgr list user $USER -s format=partition%20,MaxJobs,MaxSubmit,MaxNodes,MaxCPUs,MaxWall
```


As filas que temos acesso tem essas características:


## Fila sequana_cpu_dev

A fila sequana_cpu_dev pode ser usada para testes em CPU, permitindo a execução de apenas 1 job por vez,  utilizando até 4 nós e 192 CPUs por job, com tempo limite de 20 minutos. A infraestrutura disponível conta com 166 nós e 7968 CPUs no total, com 8 GB de memória por CPU, somando aproximadamente 62 TB de memória. 

## Fila sequana_gpu_dev

A fila sequana_gpu_dev pode ser usada para testes com uso de GPU, também limitada a 1 job por vez, podendo utilizar até 4 nós e 192 CPUs por job, com tempo máximo de 20 minutos. Disponibiliza 61 nós, 2928 CPUs e 244 GPUs, com configuração padrão de 12 núcleos de CPUs por GPU e 94 GB de memória por GPU. A memória por CPU é de 8 GB. 


### Explorando com o SRUN
Vale lembrar que podemos pedir via SRUN um terminal dentro do nó de computação, para, de forma livre, executar qualquer comando:

```bash
srun --partition=sequana_cpu_dev --pty bash
```

Para sair, basta digitar no terminal:
```bash
exit
```

Ou, podemos usar o SRUN com um comando definido que será executado no nó de computação de forma direta pelo terminal:


Para visualizar o hardware disponível na fila CPU do SDumont:
```bash
srun --partition=sequana_cpu_dev --ntasks=1 --pty bash -c \
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
Você vai ver algo como:

```bash
srun: job 11588872 queued and waiting for resources
srun: job 11588872 has been allocated resources
=== HOSTNAME ===
sdumont6048

=== MEMORIA (GB) ===
MemTotal: 376.18 GB
MemFree: 287.27 GB
MemAvailable: 319.21 GB
SwapCached: 0.00 GB
SwapTotal: 0.00 GB
SwapFree: 0.00 GB

=== CPU INFO ===
CPU(s):              48
On-line CPU(s) list: 0-47
Thread(s) per core:  1
Core(s) per socket:  24
Socket(s):           2
Model name:          Intel(R) Xeon(R) Gold 6252 CPU @ 2.10GHz
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            36608K
NUMA node0 CPU(s):   0-23
NUMA node1 CPU(s):   24-47
=== GPU INFO ===
nvidia-smi não disponível
```

**Você sempre pode usar o HW de CPU da fila GPU**

Para visualizar o hardware disponível na fila GPU do SDumont:
```bash
srun --partition=sequana_gpu_dev --gres=gpu:1 --ntasks=1 --pty bash -c \
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

Vai aparecer algo como:

```bash
srun: job 11588877 queued and waiting for resources
srun: job 11588877 has been allocated resources
=== HOSTNAME ===
sdumont8046

=== MEMORIA (GB) ===
MemTotal: 376.18 GB
MemFree: 369.89 GB
MemAvailable: 370.89 GB
SwapCached: 0.00 GB
SwapTotal: 0.00 GB
SwapFree: 0.00 GB

=== CPU INFO ===
CPU(s):              48
On-line CPU(s) list: 0-47
Thread(s) per core:  1
Core(s) per socket:  24
Socket(s):           2
Model name:          Intel(R) Xeon(R) Gold 6252 CPU @ 2.10GHz
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            36608K
NUMA node0 CPU(s):   0-23
NUMA node1 CPU(s):   24-47
=== GPU INFO ===
Tue Sep  1 10:13:27 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-SXM2-32GB           On  |   00000000:60:00.0 Off |                    0 |
| N/A   46C    P0             44W /  300W |       1MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```


O comando `sinfo` mostra quais são as filas e quais são os status dos nós 

```bash
sinfo
```
Como o Santos Dumont é utilizado por pessoas de todo o país, a quantidade de informações exibidas pode ser muito grande. Por isso, vamos aplicar alguns filtros para visualizar apenas o que é relevante para nós:


Este comando filtra por projeto, então só veremos os jobs relacionados aos alunos do Insper

```bash
squeue -A insperhpc
```
Se quiser filtrar apenas o seu usuário:

```bash
squeue -u $USER
```

Este filtra pela fila

```bash
sinfo -p sequana_gpu_dev
```

```bash
sinfo -p sequana_cpu_dev
```

### Testando a APS1 no SDumont

lembrando, para estar na pasta `SCRATCH` utilize o comando:

```bash
cd /scratch/insperhpc/seu-login
```

Crie um novo arquivo dentro da pasta scratch e cole o código base da APS1:

Você pode utilizar o nano para criar um arquivo novo
```bash
nano base.cpp
```
??? note "Código base da APS1 para facilitar a sua vida"
    `base.cpp`
    ```cpp
    #include <algorithm>
    #include <chrono>
    #include <cmath>
    #include <cstdlib>
    #include <iostream>
    #include <limits>
    #include <string>
    #include <vector>

    using namespace std;

    const int CAPACIDADE_NAVIO = 5;

    struct Porto {
        string nome;
        double x;
        double y;
    };

    /*
    POLEMICA:
    Este trecho contém problemas que deixam o código ineficiente.
    */
    double distancia(Porto origem, Porto destino) {
        return sqrt(pow(origem.x - destino.x, 2) +
                    pow(origem.y - destino.y, 2));
    }

    /*
    POLEMICA:
    - Recebe o vetor de portos por valor.
    - Preenche a matriz por colunas, prejudicando a localidade de memória.
    */
    vector<double> criarMatrizDistancias(vector<Porto> portos) {
        int n = static_cast<int>(portos.size());

        // Matriz n x n armazenada em um vetor simples.
        vector<double> matriz(n * n);

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                matriz[i * n + j] = distancia(portos[i], portos[j]);
            }
        }

        return matriz;
    }

    /*
    Calcula o custo de uma ordem de entrega.

    O navio parte do porto de origem carregando ate CAPACIDADE_NAVIO.
    Cada destino representa a entrega de um conteiner.

    Quando a carga termina, o navio retorna ao porto de origem para buscar
    um novo lote. Depois da ultima entrega, ele retorna ao porto de origem.

    POLEMICA:
    - O porto, o vetor de destinos e a rota são recebidos por valor.
    - A matriz de distancias e reconstruida para cada rota avaliada.
    */
    double calcularCusto(Porto origem,
                        vector<Porto> destinos,
                        vector<int> rota) {
        vector<Porto> todosPortos;
        todosPortos.push_back(origem);

        for (int i = 0; i < static_cast<int>(destinos.size()); ++i) {
            todosPortos.push_back(destinos[i]);
        }

        vector<double> matriz = criarMatrizDistancias(todosPortos);
        int quantidadePortos = static_cast<int>(todosPortos.size());

        double custo = 0.0;
        int carga = CAPACIDADE_NAVIO;
        int atual = 0;

        for (int i = 0; i < static_cast<int>(rota.size()); ++i) {
            if (carga == 0) {
                // Distancia entre o porto atual e o porto de origem (indice 0).
                custo += matriz[atual * quantidadePortos];
                atual = 0;
                carga = CAPACIDADE_NAVIO;
            }

            // O destino 0 esta armazenado na posicao 1 da matriz, pois a
            // posicao 0 e reservada para o porto de origem.
            int destino = rota[i] + 1;

            custo += matriz[atual * quantidadePortos + destino];
            atual = destino;
            --carga;
        }

        // Retorno ao porto de origem depois da ultima entrega.
        custo += matriz[atual * quantidadePortos];

        return custo;
    }

    /*
    Busca exaustiva por todas as permutacoes dos portos de destino.

    POLEMICA:
    Origem, destinos e rota são recebidos por valor. Dessa forma, varios
    dados são copiados a cada chamada recursiva.

    melhorCusto e melhorRota são referencias apenas para que a versao base
    produza corretamente a melhor solucao encontrada.
    */
    void permutar(Porto origem,
                vector<Porto> destinos,
                vector<int> rota,
                int inicio,
                double& melhorCusto,
                vector<int>& melhorRota) {
        if (inicio == static_cast<int>(rota.size())) {
            double custo = calcularCusto(origem, destinos, rota);

            if (custo < melhorCusto) {
                melhorCusto = custo;
                melhorRota = rota;
            }

            return;
        }

        for (int i = inicio; i < static_cast<int>(rota.size()); ++i) {
            swap(rota[inicio], rota[i]);

            permutar(origem,
                    destinos,
                    rota,
                    inicio + 1,
                    melhorCusto,
                    melhorRota);

            swap(rota[inicio], rota[i]);
        }
    }

    /*
    POLEMICA:
    Origem, destinos e rota também são recebidos por valor nesta funcao.
    */
    void imprimirRota(Porto origem,
                    vector<Porto> destinos,
                    vector<int> rota) {
        cout << origem.nome;

        int carga = CAPACIDADE_NAVIO;

        for (int i = 0; i < static_cast<int>(rota.size()); ++i) {
            if (carga == 0) {
                cout << " -> " << origem.nome << " [novo lote]";
                carga = CAPACIDADE_NAVIO;
            }

            cout << " -> " << destinos[rota[i]].nome;
            --carga;
        }

        cout << " -> " << origem.nome << '\n';
    }

    int main(int argc, char* argv[]) {
        if (argc < 2) {
            cout << "Uso: " << argv[0] << " <numero_de_destinos>\n";
            cout << "Exemplo: " << argv[0] << " 8\n";
            return 1;
        }

        int n = atoi(argv[1]);

        vector<Porto> portosDisponiveis = {
            {"Paranagua", 100, 180},
            {"Rio_Grande", 120, 300},
            {"Itajai", 90, 220},
            {"Rio_de_Janeiro", 210, 80},
            {"Salvador", 350, 310},
            {"Suape", 430, 420},
            {"Pecem", 510, 500},
            {"Manaus", 260, 650},
            {"Las_Palmas", 800, 460},
            {"Rotterdam", 1100, 220},
            {"Hamburgo", 1160, 190},
            {"Antuerpia", 1120, 230}
        };

        if (n <= 0 || n > static_cast<int>(portosDisponiveis.size())) {
            cout << "Numero de destinos invalido. Use um valor entre 1 e "
                << portosDisponiveis.size() << ".\n";
            return 1;
        }

        Porto origem{"Santos", 150, 100};

        vector<Porto> destinos;
        for (int i = 0; i < n; ++i) {
            destinos.push_back(portosDisponiveis[i]);
        }

        vector<int> rota;
        for (int i = 0; i < n; ++i) {
            rota.push_back(i);
        }

        vector<int> melhorRota;
        double melhorCusto = numeric_limits<double>::max();

        auto inicioTempo = chrono::steady_clock::now();

        permutar(origem,
                destinos,
                rota,
                0,
                melhorCusto,
                melhorRota);

        auto fimTempo = chrono::steady_clock::now();
        chrono::duration<double> tempo = fimTempo - inicioTempo;

        cout << "Melhor custo: " << melhorCusto << " milhas nauticas\n";
        cout << "Melhor rota: ";
        imprimirRota(origem, destinos, melhorRota);
        cout << "Tempo: " << tempo.count() << " segundos\n";

        return 0;
    }

    ```


### Para gerar o binário do código sequencial:

```bash
g++ -SuaFlagDeOtimizaçao base.cpp -o base
```

### Para gerar o binário do código com suporte a paralelismo:

```bash
g++ -fopenmp -SuaFlagDeOtimizaçao paralelo.cpp -o paralelo
```

### Para submeter o seu binário com o `srun`:

Sem suporte a paralelismo o comando fica assim:

Usando a fila de CPU

```bash
srun --partition=sequana_cpu_dev --mem=1G ./base
```

Usando a fila de GPU

```bash
srun --partition=sequana_gpu_dev --mem=1G --gres=gpu:1 ./base
```

Com suporte a paralelismo:

```bash
srun --partition=sequana_cpu_dev --mem=1G --cpus-per-task=4 ./paralelo
```

Usando a fila de GPU

```bash
srun --partition=sequana_gpu_dev --mem=1G --gres=gpu:1 --cpus-per-task=4 ./paralelo
```

### Para submeter o seu binário com o `sbatch`:

Se quiser submeter um job com sbatch na fila CPU sequencial:

`runCPU.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=exemplo_sequencial_cpu
#SBATCH --output=saida_%j.txt
#SBATCH --time=00:10:00
#SBATCH --partition=sequana_cpu_dev
#SBATCH --mem=1G                  # 1 GiB por nó

./base

```


```bash
sbatch runCPU.slurm
```


Se quiser submeter um job com sbatch na fila GPU:

`runGPU.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=exemplo
#SBATCH --output=saida_%j.txt
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=sequana_gpu_dev
#SBATCH --mem=1G                  # 1 GiB por nó

./base

```

```bash
sbatch runGPU.slurm
```

Se quiser submeter um job com sbatch na fila CPU com suporte a paralelismo:

`runCPU.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=exemplo_paralelo_cpu      # nome do job
#SBATCH --output=saida_%j.txt                # arquivo de saida com identificação única
#SBATCH --time=00:10:00                      # pedimos 10 minutos
#SBATCH --cpus-per-task=48                   # valor máximo de threads permitidas
#SBATCH --partition=sequana_cpu_dev          # nome da fila
#SBATCH --mem=1G                             # 1 GiB por nó

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # informando a quantidade de threads disponível para o binário

./paralelo # binário que deve ser compilado com a flag '-fopenmp'

```

Se quiser automatizar os testes variando o número de threads você pode usar algo como:

??? note "Código demo se quiser testar"
    `demo.cpp`
    ```cpp
    #include <iostream>
    #include <vector>
    #include <omp.h>
    #include <cstdlib>

    bool eh_primo(long long n) {
        if (n < 2)
            return false;

        for (long long i = 2; i * i <= n; i++) {
            if (n % i == 0)
                return false;
        }

        return true;
    }

    int main(int argc, char* argv[]) {

        long long LIMITE = 10000000;

        omp_set_dynamic(0);

        int num_threads = omp_get_max_threads();

        std::vector<long long> quantidade_primos(num_threads, 0);
        std::vector<long long> quantidade_numeros(num_threads, 0);
        std::vector<long long> primeiro_numero(num_threads, -1);
        std::vector<long long> ultimo_numero(num_threads, -1);

        double inicio = omp_get_wtime();

        #pragma omp parallel
        {
            int id = omp_get_thread_num();

            long long primos = 0;
            long long numeros = 0;

            #pragma omp for schedule(guided)
            for (long long numero = 0; numero < LIMITE; numero++) {

                if (primeiro_numero[id] == -1) {
                    primeiro_numero[id] = numero;
                }

                ultimo_numero[id] = numero;

                numeros++;

                if (eh_primo(numero)) {
                    primos++;
                }
            }

            quantidade_primos[id] = primos;
            quantidade_numeros[id] = numeros;
        }

        double fim = omp_get_wtime();

        long long total_primos = 0;

        std::cout << "\nDistribuicao das tarefas\n";
        std::cout << "============================================================\n";

        for (int i = 0; i < num_threads; i++) {

            total_primos += quantidade_primos[i];

            std::cout
                << "Thread " << i
                << " | numeros = " << quantidade_numeros[i]
                << " | intervalo = ["
                << primeiro_numero[i]
                << ", "
                << ultimo_numero[i]
                << "]"
                << " | primos = "
                << quantidade_primos[i]
                << "\n";
        }

        std::cout << "\n============================================================\n";
        std::cout << "Threads: " << num_threads << "\n";
        std::cout << "Total de primos: " << total_primos << "\n";
        std::cout << "Tempo total: " << fim - inicio << " s\n";

        return 0;
    }

    ```

    Para compilar:

    ```
    g++ -fopenmp demo.cpp -o demo
    ```



```bash
#!/bin/bash
#SBATCH --job-name=teste_escalabilidade
#SBATCH --output=demo.txt
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=48    #valor máximo de threads permitidas
#SBATCH --partition=sequana_cpu_dev
#SBATCH --mem=1G

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nó: $SLURMD_NODENAME"
echo "CPUs disponíveis: $SLURM_CPUS_PER_TASK"
echo "========================================"

# Testa de 1 até o número de CPUs solicitado ao Slurm
for N in $(seq 1 $SLURM_CPUS_PER_TASK)
do
    export OMP_NUM_THREADS=$N

    echo
    echo "========================================"
    echo "Teste com $N thread(s)"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "========================================"

    time ./demo
done
```


```bash
sbatch runCPU.slurm
```


Se quiser submeter um job com sbatch na fila GPU, mas com suporte a paralelismo em CPU:

`runGPU.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=exemplo              # nome do job
#SBATCH --output=saida_%j.txt           # arquivo de saida com identificação única
#SBATCH --time=00:10:00                 # pedimos 10 minutos
#SBATCH --gres=gpu:1                    # pedimos 1 GPU
#SBATCH --cpus-per-task=16              # Quantidade de threads
#SBATCH --partition=sequana_cpu_dev     # Nome da fila
#SBATCH --mem=1G                        # 1 GiB por nó

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./paralelo

```

```bash
sbatch runGPU.slurm
```

