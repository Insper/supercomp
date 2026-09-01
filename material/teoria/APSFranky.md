# Cluster Franky

Para ter acesso ao Cluster Franky você precisa configurar suas credenciais de acesso e realizar acesso remoto via SSH.

As chaves foram enviadas para o seu email Insper, Faça o download da pasta completa, que contém os arquivos `id_rsa` (chave privada) e `id_rsa.pub` (chave pública), salve essas chaves em algum lugar que você não vai esquecer, depois, siga as instruções abaixo para configurar corretamente seu acesso ao Cluster Franky.


Conecte-se ao cluster utilizando o comando SSH:

O login é o seu "usuario Insper", o endereço de IP foi fornecido durante a aula.


Se você está com o terminal aberto na pasta em que está a sua chave SSH, basta usar o comando:
```bash
ssh -i id_rsa login@ip_do_cluster
```
ou

Se você abriu o terminal em qualquer lugar, então, o comando é este aqui:
```bash
ssh -i caminho_para_a_chave_ssh/id_rsa login@ip_do_cluster
```


### **Ambientação no Cluster Franky**

Antes de começar a fazer pedidos de recursos pro SLURM, vamos conhecer os diferentes hardwares que temos disponível no Franky. Vamos utilizar alguns comandos de sistema operacional para ler os recursos de CPU, memória e GPU disponíveis


### Comandos linux que serão utilizados:

* `lscpu`: mostra detalhes da CPU (núcleos, threads, memória cache...)
* `cat /proc/meminfo`: mostra detalhes sobre a memória RAM 
* `nvidia-smi`: mostra detalhes de GPU, se disponível

### Explorando com o SRUN
Vale lembrar que podemos pedir via SRUN um terminal dentro do nó de computação, para, de forma livre, executar qualquer comando:

```bash
srun --partition=sunny_cpu --mem=1G --pty bash
```

Para sair, basta digitar no terminal:
```bash
exit
```

Ou, podemos usar o SRUN com um comando definido que será executado no nó de computação de forma direta pelo terminal:


Para visualizar o hardware disponível na fila escolhida:

```bash
srun --partition=sunny_cpu --mem=1G --pty bash -c \
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

**Você sempre pode usar o HW de CPU da fila GPU**

```bash
srun --partition=sunny_gpu --mem=1G --pty bash -c \
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

O comando `sinfo` mostra quais são as filas e quais são os status dos nós 

```bash
sinfo
```
O comando abaixo mostra detalhes sobre os recursos de cada fila

```bash
scontrol show partition 
```
### SBATCH — Submissão de Jobs no SLURM

`sbatch` é o comando usado para **enviar um job para a fila do cluster**.

Diferente do `srun`, ele **não é interativo**.  
Você cria um script e o SLURM executa quando houver recursos disponíveis.


Para criar um arquivo lançador de job:

```bash
nano teste.slurm
```

Cole o conteúdo abaixo:

```bash
#!/bin/bash

#SBATCH --job-name=sbatch_belezinha        # Nome do job (aparece no squeue)
#SBATCH --partition=sunny_gpu           # Fila (partition) onde o job será executado
#SBATCH --cpus-per-task=1            # Número de threads 
#SBATCH --mem=1G                     # Memória RAM solicitada
#SBATCH --time=00:05:00              # Tempo máximo de execução (HH:MM:SS)
#SBATCH --output=meu_orgulho_%j.log  # Arquivo de saída (%j = ID do job)

echo "=== HOSTNAME ==="
hostname
echo

echo "=== MEMORIA (GB) ==="
cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable|Swap' | \
awk '{printf "%s %.2f GB\n", $1, $2 / 1048576}'
echo

echo "=== CPU INFO ==="
lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)|cache'
echo

echo "=== GPU INFO ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi não disponível"
fi
#Sleep desnecessário, só para você conseguir enxergar o seu job na fila do slurm
sleep 20
```

Salve usando Crtl + s, 

e saia usando Crlt + x.


Para submeter o job:

```bash
sbatch teste.sh
```

Você verá algo como:

```
Submitted batch job 12345
```

O número é o **ID do job**.


Para verificar se o lançador deu certo e o seu job está rodando:

```bash
squeue 
```
Se quiser filtrar apenas o seu usuário:

```bash
squeue -u $USER
```

Quando o job terminar, será criado um arquivo como:

```
saida_12345.log
```

Visualize com:

```bash
cat saida_12345.log
```


### Testando a APS1 no Cluster Franky

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
srun --partition=sunny_cpu --mem=1G ./meu_programa_sequencial
```

Usando a fila de GPU

```bash
srun --partition=sunny_cpu --mem=1G --gres=gpu:1 ./meu_programa_sequencial
```

Com suporte a paralelismo:

```bash
srun --partition=sunny_cpu --mem=1G --cpus-per-task=4 ./meu_programa_paralelo
```

Usando a fila de GPU

```bash
srun --partition=sunny_gpu --mem=1G --cpus-per-task=4 ./meu_programa_paralelo
```

### Para submeter o seu binário com o `sbatch`:

Se quiser submeter um job com sbatch na fila CPU sequencial:

`runCPU.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=exemplo_sequencial_cpu
#SBATCH --output=saida_%j.txt
#SBATCH --time=00:10:00
#SBATCH --partition=sunny_cpu
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
#SBATCH --partition=sunny_gpu
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
#SBATCH --cpus-per-task=16                   # quantidade de threads
#SBATCH --partition=sunny_cpu                # nome da fila
#SBATCH --mem=1G                             # 1 GiB por nó

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # informando a quantidade de threads disponível para o binário

./meu_binario # binário que deve ser compilado com a flag '-fopenmp'

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


    Para compilar:

    ```
    g++ -fopenmp demo.cpp -o demo
    ```
    ```



```bash
#!/bin/bash
#SBATCH --job-name=teste_escalabilidade
#SBATCH --output=demo.txt
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=16    #valor máximo de threads permitidas
#SBATCH --partition=sunny_cpu
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
#SBATCH --cpus-per-task=16              # Quantidade de threads
#SBATCH --partition=sunny_gpu     # Nome da fila
#SBATCH --mem=1G                        # 1 GiB por nó

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./meu_binario

```

```bash
sbatch runGPU.slurm
```

