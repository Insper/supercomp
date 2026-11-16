# Estudo de caso: pipeline assíncrono CPU-GPU

Imagine que você recebeu um vídeo de trânsito e precisa destacar regiões de interesse em cada imagem. Você começa com uma solução simples: extrai os frames, identifica bordas, agrupa os pixels conectados e desenha retângulos ao redor das regiões encontradas.

A primeira versão funciona de forma sequencial em CPU, ela realiza todo o processamento de uma imagem antes de começar a próxima.

Neste estudo de caso, vamos analizar como otimizar o código disponível em [`base.cpp`](https://github.com/liciascl/segue-carros/blob/main/base.cpp).

Considerando que vamos precisar responder duas perguntas: **como processar os pixels com mais eficiência e como evitar que CPU e GPU fiquem esperando uma pela outra?**

### Conhecendo o problema e o código base

Antes de otimizar, precisamos entender o que o programa faz.

O código base recebe uma sequência de arquivos como `frames/frame_0001.png`, converte cada imagem para tons de cinza e aplica o filtro Sobel para destacar mudanças de intensidade. Em seguida, transforma o resultado em uma máscara binária: pixels acima do limiar tornam-se brancos; os demais tornam-se pretos.

A busca de componentes conectados percorre essa máscara, agrupa pixels brancos ligados por quatro direções e mantém regiões com mais de 500 pixels. Por fim, o programa desenha os retângulos na imagem RGB e salva o resultado.

Observe o trecho central do código original:

```cpp
// Buffers intermediários
unsigned char* gray = new unsigned char[w*h];
unsigned char* sob  = new unsigned char[w*h];
unsigned char* bin  = new unsigned char[w*h];

// Pipeline de visão computacional
rgb2gray(input, gray, w, h);
sobel(gray, sob, w, h);
threshold_bin(sob, bin, w, h, 100);

// Componentes conectados
vector<Box> objects = findComponents(bin, w, h);

// Desenho dos retângulos
draw_boxes(input, w, h, objects);
```

Esse trecho está dentro do laço de frames. Depois dele, o programa salva o PNG, libera os buffers e avança para a próxima imagem.

Perceba a consequência: todas as etapas executam na CPU, uma após a outra. Além disso, os três buffers intermediários são alocados novamente a cada frame.

Embora o projeto use a imagem de trânsito como aplicação, o algoritmo identifica **regiões conectadas nas bordas binarizadas**. Ele não reconhece semanticamente um carro nem mantém sua identidade ao longo do vídeo.

### Preparando e executando a versão base

Vamos primeiro reproduzir o ponto de partida.

Faça o clone o repositório e entre no diretório:

```bash
git clone https://github.com/liciascl/segue-carros.git
cd segue-carros
```

O repositório contém `base.cpp`, os headers STB, o vídeo `transito.mp4` e um diretório com FFmpeg.

Crie o diretório de entrada e extraia os frames, seguindo o comando do [README do projeto](https://github.com/liciascl/segue-carros/blob/main/README.md):

```bash
mkdir -p frames

./ffmpeg/ffmpeg -i transito.mp4 -r 30 frames/frame_%04d.png
```

Para um experimento inicial menor, limite a extração a 100 frames:

```bash
./ffmpeg/ffmpeg -i transito.mp4 -r 30 \
  -frames:v 100 frames/frame_%04d.png
```

O comando abaixo compila o código base com otimização do compilador:

```bash
g++ -O3  base.cpp -o carros_base
```

Execute inicialmente com cinco frames:

```bash
srun --partition=merry_cpu --mem=1G ./carros_base 5
```

Os resultados são gravados em `out/frame_XXXX.png`. Execute a aplicação na raiz do projeto para que os caminhos relativos funcionem.

Para reconstruir o vídeo, o README apresenta:

```bash
./ffmpeg/ffmpeg -framerate 30 -i out/frame_%04d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p saida.mp4
```

Ao comparar execuções com quantidades diferentes de frames, confira se `out` contém apenas os resultados que você pretende usar.

### Primeira decisão: distribuir o trabalho dos pixels

Agora que conhecemos a versão base, observe a conversão para cinza:

```cpp
for (int i = 0; i < w * h; i++) {
    int idx = i * 3;

    float r = input[idx];
    float g = input[idx + 1];
    float b = input[idx + 2];

    gray[i] = (unsigned char)(
        0.299f * r + 0.587f * g + 0.114f * b
    );
}
```

Para calcular o pixel de índice `i`, não precisamos conhecer o resultado do pixel anterior. Essa independência permite dividir o trabalho entre threads.

Na versão CUDA:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= N) return;

int idx = i * 3;

gray[i] = (unsigned char)(
    0.299f * rgb[idx] +
    0.587f * rgb[idx + 1] +
    0.114f * rgb[idx + 2]
);
```

Cada thread passa a cuidar de um pixel. O threshold utiliza a mesma distribuição:

```cpp
bin[i] = (in[i] > T) ? 255 : 0;
```

A técnica é o **paralelismo de dados**: aplicar uma mesma operação a muitos elementos independentes.

O ganho potencial vem da execução de muitos cálculos em paralelo. Entretanto, ainda precisamos enviar os dados à GPU e receber o resultado. Por isso, analisar somente o tempo dos kernels não basta para avaliar a aplicação.

### O ponto central: por que usar tiling no Sobel?

O Sobel apresenta um problema diferente. Um pixel de saída depende dos pixels de uma janela 3 × 3.

No código base, essa vizinhança é percorrida assim:

```cpp
for (int ky = -1; ky <= 1; ky++) {
    for (int kx = -1; kx <= 1; kx++) {
        int pixel = gray[(y + ky) * w + (x + kx)];

        sumX += pixel * Gx[ky + 1][kx + 1];
        sumY += pixel * Gy[ky + 1][kx + 1];
    }
}
```

Pense agora em duas posições vizinhas. Quando a janela avança uma coluna, seis das nove posições continuam pertencendo à vizinhança.

Ou seja: **há uma grande oportunidade de reaproveitar pixels já consultados**.

Uma versão GPU direta pode deixar cada thread buscar seus vizinhos na memória global. A versão do handout usa outra organização: os threads carregam coletivamente uma região para a memória compartilhada e consultam essa região durante o cálculo.

Essa organização é o **tiling**.

O código utiliza um bloco de 32 × 32 threads:

```cpp
static const int SOBEL_BX = 32;
static const int SOBEL_BY = 32;
```

Esse bloco pode produzir 32 × 32 resultados. Porém, os threads das extremidades também precisam consultar vizinhos que ficam fora da região central.

Por isso, o tile contém uma borda adicional, chamada **halo**:

| Região | Dimensão |
|---|---:|
| Região de saída do bloco | 32 × 32 |
| Região de entrada com halo | 34 × 34 |
| Armazenamento compartilhado | 1.156 bytes |

Como cada intensidade ocupa um byte:

\[
(32+2)(32+2)=1.156\text{ bytes}
\]

O código reserva essa memória no lançamento:

```cpp
size_t shmem =
    (size_t)(SOBEL_BX + 2) * (SOBEL_BY + 2);
```

Dentro do kernel:

```cpp
extern __shared__ unsigned char tile[];
```

Cada thread carrega seu pixel central. Os threads das extremidades também carregam as faixas e os cantos do halo.

### Qual é o impacto concreto do tiling?

Vamos quantificar o reúso para um bloco completo no interior da imagem.

Uma implementação direta que consulte nove posições por saída realiza, no nível do algoritmo:

\[
1.024 \times 9 = 9.216\text{ consultas}
\]

O carregamento coletivo do tile cobre essas vizinhanças com:

\[
34 \times 34 = 1.156\text{ pixels de entrada}
\]

Isso representa cerca de oito vezes menos carregamentos explícitos da memória global nesse modelo simplificado.

Há uma ressalva importante: no Sobel, o coeficiente central é zero. Uma implementação que omita essa posição já precisaria consultar somente oito vizinhos:

\[
1.024 \times 8 = 8.192
\]

Mesmo nessa comparação, existe uma redução expressiva das consultas globais explícitas: aproximadamente sete vezes.

**Essas razões descrevem o reúso de dados; não são fatores de aceleração medidos.** Caches podem atender consultas repetidas de uma versão direta, e o tiling acrescenta trabalho para carregar o halo, indexar a memória compartilhada e sincronizar os threads.

O efeito esperado é reduzir a pressão sobre a memória global. Quanto isso diminui o tempo do Sobel depende da GPU e da implementação usada na comparação.

### Por que a sincronização faz parte da técnica?

Após carregar o tile, o código executa:

```cpp
__syncthreads();
```

Imagine que seu thread precisa do pixel carregado pelo thread vizinho. Sem uma barreira, ele poderia tentar usar essa posição antes de o vizinho terminar a escrita.

A barreira separa duas fases: primeiro todos colaboram no carregamento; depois utilizam os dados.

Por esse motivo, a verificação que encerra threads fora da imagem fica depois dela:

```cpp
__syncthreads();

if (x >= w || y >= h) return;
```

Isso também importa nos blocos parciais, quando as dimensões da imagem não são múltiplas de 32.

### Um tile maior sempre é melhor?

Observe a proporção entre dados carregados e resultados:

| Bloco | Tile com halo | Pixels carregados por saída |
|---|---:|---:|
| 16 × 16 | 18 × 18 = 324 | 1,266 |
| 32 × 32 | 34 × 34 = 1.156 | 1,129 |

O bloco maior dilui melhor o custo do halo. Porém, utiliza 1.024 threads, e isso pode limitar a quantidade de blocos simultâneos em um multiprocessador, dependendo dos recursos da GPU.

Portanto, escolher o tamanho do tile envolve um equilíbrio entre reúso, custo do halo e ocupação. A configuração 32 × 32 é uma decisão da solução, e seu desempenho precisa ser comparado experimentalmente com outras configurações.

### Segunda decisão: manter os intermediários na GPU

Depois de acelerar os filtros, seria pouco eficiente devolver cada resultado intermediário à CPU.

A solução mantém quatro buffers no dispositivo:

```cpp
unsigned char* d_rgb;
unsigned char* d_gray;
unsigned char* d_sobel;
unsigned char* d_bin;
```

A imagem entra como RGB, passa pelos três kernels e retorna apenas como máscara binária.

Para \(N=w\times h\):

| Comunicação por frame | Volume |
|---|---:|
| RGB enviado à GPU | \(3N\) bytes |
| Máscara devolvida à CPU | \(N\) bytes |
| Total | \(4N\) bytes |

A CPU preserva o RGB original para desenhar os retângulos. Assim, não precisa recebê-lo novamente.

Aqui, a otimização atua sobre a **movimentação dos dados**. Os intermediários continuam na memória global da GPU entre kernels; a solução não faz fusão dos três filtros em um único kernel.

### Terceira decisão: reservar os buffers uma vez

No código base, `gray`, `sob` e `bin` são alocados e liberados a cada imagem.

Na versão otimizada, os buffers CUDA e pinned são reservados antes do laço e reutilizados:

```cpp
cudaMallocHost(&bufs[s].h_rgb_pin, bytes_rgb);
cudaMallocHost(&bufs[s].h_bin_pin, bytes_gray);

cudaMalloc(&bufs[s].d_rgb, bytes_rgb);
cudaMalloc(&bufs[s].d_gray, bytes_gray);
cudaMalloc(&bufs[s].d_sobel, bytes_gray);
cudaMalloc(&bufs[s].d_bin, bytes_gray);
```

A cada frame, trocamos o conteúdo dos buffers, isso retira alocações CUDA do processamento repetitivo. Também mantém o consumo reservado do pipeline constante ao aumentar o número de frames.

O programa ainda possui alocações temporárias na leitura PNG e na busca de componentes. Portanto, o reúso se aplica aos buffers persistentes do pipeline.

### Quarta decisão: permitir que CPU e GPU trabalhem juntas

Até aqui, discutimos principalmente o custo de processar um frame. Agora vamos olhar para a sequência inteira.

A solução usa dois conjuntos independentes de buffers:

```cpp
int s = (frame - 1) % NUM_STREAMS;
StreamBuffer& buf = bufs[s];
```

Com duas streams:

| Frame | Slot utilizado |
|---|---:|
| 1 | 0 |
| 2 | 1 |
| 3 | 0 |
| 4 | 1 |

Essa alternância é o **double buffering**.

Enquanto a GPU processa o frame 1, a CPU pode ler o frame 2. Depois, enquanto a GPU trabalha no frame 2, a CPU pode identificar componentes, desenhar e salvar o frame 1.

Cada stream preserva a ordem das operações daquele frame:

```cpp
cudaMemcpyAsync(..., buf.stream);

rgb2gray_kernel<<<..., buf.stream>>>(...);
sobel_tiled_shared_kernel<<<..., buf.stream>>>(...);
threshold_kernel<<<..., buf.stream>>>(...);

cudaMemcpyAsync(..., buf.stream);
```

As etapas de um mesmo frame continuam dependentes. O que muda é a possibilidade de manter frames diferentes em andamento.

### O papel da memória pinned

Os buffers host são alocados com `cudaMallocHost`. Essa memória permite as transferências assíncronas utilizadas pelo pipeline.

Como `stbi_load` devolve memória comum, o programa realiza uma cópia antes do envio:

```cpp
memcpy(buf.h_rgb_pin, disk_data, bytes_rgb);
stbi_image_free(disk_data);
```

Existe um custo adicional nessa cópia. Seu benefício potencial aparece quando a organização assíncrona permite esconder parte do tempo de transferência ou de processamento atrás de outras tarefas.

### Quando o programa espera?

Antes de consumir e reutilizar um slot ocupado:

```cpp
cudaStreamSynchronize(buf.stream);
```

A CPU precisa dessa espera para garantir que a máscara esteja disponível. Depois, encontra componentes, desenha e salva. Só então o buffer pode receber outro frame.

A posição dessa sincronização é fundamental: esperar imediatamente após cada kernel reduziria a oportunidade de sobreposição.

Streams e memória pinned **permitem** a sobreposição. Sua ocorrência e intensidade dependem do hardware e da duração das etapas.


??? note "Codigo completo otimizado"
    ```cpp
    /*
    ESTUDO DE CASO — Pipeline de imagens com CUDA

    Vamos acompanhar um frame: a CPU lê o PNG, a GPU converte RGB para cinza,
    aplica Sobel e threshold, e a CPU encontra componentes, desenha e salva.
    O programa destaca regiões conectadas; não reconhece nem rastreia veículos.

    As otimizações atuam em dois níveis:
    1. Dentro do frame: paralelismo por pixel e tiling no Sobel.
    2. Entre frames: dois slots com buffers próprios e streams CUDA.

    Leia os comentários como um percurso pelos dados. Host (h_) significa CPU;
    device (d_) significa GPU. H2D é host -> device; D2H é device -> host.
    A CPU usa uma única thread; a sobreposição vem do enfileiramento na GPU.

    */

    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"
    #include <iostream>
    #include <vector>
    #include <queue>
    #include <cmath>
    #include <cstdio>
    #include <cstring>
    #include <filesystem>
    #include <chrono>
    #include <algorithm>
    #include <cuda_runtime.h>
    using namespace std;
    namespace fs = std::filesystem;
    using namespace std::chrono;

    // =============================================================
    // CONFIGURAÇÕES

    // =============================================================
    // O bloco Sobel cobre 32 x 32 posições: 1.024 threads. Verifique o limite
    // Um bloco maior dilui o halo, mas não garante melhor ocupação/tempo.
    static const int SOBEL_BX    = 32;
    static const int SOBEL_BY    = 32;
    // Grayscale e threshold usam blocos 1D, pois cada saída depende de um pixel.
    static const int THREADS_1D  = 256;   
    // Dois slots permitem manter dois frames em andamento.
    // Cada slot precisa de buffers próprios para não sobrescrever dados em uso.
    static const int NUM_STREAMS = 2;
    // Apenas magnitudes estritamente maiores que 100 viram pixels brancos.
    static const int THRESHOLD   = 100;
    // Área é a quantidade de pixels brancos conectados, não a área do retângulo.
    static const int MIN_AREA    = 500;
    struct Box { int minx, miny, maxx, maxy; };

    // =============================================================
    // KERNELS 

    // =============================================================
    // rgb2gray
    // Aqui, um thread calcula um pixel, sem depender dos demais: paralelismo de dados.
    __global__ void rgb2gray_kernel(const unsigned char* rgb,
                                    unsigned char*       gray,
                                    int N)
    {
        // Índice linear: posição do bloco no grid + posição do thread no bloco.
        // Threads excedentes no último bloco são descartados pela checagem seguinte.
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        // RGB é intercalado: R,G,B,R,G,B,...; cinza tem apenas um byte por pixel.
        int idx = i * 3;
        float r = rgb[idx];
        float g = rgb[idx + 1];
        float b = rgb[idx + 2];
        // A soma ponderada produz a intensidade. A conversão para unsigned char
        // descarta a parte fracionária: diferenças numéricas podem afetar o threshold.
        gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
    // sobel_tiled_shared_kernel 
    // TILING: cada bloco carrega uma vizinhança para a memória compartilhada.
    // Os pixels carregados são reutilizados por vários threads durante o Sobel.
    // Para 32 x 32 saídas, o tile de entrada tem 34 x 34 pixels (halo de raio 1).
    // Em um bloco interior completo, isso representa 1.156 cargas globais explícitas,
    // em vez de 8.192 consultas de uma versão direta que leia oito vizinhos por saída.
    // A razão ~7,1 descreve reúso, não speedup: caches, sincronização e recursos da
    // GPU também contam. Os cálculos ainda consultam dados na memória compartilhada.
    // Tiles de blocos vizinhos repetem parte dos halos: o reúso é local ao bloco.
    __global__ void sobel_tiled_shared_kernel(const unsigned char* gray,
                                            unsigned char*       out,
                                            int w, int h)
    {
        // Memória compartilhada dinâmica: o tamanho em bytes é fornecido no terceiro
        // parâmetro do lançamento <<<grid, bloco, bytes_shared, stream>>>.
        // Cada bloco possui seu próprio tile, usado apenas durante esse kernel.
        extern __shared__ unsigned char tile[];
        // tx/ty são coordenadas locais; x/y são coordenadas na imagem completa.
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x  = blockIdx.x * blockDim.x + tx;
        int y  = blockIdx.y * blockDim.y + ty;
        // +2 reserva uma posição de halo em cada lado. O centro é deslocado em +1.
        // Acesso 2D em vetor linear: linha * largura + coluna.
        int tile_w = blockDim.x + 2;
        int s_x    = tx + 1;
        int s_y    = ty + 1;
        // Todo thread participa do carregamento, inclusive nos blocos parciais.
        // Posições centrais fora da imagem recebem zero.
        // Pixel central
        tile[s_y * tile_w + s_x] = (x < w && y < h) ? gray[y * w + x] : 0;
        // Threads das extremidades assumem trabalho adicional: carregar o halo.
        // As condições evitam leituras globais fora da imagem.
        // Bordas laterais
        if (tx == 0)
            tile[s_y * tile_w + 0]              = (x > 0   && y < h) ? gray[y * w + (x-1)] : 0;
        if (tx == blockDim.x - 1)
            tile[s_y * tile_w + tile_w - 1]     = (x+1 < w && y < h) ? gray[y * w + (x+1)] : 0;
        if (ty == 0)
            tile[s_x]                            = (y > 0   && x < w) ? gray[(y-1) * w + x] : 0;
        if (ty == blockDim.y - 1)
            tile[(blockDim.y+1) * tile_w + s_x] = (y+1 < h && x < w) ? gray[(y+1) * w + x] : 0;
        // As diagonais também entram no Sobel, por isso as quatro posições de canto
        // precisam ser carregadas. O halo fica pronto pela colaboração do bloco.
        // Quatro cantos
        if (tx == 0 && ty == 0)
            tile[0] = (x > 0 && y > 0) ? gray[(y-1) * w + (x-1)] : 0;
        if (tx == blockDim.x-1 && ty == 0)
            tile[tile_w - 1] = (x+1 < w && y > 0) ? gray[(y-1) * w + (x+1)] : 0;
        if (tx == 0 && ty == blockDim.y-1)
            tile[(blockDim.y+1) * tile_w] = (x > 0 && y+1 < h) ? gray[(y+1) * w + (x-1)] : 0;
        if (tx == blockDim.x-1 && ty == blockDim.y-1)
            tile[(blockDim.y+1) * tile_w + tile_w - 1] = (x+1 < w && y+1 < h) ? gray[(y+1) * w + (x+1)] : 0;
        // Espere todos terminarem a carga: um thread usará pixels escritos por outros.
        // O retorno por coordenadas inválidas fica DEPOIS desta barreira, mantendo
        // os threads participantes até terminar a fase coletiva.
        __syncthreads();
        if (x >= w || y >= h) return;
        // As consultas abaixo usam o tile local.
        int sumX = 0, sumY = 0;
        sumX += -tile[(s_y-1)*tile_w + (s_x-1)] + tile[(s_y-1)*tile_w + (s_x+1)];
        sumX += -2*tile[s_y*tile_w   + (s_x-1)] + 2*tile[s_y*tile_w   + (s_x+1)];
        sumX += -tile[(s_y+1)*tile_w + (s_x-1)] + tile[(s_y+1)*tile_w + (s_x+1)];
        sumY += -tile[(s_y-1)*tile_w + (s_x-1)] - 2*tile[(s_y-1)*tile_w + s_x] - tile[(s_y-1)*tile_w + (s_x+1)];
        sumY +=  tile[(s_y+1)*tile_w + (s_x-1)] + 2*tile[(s_y+1)*tile_w + s_x] + tile[(s_y+1)*tile_w + (s_x+1)];
        // Magnitude euclidiana do gradiente. 
        int mag = (int)sqrtf((float)(sumX*sumX + sumY*sumY));
        if (mag > 255) mag = 255;
        out[y * w + x] = (unsigned char)mag;
    }
    // threshold_kernel
    // Outro kernel independente por pixel: transforma intensidade em máscara 0/255.
    // Esta máscara, e não os intermediários gray/Sobel, será devolvida à CPU.
    __global__ void threshold_kernel(const unsigned char* in,
                                    unsigned char*       bin,
                                    int N, int T)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        bin[i] = (in[i] > T) ? 255 : 0;
    }
    // componentes conectados
    // Daqui em diante, o trabalho é CPU. BFS percorre cada região branca conexa.
    // As alocações visited/fila ainda acontecem por resultado: o reúso dos buffers
    // CUDA não elimina todas as alocações da aplicação.
    vector<Box> findComponents(const unsigned char* bin, int w, int h)
    {
        vector<Box> boxes;
        // Marcar ao enfileirar evita inserir o mesmo pixel várias vezes na BFS.
        vector<int> visited(w * h, 0);
        int dx[4] = {1, -1, 0,  0};
        int dy[4] = {0,  0, 1, -1};
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                // Um pixel branco ainda não visitado inicia um novo componente.
                if (bin[idx] != 255 || visited[idx]) continue;
                // A fila guarda a fronteira da busca. Cada remoção expande os vizinhos.
                queue<pair<int,int>> q;
                q.push({x, y});
                visited[idx] = 1;
                // O primeiro pixel inicia a caixa; cada pixel encontrado amplia seus limites.
                Box b;
                b.minx = b.maxx = x;
                b.miny = b.maxy = y;
                int area = 0;
                while (!q.empty()) {
                    auto [cx, cy] = q.front(); q.pop();
                    // Conte pixels do componente; este valor será usado no filtro de tamanho.
                    area++;
                    b.minx = min(b.minx, cx); b.miny = min(b.miny, cy);
                    b.maxx = max(b.maxx, cx); b.maxy = max(b.maxy, cy);
                    for (int k = 0; k < 4; k++) {
                        int nx = cx + dx[k], ny = cy + dy[k];
                        // Vizinhos fora da imagem não podem ser consultados.
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        int ni = ny * w + nx;
                        if (bin[ni] == 255 && !visited[ni]) {
                            visited[ni] = 1;
                            q.push({nx, ny});
                        }
                    }
                }
                // Guarde somente componentes que passam pelo limiar de área.
                if (area > MIN_AREA) boxes.push_back(b);
            }
        }
        return boxes;
    }

    // =============================================================
    // DESENHAR CAIXAS (CPU)

    // =============================================================
    // Altera o RGB original preservado no host, desenhando R=255, G=0, B=0.
    // Não é preciso trazer a imagem colorida de volta da GPU.
    void draw_boxes(unsigned char* img, int w, int h, vector<Box>& boxes)
    {
        for (auto& b : boxes) {
            for (int x = b.minx; x <= b.maxx; x++) {
                int top = (b.miny * w + x) * 3;
                int bot = (b.maxy * w + x) * 3;
                img[top]=255; img[top+1]=0; img[top+2]=0;
                img[bot]=255; img[bot+1]=0; img[bot+2]=0;
            }
            for (int y = b.miny; y <= b.maxy; y++) {
                int L = (y * w + b.minx) * 3;
                int R = (y * w + b.maxx) * 3;
                img[L]=255; img[L+1]=0; img[L+2]=0;
                img[R]=255; img[R+1]=0; img[R+2]=0;
            }
        }
    }
    // BUFFER DE STREAM
    // Um slot reúne o estado de UM frame em andamento. A independência dos buffers
    // permite que o outro slot trabalhe sem disputar o mesmo armazenamento.
    // Por slot: host pinned = RGB (3N) + máscara (N) = 4N bytes;
    // device = RGB (3N) + gray (N) + Sobel (N) + máscara (N) = 6N bytes.
    // Com dois slots: 8N bytes pinned e 12N device, além de temporários da CPU.
    struct StreamBuffer {
        // Uma stream é uma fila ordenada: operações deste frame respeitam sua ordem.
        // Streams distintas permitem concorrência, conforme o hardware e os recursos.
        cudaStream_t stream;
        // Pinned: páginas fixadas na memória física, usadas nas cópias assíncronas.
        // O RGB permanece válido até concluir componentes, desenho e escrita.
        unsigned char* h_rgb_pin;  // RGB original 
        unsigned char* h_bin_pin;  // resultado D2H
        unsigned char* d_rgb;
        unsigned char* d_gray;
        unsigned char* d_sobel;
        unsigned char* d_bin;
        // Identifica o frame mesmo quando seu resultado é consumido depois.
        // in_use indica resultado pendente; não informa sozinho se a GPU já terminou.
        int  frame_id;
        bool in_use;
    };

    // =============================================================
    // MAIN

    // =============================================================
    int main(int argc, char* argv[])
    {
        // -1 percorre até o primeiro arquivo não carregado. 
        int max_frames = -1;
        if (argc > 1) {
            max_frames = atoi(argv[1]);
            cout << "Modo teste: " << max_frames << " frames\n";
        }
        fs::create_directory("out");
        // Leia o primeiro frame só para dimensionar os buffers persistentes.
        // Essa leitura será repetida no laço e está fora da medição abaixo.
        int w, h, c_dummy;
        {
            unsigned char* probe = stbi_load("frames/frame_0001.png", &w, &h, &c_dummy, 3);
            if (!probe) { cerr << "Nao encontrei frames/frame_0001.png\n"; return 1; }
            stbi_image_free(probe);
        }
        int    N          = w * h;
        size_t bytes_rgb  = (size_t)N * 3;
        size_t bytes_gray = (size_t)N;
        // Divisão inteira arredondada para cima cobre também linhas/colunas restantes.
        // O kernel trata os threads excedentes de cada bloco parcial.
        // Grid Sobel 2D
        dim3 block_sobel(SOBEL_BX, SOBEL_BY);
        dim3 grid_sobel((w + SOBEL_BX - 1) / SOBEL_BX,
                        (h + SOBEL_BY - 1) / SOBEL_BY);
        // Grid 1D para rgb2gray e threshold
        int blocks_1d = (N + THREADS_1D - 1) / THREADS_1D;
        // 34 x 34 x sizeof(unsigned char) = 1.156 bytes compartilhados por bloco.
        // shared memory do Sobel
        size_t shmem = (size_t)(SOBEL_BX + 2) * (SOBEL_BY + 2);
        // Alocar antes do laço evita cudaMalloc/cudaMallocHost a cada frame.

        StreamBuffer bufs[NUM_STREAMS];
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamCreate(&bufs[s].stream);
            // Memória pinned viabiliza as transferências assíncronas deste pipeline.
            cudaMallocHost(&bufs[s].h_rgb_pin, bytes_rgb);
            cudaMallocHost(&bufs[s].h_bin_pin, bytes_gray);
            cudaMalloc(&bufs[s].d_rgb,   bytes_rgb);
            cudaMalloc(&bufs[s].d_gray,  bytes_gray);
            cudaMalloc(&bufs[s].d_sobel, bytes_gray);
            cudaMalloc(&bufs[s].d_bin,   bytes_gray);
            bufs[s].frame_id = -1;
            bufs[s].in_use   = false;
        }

        auto t0 = high_resolution_clock::now();
        int frame = 1;
        while (true) {
            if (max_frames != -1 && frame > max_frames) break;
            int s = (frame - 1) % NUM_STREAMS;
            StreamBuffer& buf = bufs[s];
            // processa frame anterior deste slot
            if (buf.in_use) {
                cudaStreamSynchronize(buf.stream);
                vector<Box> boxes = findComponents(buf.h_bin_pin, w, h);
                draw_boxes(buf.h_rgb_pin, w, h, boxes);  // RGB já em memória
                char outname[256];
                sprintf(outname, "out/frame_%04d.png", buf.frame_id);
                stbi_write_png(outname, w, h, 3, buf.h_rgb_pin, w * 3);
            }
            // Só após terminar desenho/escrita é seguro sobrescrever o RGB deste slot.
            // lê frame do disco
            char filename[256];
            sprintf(filename, "frames/frame_%04d.png", frame);
            int fw, fh, fc;
            unsigned char* disk_data = stbi_load(filename, &fw, &fh, &fc, 3);
            if (!disk_data) {
                buf.in_use = false;
                cout << "\nFim: " << filename << endl;
                break;
            }
            memcpy(buf.h_rgb_pin, disk_data, bytes_rgb);
            stbi_image_free(disk_data);
            // Todas as operações recebem a mesma stream: H2D -> gray -> Sobel ->
            // threshold -> D2H. Não há necessidade de uma espera CPU entre elas.
            // Depois de enfileirar, a CPU pode preparar/consumir o outro slot.
            // pipeline GPU assíncrono
            cudaMemcpyAsync(buf.d_rgb, buf.h_rgb_pin, bytes_rgb,
                            cudaMemcpyHostToDevice, buf.stream);
            rgb2gray_kernel<<<blocks_1d, THREADS_1D, 0, buf.stream>>>(
                buf.d_rgb, buf.d_gray, N);
            sobel_tiled_shared_kernel<<<grid_sobel, block_sobel, shmem, buf.stream>>>(
                buf.d_gray, buf.d_sobel, w, h);
            threshold_kernel<<<blocks_1d, THREADS_1D, 0, buf.stream>>>(
                buf.d_sobel, buf.d_bin, N, THRESHOLD);
            cudaMemcpyAsync(buf.h_bin_pin, buf.d_bin, bytes_gray,
                            cudaMemcpyDeviceToHost, buf.stream);
            buf.frame_id = frame;
            buf.in_use   = true;
            cout << "\rProcessando: " << frame << flush;
            frame++;
        }
        // Flush final
        for (int s = 0; s < NUM_STREAMS; s++) {
            StreamBuffer& buf = bufs[s];
            if (!buf.in_use) continue;
            cudaStreamSynchronize(buf.stream);
            vector<Box> boxes = findComponents(buf.h_bin_pin, w, h);
            draw_boxes(buf.h_rgb_pin, w, h, boxes);
            char outname[256];
            sprintf(outname, "out/frame_%04d.png", buf.frame_id);
            stbi_write_png(outname, w, h, 3, buf.h_rgb_pin, w * 3);
            buf.in_use = false;
        }
        auto t1 = high_resolution_clock::now();
        double total = duration<double>(t1 - t0).count();
        cout << "\n\n===== FINAL =====\n";
        cout << "Frames processados: " << frame - 1 << "\n";
        cout << "Tempo total:        " << total << " s\n";
        cout << "Media por frame:    " << total / (frame - 1) * 1000.0 << " ms\n";
        for (int s = 0; s < NUM_STREAMS; s++) {
            cudaStreamDestroy(bufs[s].stream);
            cudaFreeHost(bufs[s].h_rgb_pin);
            cudaFreeHost(bufs[s].h_bin_pin);
            cudaFree(bufs[s].d_rgb);
            cudaFree(bufs[s].d_gray);
            cudaFree(bufs[s].d_sobel);
            cudaFree(bufs[s].d_bin);
        }
        return 0;
    }

    ```


### Desafio: qual configuração processa mais frames por segundo?

Você já tem um Sobel com memória compartilhada e um pipeline com duas streams. Agora, sua missão é descobrir quanto cada técnica contribui para o desempenho e explicar os resultados.

**Etapa 1 — Construa uma versão de comparação**

Implemente um kernel Sobel que consulte os vizinhos diretamente na memória global, sem usar tile ou `__syncthreads()`.

Preserve a fórmula, o threshold e a política de bordas da versão atual. Assim, você compara a estratégia de acesso à memória mantendo o resultado esperado.

**Etapa 2 — Experimente diferentes blocos**

Compare estas configurações:

| Versão Sobel | Bloco | Objetivo |
|---|---|---|
| Memória global | 16 × 16 | Referência sem tiling |
| Memória global | 32 × 8 | Investigar formato do bloco |
| Memória compartilhada | 16 × 16 | Tile menor |
| Memória compartilhada | 32 × 8 | Tile retangular |
| Memória compartilhada | 32 × 32 | Configuração atual |

Antes de executar, registre sua hipótese: *o menor custo proporcional do halo será suficiente para tornar o bloco 32 × 32 mais rápido?*

**Etapa 3 — Separe cálculo e pipeline**

Faça os testes necessários para responder investigar a questão: *um Sobel mais rápido também torna a aplicação inteira mais rápida?*


**Etapa 4 — Testando outra otimização**

Como você otimizaria ainda mais esse código? Qual técnica de otimização faltou implementar?