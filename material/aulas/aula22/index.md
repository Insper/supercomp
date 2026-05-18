# Simulado para a Avaliação Final de Supercomp

O Dataset necessário está disponível na pasta `/tmp/frames` do Cluster Franky
Faça uma cópia para sua pasta de trabalho com o comando:

```bash
cp /tmp/frames/* ~/scratch/seu-diretorio-de-trabalho/frames/
```

Código base para realização dos Exercícios:

??? note "Ver o código"

        #define STB_IMAGE_IMPLEMENTATION
        #include "stb_image.h"
        #define STB_IMAGE_WRITE_IMPLEMENTATION
        #include "stb_image_write.h"

        #include <iostream>
        #include <vector>
        #include <queue>
        #include <cmath>
        #include <cstdio>
        #include <filesystem>
        #include <chrono>
        #include <algorithm>

        using namespace std;
        namespace fs = std::filesystem;
        using namespace std::chrono;

        // ==========================
        // ESTRUTURA
        // ==========================

        struct Box {
            int minx, miny, maxx, maxy;
        };

        // ==========================
        // RGB → GRAY
        // ==========================

        void rgb2gray(unsigned char* input, unsigned char* gray, int w, int h) {
            for (int i = 0; i < w * h; i++) {
                int idx = i * 3;

                float r = input[idx];
                float g = input[idx + 1];
                float b = input[idx + 2];

                gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            }
        }

        // ==========================
        // SOBEL
        // ==========================

        void sobel(unsigned char* gray, unsigned char* out, int w, int h) {

            int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
            int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

            for (int y = 1; y < h - 1; y++) {
                for (int x = 1; x < w - 1; x++) {

                    int sumX = 0;
                    int sumY = 0;

                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {

                            int pixel = gray[(y + ky) * w + (x + kx)];

                            sumX += pixel * Gx[ky + 1][kx + 1];
                            sumY += pixel * Gy[ky + 1][kx + 1];
                        }
                    }

                    int mag = (int)sqrt(sumX * sumX + sumY * sumY);
                    if (mag > 255) mag = 255;

                    out[y * w + x] = (unsigned char)mag;
                }
            }
        }

        // ==========================
        // THRESHOLD
        // ==========================

        void threshold_bin(unsigned char* in, unsigned char* bin, int w, int h, int T) {
            for (int i = 0; i < w * h; i++) {
                bin[i] = (in[i] > T) ? 255 : 0;
            }
        }


        // ==========================
        // MAIN
        // ==========================

        int main(int argc, char* argv[]) {

            int max_frames = -1;

            if (argc > 1) {
                max_frames = atoi(argv[1]);
                cout << "Modo teste: " << max_frames << " frames\n";
            }

            fs::create_directory("out");

            auto t0 = high_resolution_clock::now();

            int frame = 1;

            while (true) {

                if (max_frames != -1 && frame > max_frames) break;

                char filename[256];
                sprintf(filename, "frames/img%07d.jpg", frame);

                int w, h, c;

                unsigned char* input = stbi_load(filename, &w, &h, &c, 3);

                if (!input) {
                    cout << "\nFim ou erro: " << filename << endl;
                    break;
                }

                unsigned char* gray = new unsigned char[w*h];
                unsigned char* sob  = new unsigned char[w*h];
                unsigned char* bin  = new unsigned char[w*h];

                rgb2gray(input, gray, w, h);
                sobel(gray, sob, w, h);
                threshold_bin(sob, bin, w, h, 100);

                char outname[256];
                sprintf(outname, "out/img%07d.jpg", frame);

                stbi_write_jpg(outname, w, h, 1, bin, 100);

                cout << "\rProcessando: " << frame << flush;

                delete[] gray;
                delete[] sob;
                delete[] bin;
                stbi_image_free(input);

                frame++;
            }

            auto t1 = high_resolution_clock::now();
            double total_time = duration<double>(t1 - t0).count();

            cout << "\n\n===== FINAL =====\n";
            cout << "Frames processados: " << frame - 1 << endl;
            cout << "Tempo total: " << total_time << " s\n";

            return 0;
        }


### Exercício 1 — Paralelização do filtro Sobel em CUDA

Um sistema de monitoramento urbano precisa detectar bordas em imagens aéreas de trânsito utilizando o filtro Sobel. Atualmente, o processamento ocorre de forma sequencial na CPU, tornando o pipeline lento para grandes quantidades de imagens.

**Passe as partes computacionalmente complexas do código para GPU.**


| Rúbrica               | Peso    |
| --------------------- | ------- |
| O kernel CUDA foi corretamente implementado, o gerenciamento dos dados entre CPU e GPU foi realizado de forma adequada   | 1.5 |     
| Arquivo `run.slurm` configurado corretamente para execução no Cluster Franky, incluindo solicitação adequada de recursos e carregamento dos módulos necessários. | 0.2 |
| Arquivo `run.slurm` configurado corretamente para execução no Cluster Santos Dumont, incluindo solicitação adequada de recursos e carregamento dos módulos necessários. | 0.3 |
| **Total**             | **2.0** |



### Exercício 2 — Sobel com Shared Memory e Tiling


Após paralelizar o Sobel, percebeu-se que o kernel ainda realiza muitos acessos repetidos à memória global, principalmente porque pixels vizinhos são reutilizados várias vezes durante a convolução.

O objetivo agora é reduzir acessos à memória global utilizando memória compartilhada.

**Otimize o Sobel aplicando as técnicas de tiling; shared memory e halo.**


| Rúbrica                       | Peso    |
| ----------------------------- | ------- |
| A versão otimizada com Tilling em GPU apresenta um desempenho melhor do que a versão base sequencial em CPU | 3.0 |
| Arquivo `run.slurm` configurado corretamente para execução no Cluster Franky, incluindo solicitação adequada de recursos e carregamento dos módulos necessários. | 0.2 |
| Arquivo `run.slurm` configurado corretamente para execução no Cluster Santos Dumont, incluindo solicitação adequada de recursos e carregamento dos módulos necessários. | 0.3 |
| **Total**                     | **3.5** |

### Exercício 3 — Processamento Assíncrono com CUDA Streams

Mesmo com as otimziações a GPU ainda fica ociosa enquanto:

* novas imagens são carregadas;
* dados são copiados;
* arquivos são salvos.


Implemente uma pipeline que executa o código de forma assíncrona sobrepondo as etapas de uso de CPU e GPU.


| Rúbrica                          | Peso    |
| -------------------------------- | ------- |
| Pipeline assíncrono corretamente implementado e funcionando  | 3.5     |
| Comparação de desempenho e análise dos resultados obtidos | 1.0  |
| Arquivo `run.slurm` configurado corretamente para execução no Cluster Franky, incluindo solicitação adequada de recursos e carregamento dos módulos necessários. | 0.2 |
| Arquivo `run.slurm` configurado corretamente para execução no Cluster Santos Dumont, incluindo solicitação adequada de recursos e carregamento dos módulos necessários. | 0.3 |
| **Total**                        | **4.5** |
