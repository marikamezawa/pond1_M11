// test_dsp_host.cpp -- harness de validacao numerica: compila e roda no
// computador (nao no ESP32) para comparar dsp.cpp contra dsp_common.py.
//
// Uso: le um arquivo binario de floats32 (amostras mono, [-1,1]) e imprime
// as 15 features extraidas, uma por linha.
//
// Compilar:
//   g++ -O2 -I.. ../dsp.cpp test_dsp_host.cpp -o test_dsp_host -lm
// Rodar:
//   ./test_dsp_host caminho/para/amostras.f32
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../detector_anomalia/dsp.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "uso: %s arquivo.f32\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "erro ao abrir %s\n", argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long tamanho_bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(tamanho_bytes / sizeof(float));

    std::vector<float> amostras(n);
    fread(amostras.data(), sizeof(float), n, f);
    fclose(f);

    float features[DSP_FEATURE_DIM];
    dsp_extrair_features(amostras.data(), n, features);

    for (int i = 0; i < DSP_FEATURE_DIM; i++) {
        printf("%.8f\n", features[i]);
    }
    return 0;
}
