// test_svm_host.cpp -- valida svm_infer.cpp contra sklearn predict_proba.
// Le vetores de features de um CSV (uma linha por amostra, 15 valores
// separados por virgula) e imprime P(masculino) para cada um.
//
// Compilar:
//   g++ -O2 -I.. ../svm_infer.cpp test_svm_host.cpp -o test_svm_host -lm
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../detector_anomalia/svm_infer.h"
#include "../detector_anomalia/svm_params.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "uso: %s features.csv\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "erro ao abrir %s\n", argv[1]);
        return 1;
    }

    char linha[4096];
    while (fgets(linha, sizeof(linha), f)) {
        float feat[SVM_N_FEATURES];
        char *tok = strtok(linha, ",");
        for (int i = 0; i < SVM_N_FEATURES && tok; i++) {
            feat[i] = atof(tok);
            tok = strtok(NULL, ",");
        }
        float prob = svm_prob_masculino(feat);
        printf("%.6f\n", prob);
    }
    fclose(f);
    return 0;
}
