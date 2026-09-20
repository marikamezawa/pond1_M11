// test_dsp_stream_host.cpp -- valida que a API streaming (dsp_stream_*)
// produz EXATAMENTE o mesmo resultado que a API batch (dsp_extrair_features)
// para o mesmo sinal, processado em blocos de FRAME_SIZE amostras (como a
// Task 1 entrega de verdade no ESP32).
//
// Le um arquivo int16 (PCM cru, formato usado pelo circular buffer real).
//
// Compilar:
//   g++ -O2 -I../detector_anomalia ../detector_anomalia/dsp.cpp test_dsp_stream_host.cpp -o test_dsp_stream_host -lm
// Rodar:
//   ./test_dsp_stream_host amostras.i16
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "../detector_anomalia/dsp.h"

#define FRAME_SIZE 1024

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "uso: %s amostras.i16\n", argv[0]);
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
    int n = (int)(tamanho_bytes / sizeof(int16_t));

    std::vector<int16_t> amostras_i16(n);
    if (fread(amostras_i16.data(), sizeof(int16_t), n, f) != (size_t)n) {
        fprintf(stderr, "leitura incompleta\n");
        return 1;
    }
    fclose(f);

    // --- batch ---
    std::vector<float> amostras_f32(n);
    for (int i = 0; i < n; i++) amostras_f32[i] = amostras_i16[i] / 32768.0f;
    float rms_batch;
    float feat_batch[DSP_FEATURE_DIM];
    dsp_extrair_features(amostras_f32.data(), n, &rms_batch, feat_batch);

    // --- streaming, em blocos de FRAME_SIZE (como a Task 1 entrega) ---
    dsp_stream_t st;
    dsp_stream_reset(&st);
    int pos = 0;
    while (pos < n) {
        int bloco = (n - pos < FRAME_SIZE) ? (n - pos) : FRAME_SIZE;
        dsp_stream_push(&st, amostras_i16.data() + pos, bloco);
        pos += bloco;
    }
    float rms_stream;
    float feat_stream[DSP_FEATURE_DIM];
    dsp_stream_finalizar(&st, &rms_stream, feat_stream);

    // --- streaming em segmentos de 1s combinados (janela deslizante do firmware) ---
    dsp_stream_t st2;
    dsp_stream_reset(&st2);
    const int SEG = DSP_SAMPLE_RATE;  // 1s
    dsp_segmento_t segs[16];
    int n_segs = 0, na_seg = 0;
    pos = 0;
    while (pos < n) {
        int bloco = (n - pos < FRAME_SIZE) ? (n - pos) : FRAME_SIZE;
        if (bloco > SEG - na_seg) bloco = SEG - na_seg;
        dsp_stream_push(&st2, amostras_i16.data() + pos, bloco);
        pos += bloco;
        na_seg += bloco;
        if (na_seg == SEG && n_segs < 16) {
            dsp_stream_fechar_segmento(&st2, &segs[n_segs++]);
            na_seg = 0;
        }
    }
    if (na_seg > 0 && n_segs < 16) dsp_stream_fechar_segmento(&st2, &segs[n_segs++]);
    float rms_seg;
    float feat_seg[DSP_FEATURE_DIM];
    dsp_segmentos_finalizar(segs, n_segs, &rms_seg, feat_seg);
    float max_diff_seg = fabsf(rms_stream - rms_seg);
    for (int i = 0; i < DSP_FEATURE_DIM; i++) {
        float d = fabsf(feat_stream[i] - feat_seg[i]);
        if (d > max_diff_seg) max_diff_seg = d;
    }

    printf("%-12s %14s %14s %14s\n", "feature", "batch", "streaming", "diferenca");
    printf("%-12s %14.6f %14.6f %14.6f\n", "rms(gate)", rms_batch, rms_stream, fabsf(rms_batch - rms_stream));

    const char *nomes[DSP_FEATURE_DIM] = {"centroid", "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3",
                                            "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8",
                                            "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12"};
    float max_diff = fabsf(rms_batch - rms_stream);
    for (int i = 0; i < DSP_FEATURE_DIM; i++) {
        float diff = fabsf(feat_batch[i] - feat_stream[i]);
        if (diff > max_diff) max_diff = diff;
        printf("%-12s %14.6f %14.6f %14.6f\n", nomes[i], feat_batch[i], feat_stream[i], diff);
    }
    printf("\ndiferenca maxima: %.6f\n", max_diff);
    printf("n_frames streaming: %d\n", st.n_frames);
    printf("diferenca maxima streaming vs segmentos combinados (%d segs): %.6f\n", n_segs, max_diff_seg);
    return 0;
}
