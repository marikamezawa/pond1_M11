// dsp.h -- extracao de features (Spectral Centroid + 13 MFCCs) para o
// classificador, espelhando EXATAMENTE scripts/dsp_common.py (Python).
// Validado numericamente contra o Python -- ver esp32/host_test/.
//
// O RMS bruto NAO alimenta mais o classificador (ver docstring de
// dsp_common.py: variava demais entre hardwares de captura e enviesava a
// decisao). Cada frame de DSP_N_FFT amostras e normalizado individualmente
// para uma energia alvo antes do MFCC -- torna a forma espectral (timbre)
// independente do volume absoluto de captura. O RMS da janela inteira
// (sinal cru, sem pre-enfase) continua sendo calculado -- devolvido
// separadamente, usado so como gatilho de silencio (VAD).
//
// Nao usa nenhuma biblioteca de DSP externa (nem esp-dsp): FFT radix-2
// propria, mel filterbank e DCT calculados aqui mesmo. Codigo C++
// portavel (sem chamadas especificas do Arduino), compila tanto no
// ESP32 (Arduino IDE) quanto no host (g++) para testes.
#pragma once
#include <stdint.h>

#define DSP_SAMPLE_RATE   16000
#define DSP_N_FFT         512
#define DSP_HOP           160     // 10ms @ 16kHz
#define DSP_N_MELS        26
#define DSP_N_MFCC        13
#define DSP_FEATURE_DIM   (1 + DSP_N_MFCC)   // centroid, mfcc_0..12 (SEM rms)

#define DSP_WINDOW_SEC     3.0f
#define DSP_WINDOW_SAMPLES (int)(DSP_WINDOW_SEC * DSP_SAMPLE_RATE)  // 48000

// Normalizacao de energia por frame (ver dsp_common.py para a justificativa).
#define DSP_TARGET_FRAME_RMS  0.1f
#define DSP_MIN_FRAME_RMS     0.003f
#define DSP_MAX_SCALE         15.0f

// ---- API "batch" (buffer inteiro em memoria) ----
// Usada pelos testes de validacao numerica (esp32/host_test) contra
// scripts/dsp_common.py. NAO e o que a Task 2 do ESP32 usa de fato (ver
// API streaming abaixo) -- guardar uma janela de 3s inteira (48000 floats
// = 187KB) estoura a DRAM disponivel no ESP32.
//
// amostras: audio mono normalizado em [-1,1]. ATENCAO: aplica pre-enfase
// IN-PLACE no buffer (economiza memoria). *rms_janela recebe o RMS do
// sinal CRU (antes da pre-enfase) -- so para gatilho de silencio, nao e
// uma das SVM_N_FEATURES. saida_features_svm recebe [centroid,
// mfcc_0..12] (DSP_FEATURE_DIM = 14 valores). n_amostras deve ser >=
// DSP_N_FFT.
void dsp_extrair_features(float *amostras, int n_amostras, float *rms_janela, float *saida_features_svm);

// ---- API streaming (usada de fato na Task 2 do ESP32) ----
// Processa a janela de analise em pedacos pequenos -- os blocos de
// FRAME_SIZE amostras que chegam do buffer circular da Task 1 -- sem
// nunca guardar a janela inteira na RAM. So mantem um buffer deslizante
// pequeno (DSP_STREAM_BUF_CAP amostras) e acumuladores escalares.
// Matematicamente identico a dsp_extrair_features (mesmo helper interno
// de processamento de frame) -- validado numericamente contra ela em
// esp32/host_test/test_dsp_stream_host.cpp.
#define DSP_STREAM_BUF_CAP 2048

typedef struct {
    float buf[DSP_STREAM_BUF_CAP];
    int buf_count;
    float preemph_anterior;
    int tem_anterior;          // bool (0/1)
    double soma_sq_bruto;      // acumula amostras CRUAS (sem pre-enfase) p/ RMS de gatilho
    long n_amostras_total;
    double centroid_sum;
    float mfcc_sum[DSP_N_MFCC];
    int n_frames;              // frames que passaram do piso de energia (contribuiram)
} dsp_stream_t;

void dsp_stream_reset(dsp_stream_t *st);

// amostras: PCM cru (int16), NAO normalizado -- a normalizacao para
// [-1,1] e feita internamente. n deve ser <= DSP_STREAM_BUF_CAP - DSP_N_FFT
// (folga generosa para blocos de ate 1024 amostras, como os da Task 1).
void dsp_stream_push(dsp_stream_t *st, const int16_t *amostras, int n);

// Calcula o RMS da janela (*rms_janela, gatilho de silencio) e as features
// finais do classificador (saida_features_svm, DSP_FEATURE_DIM valores).
// Nao reseta o estado -- chame dsp_stream_reset() antes da proxima janela.
void dsp_stream_finalizar(const dsp_stream_t *st, float *rms_janela, float *saida_features_svm);

// ---- Janela deslizante por segmentos ----
// Para avaliar uma janela de DSP_WINDOW_SEC a cada SEGMENTO (ex: 3s de
// contexto atualizados a cada 1s) sem refazer o DSP: o stream roda uma vez
// por frame, e a cada segmento fechado guardamos so as somas parciais. A
// janela = combinacao dos ultimos N segmentos. A media de features sobre os
// frames validos e a mesma de uma janela contigua (so soma/contagem).
typedef struct {
    double soma_sq_bruto;
    long n_amostras;
    double centroid_sum;
    float mfcc_sum[DSP_N_MFCC];
    int n_frames;
} dsp_segmento_t;

// Copia os acumuladores atuais para *seg e os zera (mantem o buffer
// deslizante e a pre-enfase, para nao criar descontinuidade no proximo segmento).
void dsp_stream_fechar_segmento(dsp_stream_t *st, dsp_segmento_t *seg);

// Combina n_segs segmentos em RMS + features finais (mesmo formato de dsp_stream_finalizar).
void dsp_segmentos_finalizar(const dsp_segmento_t *segs, int n_segs, float *rms_janela, float *saida_features_svm);
