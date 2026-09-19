// dsp.h -- extracao de features (RMS + Spectral Centroid + 13 MFCCs),
// espelhando EXATAMENTE a formula de scripts/dsp_common.py (Python).
// Validado numericamente contra o Python -- ver tools/test_dsp_host.cpp.
//
// Nao usa nenhuma biblioteca de DSP externa (nem esp-dsp): FFT radix-2
// propria, mel filterbank e DCT calculados aqui mesmo. Codigo C++
// portavel (sem chamadas especificas do Arduino), compila tanto no
// ESP32 (Arduino IDE) quanto no host (g++) para testes.
#pragma once

#define DSP_SAMPLE_RATE   16000
#define DSP_N_FFT         512
#define DSP_HOP           160     // 10ms @ 16kHz
#define DSP_N_MELS        26
#define DSP_N_MFCC        13
#define DSP_FEATURE_DIM   (2 + DSP_N_MFCC)   // rms, centroid, mfcc_0..12

#define DSP_WINDOW_SEC     3.0f
#define DSP_WINDOW_SAMPLES (int)(DSP_WINDOW_SEC * DSP_SAMPLE_RATE)  // 48000

// Extrai [rms, spectral_centroid, mfcc_0..mfcc_12] de um buffer de audio
// mono, normalizado em [-1, 1].
//
// ATENCAO: esta funcao aplica pre-enfase IN-PLACE no buffer `amostras`
// (economiza memoria -- nao aloca uma copia do buffer inteiro). Se o
// chamador precisar do sinal original depois, deve copiar antes.
//
// n_amostras deve ser >= DSP_N_FFT. saida_features deve ter DSP_FEATURE_DIM
// posicoes.
void dsp_extrair_features(float *amostras, int n_amostras, float *saida_features);
