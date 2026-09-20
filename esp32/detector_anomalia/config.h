// config.h -- pinagem, constantes e tipos compartilhados entre as 3 tasks.
// Pinagem confirmada na montagem fisica real (ver README.md).
#pragma once
#include <stdint.h>
#include "dsp.h"

// ---- Pinagem ----
#define I2S_SCK_PIN       26
#define I2S_WS_PIN        25
#define I2S_SD_PIN        27
#define LED_VERDE_PIN     33   // GPIO2 tinha problema de fiacao na montagem real -- trocado
#define LED_VERMELHO_PIN  32

// ---- Captura de audio (Task 1) ----
#define FRAME_SIZE     1024   // amostras por leitura I2S (64ms @ 16kHz)
#define BUFFER_SLOTS   4      // slots do buffer circular

// ---- Janela deslizante (Task 2) ----
// Janela de analise = SEGMENTOS_POR_JANELA * 1s (3s de contexto), reavaliada a
// cada segmento de 1s -- em vez de esperar 3s inteiros entre decisoes.
#define SEGMENTO_SAMPLES       DSP_SAMPLE_RATE   // 1s
#define SEGMENTOS_POR_JANELA   3

// ---- Deteccao (Task 3) ----
#define THRESHOLD_ANOMALIA   0.75f
// Janelas de 3s consecutivas acima do threshold antes de acionar o alarme
// (reduz falso alarme; custo: ~+1s de latencia ate o LED vermelho).
#define ANOMALIA_JANELAS_SEGUIDAS 2   // janelas de 3s, avaliadas a cada 1s
// Calibrado com audio real do INMP441 (data/hardware): silencio ~0.005 e fala
// em janela de 3s entre ~0.015 e ~0.08 de RMS. 0.05 tratava fala como silencio.
#define THRESHOLD_SILENCIO   0.02f
#define LED_HOLD_MS          1500    // retencao do LED apos ultima deteccao de voz

// Vetor de features transferido pela fila (Task 2 -> Task 3)
typedef struct {
    float rms;
    float spectral_centroid;
    float mfccs[DSP_N_MFCC];
    uint32_t timestamp_captura_ms;   // quando a JANELA comecou a ser capturada (Task 1)
    uint32_t latencia_features_us;   // CPU total do DSP na janela (soma de todos os push + finalizar)
    uint32_t latencia_frame_max_us;  // pior caso de processar 1 frame de 1024 amostras
} FeatureVector;
