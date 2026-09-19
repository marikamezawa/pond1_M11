// config.h -- pinagem, constantes e tipos compartilhados entre as 3 tasks.
// Pinagem confirmada na montagem fisica real (ver planejamento.md).
#pragma once
#include <stdint.h>
#include "dsp.h"

// ---- Pinagem ----
#define I2S_SCK_PIN       26
#define I2S_WS_PIN        25
#define I2S_SD_PIN        27
#define LED_VERDE_PIN     2
#define LED_VERMELHO_PIN  32

// ---- Captura de audio (Task 1) ----
#define FRAME_SIZE     1024   // amostras por leitura I2S (64ms @ 16kHz)
#define BUFFER_SLOTS   4      // slots do buffer circular

// ---- Deteccao (Task 3) ----
#define THRESHOLD_ANOMALIA   0.65f
#define THRESHOLD_SILENCIO   0.01f   // mesmo criterio usado no treino (SILENCIO_RMS)
#define LED_HOLD_MS          1500    // retencao do LED apos ultima deteccao de voz

// Vetor de features transferido pela fila (Task 2 -> Task 3)
typedef struct {
    float rms;
    float spectral_centroid;
    float mfccs[DSP_N_MFCC];
    uint32_t timestamp_captura_ms;   // quando a JANELA comecou a ser capturada (Task 1)
    uint32_t latencia_features_us;   // tempo gasto no calculo das features (Task 2)
} FeatureVector;
