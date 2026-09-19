// feature_task.cpp -- ver feature_task.h
#include "feature_task.h"
#include "audio_capture.h"
#include "sync.h"
#include "config.h"
#include "dsp.h"
#include <Arduino.h>
#include <string.h>

// Buffer de acumulacao da janela de analise (3s = 48000 amostras float).
// Estatico (nao na stack) -- 48000*4 bytes = 187.5KB, cabe nos ~320KB de
// heap/BSS tipicamente disponiveis no ESP32-WROOM-32U.
static float s_janela[DSP_WINDOW_SAMPLES];
static int s_janela_count = 0;
static uint32_t s_janela_inicio_ms = 0;

// Task 2 (prioridade media): fica bloqueada no semaforo ate a Task 1
// sinalizar um novo frame; copia o frame sob mutex; acumula na janela de
// analise; quando a janela completa DSP_WINDOW_SAMPLES, calcula as
// features (RMS + centroid + 13 MFCCs) e envia pela fila para a Task 3.
void task_features(void *pvParameters) {
    while (1) {
        xSemaphoreTake(xSemSlotCheio, portMAX_DELAY);

        int16_t frame_local[FRAME_SIZE];
        uint32_t frame_timestamp_ms;

        xSemaphoreTake(xMutexBuffer, portMAX_DELAY);
        int slot = g_ultimo_slot_pronto;
        memcpy(frame_local, g_circular_buffer[slot], sizeof(frame_local));
        frame_timestamp_ms = g_ultimo_frame_timestamp_ms;
        xSemaphoreGive(xMutexBuffer);

        if (s_janela_count == 0) {
            s_janela_inicio_ms = frame_timestamp_ms;
        }

        int restante = DSP_WINDOW_SAMPLES - s_janela_count;
        int n_copiar = (restante < FRAME_SIZE) ? restante : FRAME_SIZE;
        for (int i = 0; i < n_copiar; i++) {
            s_janela[s_janela_count + i] = frame_local[i] / 32768.0f;
        }
        s_janela_count += n_copiar;

        if (s_janela_count >= DSP_WINDOW_SAMPLES) {
            uint32_t t_inicio = micros();
            float features[DSP_FEATURE_DIM];
            dsp_extrair_features(s_janela, DSP_WINDOW_SAMPLES, features);
            uint32_t latencia_us = micros() - t_inicio;

            FeatureVector fv;
            fv.rms = features[0];
            fv.spectral_centroid = features[1];
            memcpy(fv.mfccs, &features[2], sizeof(fv.mfccs));
            fv.timestamp_captura_ms = s_janela_inicio_ms;
            fv.latencia_features_us = latencia_us;

            // Fila com timeout 0: nao bloqueia -- se a Task 3 estiver
            // atrasada e a fila cheia, descarta a janela mais antiga em
            // vez de acumular atraso (audio em tempo real: um FeatureVector
            // velho e menos util que continuar processando o presente).
            if (xQueueSend(xFilaFeatures, &fv, 0) != pdTRUE) {
                FeatureVector descartado;
                xQueueReceive(xFilaFeatures, &descartado, 0);
                xQueueSend(xFilaFeatures, &fv, 0);
            }

            s_janela_count = 0;  // proxima janela, sem overlap
        }
    }
}
