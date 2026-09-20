// feature_task.cpp -- ver feature_task.h
#include "feature_task.h"
#include "audio_capture.h"
#include "sync.h"
#include "config.h"
#include "dsp.h"
#include <Arduino.h>
#include <string.h>

// Estado do acumulador streaming (dsp_stream_t) -- ~8KB estaticos, bem
// diferente de guardar a janela de 3s inteira (48000 floats = 187KB, que
// estourava a DRAM do ESP32). Ver dsp.h para detalhes da API streaming.
static dsp_stream_t s_stream;

// Janela deslizante: o DSP roda UMA vez por frame; a cada segmento de 1s
// guardamos so as somas parciais num anel de SEGMENTOS_POR_JANELA segmentos.
// A janela de 3s = ultimos 3 segmentos, reavaliada a cada segmento fechado.
static dsp_segmento_t s_segmentos[SEGMENTOS_POR_JANELA];   // anel
static uint32_t s_seg_inicio_ms[SEGMENTOS_POR_JANELA];     // timestamp de inicio de cada segmento
static uint32_t s_seg_dsp_us[SEGMENTOS_POR_JANELA];        // CPU do DSP em cada segmento
static int s_seg_escrita = 0;      // proxima posicao do anel
static int s_seg_validos = 0;      // segmentos ja preenchidos (satura em SEGMENTOS_POR_JANELA)
static int s_amostras_no_seg = 0;
static uint32_t s_seg_atual_inicio_ms = 0;
static uint32_t s_seg_atual_dsp_us = 0;
static uint32_t s_dsp_max_frame_us = 0;   // pior caso de um push (deadline: 64ms)

// Task 2 (prioridade media): fica bloqueada no semaforo ate a Task 1
// sinalizar um novo frame; copia o frame sob mutex; alimenta o acumulador
// streaming; a cada segmento de 1s fecha as somas parciais e, com 3 segmentos
// disponiveis, envia um FeatureVector (centroid + 13 MFCCs normalizados por
// energia, mais o RMS bruto da janela como gatilho de silencio) para a Task 3.
void task_features(void *pvParameters) {
    dsp_stream_reset(&s_stream);

    while (1) {
        xSemaphoreTake(xSemSlotCheio, portMAX_DELAY);

        int16_t frame_local[FRAME_SIZE];
        uint32_t frame_timestamp_ms;

        xSemaphoreTake(xMutexBuffer, portMAX_DELAY);
        int slot = g_ultimo_slot_pronto;
        memcpy(frame_local, g_circular_buffer[slot], sizeof(frame_local));
        frame_timestamp_ms = g_ultimo_frame_timestamp_ms;
        xSemaphoreGive(xMutexBuffer);

        if (s_amostras_no_seg == 0) {
            s_seg_atual_inicio_ms = frame_timestamp_ms;
        }

        int restante = SEGMENTO_SAMPLES - s_amostras_no_seg;
        int n_processar = (restante < FRAME_SIZE) ? restante : FRAME_SIZE;

        uint32_t t_push = micros();
        dsp_stream_push(&s_stream, frame_local, n_processar);
        uint32_t dt_push = micros() - t_push;
        s_seg_atual_dsp_us += dt_push;
        if (dt_push > s_dsp_max_frame_us) s_dsp_max_frame_us = dt_push;
        s_amostras_no_seg += n_processar;

        if (s_amostras_no_seg >= SEGMENTO_SAMPLES) {
            dsp_stream_fechar_segmento(&s_stream, &s_segmentos[s_seg_escrita]);
            s_seg_inicio_ms[s_seg_escrita] = s_seg_atual_inicio_ms;
            s_seg_dsp_us[s_seg_escrita] = s_seg_atual_dsp_us;
            s_seg_escrita = (s_seg_escrita + 1) % SEGMENTOS_POR_JANELA;
            if (s_seg_validos < SEGMENTOS_POR_JANELA) s_seg_validos++;
            s_amostras_no_seg = 0;
            s_seg_atual_dsp_us = 0;

            if (s_seg_validos == SEGMENTOS_POR_JANELA) {
                // s_seg_escrita agora aponta para o segmento MAIS ANTIGO do anel.
                uint32_t t_inicio = micros();
                float rms_janela;
                float features_svm[DSP_FEATURE_DIM];  // [centroid, mfcc_0..12]
                dsp_segmentos_finalizar(s_segmentos, SEGMENTOS_POR_JANELA, &rms_janela, features_svm);
                uint32_t latencia_us = micros() - t_inicio;
                for (int i = 0; i < SEGMENTOS_POR_JANELA; i++) latencia_us += s_seg_dsp_us[i];

                FeatureVector fv;
                fv.rms = rms_janela;  // so gatilho de silencio, nao entra no SVM
                fv.spectral_centroid = features_svm[0];
                memcpy(fv.mfccs, &features_svm[1], sizeof(fv.mfccs));
                fv.timestamp_captura_ms = s_seg_inicio_ms[s_seg_escrita];
                fv.latencia_features_us = latencia_us;
                fv.latencia_frame_max_us = s_dsp_max_frame_us;

                // Fila com timeout 0: nao bloqueia -- se a Task 3 estiver
                // atrasada e a fila cheia, descarta a janela mais antiga em
                // vez de acumular atraso (audio em tempo real: um FeatureVector
                // velho e menos util que continuar processando o presente).
                if (xQueueSend(xFilaFeatures, &fv, 0) != pdTRUE) {
                    FeatureVector descartado;
                    xQueueReceive(xFilaFeatures, &descartado, 0);
                    xQueueSend(xFilaFeatures, &fv, 0);
                }
                s_dsp_max_frame_us = 0;
            }

            // O frame de FRAME_SIZE amostras nao alinha com o segmento de 1s:
            // a sobra vai para o inicio do proximo segmento (nao descarta audio).
            int sobra = FRAME_SIZE - n_processar;
            if (sobra > 0) {
                s_seg_atual_inicio_ms = frame_timestamp_ms;
                uint32_t t_sobra = micros();
                dsp_stream_push(&s_stream, frame_local + n_processar, sobra);
                uint32_t dt_sobra = micros() - t_sobra;
                s_seg_atual_dsp_us += dt_sobra;
                if (dt_sobra > s_dsp_max_frame_us) s_dsp_max_frame_us = dt_sobra;
                s_amostras_no_seg = sobra;
            }
        }
    }
}
