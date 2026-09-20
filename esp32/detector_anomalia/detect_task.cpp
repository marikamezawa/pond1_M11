// detect_task.cpp -- ver detect_task.h
#include "detect_task.h"
#include "sync.h"
#include "config.h"
#include "svm_infer.h"
#include "led_control.h"
#include <Arduino.h>
#include <string.h>

// Task 3 (prioridade baixa): bloqueada em xQueueReceive ate a Task 2
// enviar um FeatureVector. Gate de silencio evita rodar o SVM a toa.
// Mede latencia total (do inicio da captura da janela ate o LED acender)
// e loga tudo via Serial para a analise de performance do relatorio.
void task_deteccao(void *pvParameters) {
    led_control_init();

    FeatureVector fv;
    int janelas_masc_seguidas = 0;
    while (1) {
        xQueueReceive(xFilaFeatures, &fv, portMAX_DELAY);
        uint32_t agora_ms = millis();

        bool tem_voz = fv.rms >= THRESHOLD_SILENCIO;
        bool eh_anomalia = false;
        float prob_masculino = 0.0f;
        uint32_t latencia_inferencia_us = 0;

        if (tem_voz) {
            // DSP_FEATURE_DIM = [centroid, mfcc_0..12] -- fv.rms NAO entra
            // aqui, so serviu pro gate de silencio acima (ver dsp.h).
            float features[DSP_FEATURE_DIM];
            features[0] = fv.spectral_centroid;
            memcpy(&features[1], fv.mfccs, sizeof(fv.mfccs));

            uint32_t t_inicio = micros();
            prob_masculino = svm_prob_masculino(features);
            latencia_inferencia_us = micros() - t_inicio;

            // Debounce: erro isolado do modelo nao aciona o alarme; exige
            // ANOMALIA_JANELAS_SEGUIDAS janelas consecutivas acima do threshold.
            if (prob_masculino > THRESHOLD_ANOMALIA) janelas_masc_seguidas++;
            else janelas_masc_seguidas = 0;
            eh_anomalia = janelas_masc_seguidas >= ANOMALIA_JANELAS_SEGUIDAS;
        } else {
            janelas_masc_seguidas = 0;
        }

        bool pendente = tem_voz && !eh_anomalia && janelas_masc_seguidas > 0;
        led_estado_t led = pendente ? led_control_incerto(agora_ms)
                                    : led_control_atualizar(tem_voz, eh_anomalia, agora_ms);

        uint32_t latencia_total_ms = agora_ms - fv.timestamp_captura_ms;

        if (tem_voz) {
            const char *estado_str = eh_anomalia ? "ANOMALIA" : (pendente ? "incerto" : "normal");
            Serial.printf(
                "[deteccao] %s prob_masc=%.2f seguidas=%d rms=%.4f | lat_features=%lu us "
                "(max_frame=%lu us) | lat_inferencia=%lu us | lat_total=%lu ms | led=%d\n",
                estado_str, prob_masculino, janelas_masc_seguidas, fv.rms,
                (unsigned long)fv.latencia_features_us,
                (unsigned long)fv.latencia_frame_max_us,
                (unsigned long)latencia_inferencia_us,
                (unsigned long)latencia_total_ms, (int)led);
        } else {
            Serial.printf("[deteccao] silencio rms=%.4f | led=%d\n", fv.rms, (int)led);
        }
    }
}
