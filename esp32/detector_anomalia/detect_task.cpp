// detect_task.cpp -- ver detect_task.h
#include "detect_task.h"
#include "sync.h"
#include "config.h"
#include "svm_infer.h"
#include "led_control.h"
#include <Arduino.h>
#include <string.h>

// Episodio de fala: comeca na 1a voz confirmada depois de silencio e termina
// quando a janela cai abaixo do gate de silencio. O LED pisca (2 piscadas) UMA vez
// por episodio (ao confirmar a classe da voz); so volta a piscar se o ambiente
// ficar em silencio e uma voz surgir de novo. Excecao: voz masculina que
// aparece no meio de um episodio ja "normal" pisca o vermelho (senao o
// alarme seria perdido). Depois do vermelho nao pisca verde; se a voz voltar a
// ser feminina de forma confirmada, o vermelho e rearmado (sem piscar).
typedef enum { EP_NENHUM, EP_NORMAL, EP_ANOMALIA } episodio_t;

// Task 3 (prioridade baixa): bloqueada em xQueueReceive ate a Task 2
// enviar um FeatureVector. Gate de silencio evita rodar o SVM a toa.
// Mede latencia total (do inicio da captura da janela ate a decisao)
// e loga tudo via Serial para a analise de performance do relatorio.
void task_deteccao(void *pvParameters) {
    led_control_init();

    FeatureVector fv;
    episodio_t episodio = EP_NENHUM;
    int masc_seguidas = 0;    // janelas seguidas com P(masc) > THRESHOLD_ANOMALIA
    int normal_seguidas = 0;  // janelas seguidas com P(masc) <= THRESHOLD_ANOMALIA

    // Historico de RMS das ultimas janelas -> piso de ruido de fundo (gate adaptativo).
    float historico_rms[PISO_RUIDO_JANELAS];
    int hist_n = 0, hist_i = 0;

    while (1) {
        xQueueReceive(xFilaFeatures, &fv, portMAX_DELAY);
        uint32_t agora_ms = millis();
        uint32_t latencia_total_ms = agora_ms - fv.timestamp_captura_ms;

        // Piso de ruido = menor RMS recente; gate = max(minimo fixo, fator * piso).
        historico_rms[hist_i] = fv.rms;
        hist_i = (hist_i + 1) % PISO_RUIDO_JANELAS;
        if (hist_n < PISO_RUIDO_JANELAS) hist_n++;
        float piso = historico_rms[0];
        for (int i = 1; i < hist_n; i++) if (historico_rms[i] < piso) piso = historico_rms[i];
        if (piso > PISO_RUIDO_MAX) piso = PISO_RUIDO_MAX;
        float gate = FATOR_PISO_RUIDO * piso;
        if (gate < THRESHOLD_SILENCIO) gate = THRESHOLD_SILENCIO;

        bool tem_voz = fv.rms >= gate;
        if (!tem_voz) {
            masc_seguidas = 0;
            normal_seguidas = 0;
            episodio = EP_NENHUM;   // silencio: a proxima voz e um novo episodio
            Serial.printf("[deteccao] silencio rms=%.4f gate=%.4f | led=0\n", fv.rms, gate);
            continue;
        }

        // DSP_FEATURE_DIM = [centroid, mfcc_0..12] -- fv.rms NAO entra
        // aqui, so serviu pro gate de silencio acima (ver dsp.h).
        float features[DSP_FEATURE_DIM];
        features[0] = fv.spectral_centroid;
        memcpy(&features[1], fv.mfccs, sizeof(fv.mfccs));

        uint32_t t_inicio = micros();
        float prob_masculino = svm_prob_masculino(features);
        uint32_t latencia_inferencia_us = micros() - t_inicio;

        // Debounce simetrico: a classe so vale apos varias janelas seguidas iguais.
        if (prob_masculino > THRESHOLD_ANOMALIA) { masc_seguidas++; normal_seguidas = 0; }
        else                                     { normal_seguidas++; masc_seguidas = 0; }

        led_estado_t pisca = LED_ESTADO_APAGADO;
        if (masc_seguidas >= ANOMALIA_JANELAS_SEGUIDAS && episodio != EP_ANOMALIA) {
            pisca = LED_ESTADO_VERMELHO;
            episodio = EP_ANOMALIA;
        } else if (normal_seguidas >= NORMAL_JANELAS_SEGUIDAS) {
            if (episodio == EP_NENHUM) {
                pisca = LED_ESTADO_VERDE;
                episodio = EP_NORMAL;
            } else if (episodio == EP_ANOMALIA) {
                // Voz voltou a ser feminina de forma confirmada: rearma o vermelho
                // (uma nova voz masculina volta a piscar) sem piscar o verde.
                episodio = EP_NORMAL;
            }
        }

        const char *estado_str = (masc_seguidas >= ANOMALIA_JANELAS_SEGUIDAS) ? "ANOMALIA"
                                 : (masc_seguidas > 0)                        ? "incerto"
                                                                              : "normal";
        Serial.printf(
            "[deteccao] %s prob_masc=%.2f seguidas=%d rms=%.4f gate=%.4f | lat_features=%lu us "
            "(max_frame=%lu us) | lat_inferencia=%lu us | lat_total=%lu ms | led=%d\n",
            estado_str, prob_masculino, masc_seguidas, fv.rms, gate,
            (unsigned long)fv.latencia_features_us,
            (unsigned long)fv.latencia_frame_max_us,
            (unsigned long)latencia_inferencia_us,
            (unsigned long)latencia_total_ms, (int)pisca);

        led_control_piscar(pisca);   // so pisca em eventos; bloqueia durante o pisca
    }
}
