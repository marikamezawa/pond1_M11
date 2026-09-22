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
// Janelas seguidas para confirmar voz feminina (3, mais que o vermelho: evita
// piscar o verde no inicio de uma fala masculina, cujas primeiras janelas
// parciais costumam cair como "normal").
#define NORMAL_JANELAS_SEGUIDAS   3
// Nivel minimo de RMS (janela de 3s) para considerar "voz": abaixo disso e silencio.
// Silencio real do INMP441 ~0.005; fala ao vivo perto do mic ~0.03-0.3. Quanto maior o
// valor, mais alto/perto e preciso falar (0.02 aceitava ruido de sala como voz; 0.05
// cortava fala normal). 0.03 exige falar um pouco mais alto e deixa o ruido como silencio.
#define THRESHOLD_SILENCIO   0.03f
// Gate adaptativo: alem do minimo fixo acima, o firmware mede o ruido de fundo
// da sala (menor RMS de janela nos ultimos PISO_RUIDO_JANELAS segundos) e so
// considera "voz" o que passa de FATOR_PISO_RUIDO vezes esse piso. Numa sala
// silenciosa (piso ~0.004) o gate continua no minimo; numa sala barulhenta ele sobe
// sozinho, senao o ruido de fundo seria classificado como voz feminina.
#define PISO_RUIDO_JANELAS   60      // historico: 60 janelas = ~60 s
#define FATOR_PISO_RUIDO     2.5f
#define PISO_RUIDO_MAX       0.05f   // teto do piso, p/ nao "engolir" fala continua
// Ao confirmar uma voz, o LED pisca LED_PISCADAS vezes (aceso LED_PULSO_MS, apagado
// LED_INTERVALO_MS entre uma e outra) -- um evento de pisca por episodio de fala.
#define LED_PISCADAS         2
#define LED_PULSO_MS         500     // tempo aceso em cada piscada
#define LED_INTERVALO_MS     300     // tempo apagado entre as piscadas

// Vetor de features transferido pela fila (Task 2 -> Task 3)
typedef struct {
    float rms;
    float spectral_centroid;
    float mfccs[DSP_N_MFCC];
    uint32_t timestamp_captura_ms;   // quando a JANELA comecou a ser capturada (Task 1)
    uint32_t latencia_features_us;   // CPU total do DSP na janela (soma de todos os push + finalizar)
    uint32_t latencia_frame_max_us;  // pior caso de processar 1 frame de 1024 amostras
} FeatureVector;
