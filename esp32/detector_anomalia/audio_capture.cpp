// audio_capture.cpp -- ver audio_capture.h
#include "audio_capture.h"
#include "sync.h"
#include <Arduino.h>
#include <driver/i2s.h>
#include <string.h>

#define I2S_PORT I2S_NUM_0

int16_t g_circular_buffer[BUFFER_SLOTS][FRAME_SIZE];
volatile int g_ultimo_slot_pronto = -1;
volatile uint32_t g_ultimo_frame_timestamp_ms = 0;

static int32_t s_leitura_raw[FRAME_SIZE];  // INMP441: 24 bits alinhados a esquerda em slot de 32 bits

void i2s_iniciar() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = DSP_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = FRAME_SIZE,
        .use_apll = false,
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN,
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

// Task 1 (prioridade alta): le um frame do I2S, grava no proximo slot do
// buffer circular sob mutex, e sinaliza a Task 2 via semaforo binario.
//
// Watchdog simples: se um ciclo de captura (leitura I2S + escrita no
// buffer) levar mais que WATCHDOG_MS, loga um aviso -- em produção isso
// poderia reiniciar a task; aqui documentamos a deteccao e priorizamos
// nao mascarar o problema silenciosamente.
//
// IMPORTANTE: ler FRAME_SIZE=1024 amostras via I2S a 16kHz leva no MINIMO
// 1024/16000 = 64ms fisicamente (e o tempo real de chegada do audio -- o
// i2s_read bloqueia ate ter amostras suficientes). Um limite de 50ms
// disparava o watchdog em todo ciclo (falso positivo constante). 100ms da
// margem real acima do minimo teorico de 64ms.
#define WATCHDOG_MS 100

void task_captura(void *pvParameters) {
    i2s_iniciar();
    int slot_escrita = 0;

    while (1) {
        uint32_t t_inicio = millis();

        size_t bytes_lidos = 0;
        i2s_read(I2S_PORT, s_leitura_raw, sizeof(s_leitura_raw), &bytes_lidos, portMAX_DELAY);
        int n = bytes_lidos / sizeof(int32_t);

        int16_t frame_convertido[FRAME_SIZE];
        for (int i = 0; i < n && i < FRAME_SIZE; i++) {
            // 24 bits alinhados a esquerda num slot de 32 -- shift aritmetico
            // preserva o sinal. Reduz para 16 bits (>>8 de novo) para caber
            // no int16_t do buffer circular, mantendo compatibilidade com o
            // formato usado no treino (amostras normalizadas em [-1,1]
            // equivalentes a int16).
            int32_t amostra24 = s_leitura_raw[i] >> 8;
            int32_t amostra16 = amostra24 >> 8;
            if (amostra16 > 32767) amostra16 = 32767;
            if (amostra16 < -32768) amostra16 = -32768;
            frame_convertido[i] = (int16_t)amostra16;
        }

        xSemaphoreTake(xMutexBuffer, portMAX_DELAY);
        memcpy(g_circular_buffer[slot_escrita], frame_convertido, sizeof(frame_convertido));
        g_ultimo_slot_pronto = slot_escrita;
        g_ultimo_frame_timestamp_ms = millis();
        xSemaphoreGive(xMutexBuffer);

        slot_escrita = (slot_escrita + 1) % BUFFER_SLOTS;

        xSemaphoreGive(xSemSlotCheio);

        uint32_t duracao = millis() - t_inicio;
        if (duracao > WATCHDOG_MS) {
            Serial.printf("[WATCHDOG][captura] ciclo levou %lums (> %dms)\n", duracao, WATCHDOG_MS);
        }
    }
}
