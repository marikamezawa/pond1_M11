// audio_capture.h -- Task 1: captura continua via I2S (INMP441), alta
// prioridade. Enche um buffer circular pequeno (BUFFER_SLOTS x FRAME_SIZE)
// protegido por mutex, e sinaliza um semaforo binario a cada frame novo.
#pragma once
#include <stdint.h>
#include "config.h"

// Buffer circular compartilhado (definido em audio_capture.cpp).
// Task 2 le isso sob posse do xMutexBuffer, olhando g_ultimo_slot_pronto
// para saber qual slot acabou de ser preenchido.
extern int16_t g_circular_buffer[BUFFER_SLOTS][FRAME_SIZE];
extern volatile int g_ultimo_slot_pronto;
extern volatile uint32_t g_ultimo_frame_timestamp_ms;

void i2s_iniciar();
void task_captura(void *pvParameters);
