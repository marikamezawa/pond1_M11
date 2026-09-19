// led_control.h -- logica de 3 estados com retencao (hold), a mesma
// validada em Python (scripts/06_test_live_mic.py) antes de portar pro
// ESP32. So a camada de atuacao: nao interfere na logica de deteccao.
#pragma once
#include <stdint.h>
#include <stdbool.h>

typedef enum { LED_ESTADO_APAGADO, LED_ESTADO_VERDE, LED_ESTADO_VERMELHO } led_estado_t;

void led_control_init();

// Chamado a cada FeatureVector processado pela Task 3.
//   tem_voz: true se rms >= THRESHOLD_SILENCIO (janela nao-silenciosa)
//   eh_anomalia: true se prob_masculino > THRESHOLD_ANOMALIA (so relevante se tem_voz)
//   agora_ms: timestamp atual (millis())
// Aplica o resultado nos GPIOs dos LEDs e retorna o estado resultante.
led_estado_t led_control_atualizar(bool tem_voz, bool eh_anomalia, uint32_t agora_ms);
