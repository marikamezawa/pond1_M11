// led_control.cpp -- ver led_control.h
#include "led_control.h"
#include "config.h"
#include <Arduino.h>

// Fiacao padrao (ativa-em-HIGH): GPIO -> resistor -> anodo, catodo -> GND.
// (Uma montagem anterior estava com anodo no 3.3V e catodo/resistor no
// GPIO -- ativa-em-LOW, invertido -- corrigido fisicamente na protoboard
// em vez de compensar aqui no codigo. Ver README.md.)
#define LED_ON  HIGH
#define LED_OFF LOW

static led_estado_t g_led_atual = LED_ESTADO_APAGADO;
static uint32_t g_ultima_voz_ms = 0;

void led_control_init() {
    pinMode(LED_VERDE_PIN, OUTPUT);
    pinMode(LED_VERMELHO_PIN, OUTPUT);
    digitalWrite(LED_VERDE_PIN, LED_OFF);
    digitalWrite(LED_VERMELHO_PIN, LED_OFF);
    g_led_atual = LED_ESTADO_APAGADO;
    g_ultima_voz_ms = 0;
}

led_estado_t led_control_atualizar(bool tem_voz, bool eh_anomalia, uint32_t agora_ms) {
    if (tem_voz) {
        g_led_atual = eh_anomalia ? LED_ESTADO_VERMELHO : LED_ESTADO_VERDE;
        g_ultima_voz_ms = agora_ms;
    } else if ((agora_ms - g_ultima_voz_ms) > LED_HOLD_MS) {
        g_led_atual = LED_ESTADO_APAGADO;
    }
    // senao: silencio momentaneo dentro da janela de retencao -> mantem g_led_atual

    digitalWrite(LED_VERDE_PIN, g_led_atual == LED_ESTADO_VERDE ? LED_ON : LED_OFF);
    digitalWrite(LED_VERMELHO_PIN, g_led_atual == LED_ESTADO_VERMELHO ? LED_ON : LED_OFF);

    return g_led_atual;
}

led_estado_t led_control_incerto(uint32_t agora_ms) {
    g_led_atual = LED_ESTADO_APAGADO;
    g_ultima_voz_ms = agora_ms;
    digitalWrite(LED_VERDE_PIN, LED_OFF);
    digitalWrite(LED_VERMELHO_PIN, LED_OFF);
    return g_led_atual;
}
