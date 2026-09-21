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

void led_control_init() {
    pinMode(LED_VERDE_PIN, OUTPUT);
    pinMode(LED_VERMELHO_PIN, OUTPUT);
    digitalWrite(LED_VERDE_PIN, LED_OFF);
    digitalWrite(LED_VERMELHO_PIN, LED_OFF);
}

void led_control_piscar(led_estado_t estado) {
    if (estado == LED_ESTADO_APAGADO) return;
    int pino = (estado == LED_ESTADO_VERMELHO) ? LED_VERMELHO_PIN : LED_VERDE_PIN;
    digitalWrite(pino, LED_ON);
    delay(LED_PULSO_MS);
    digitalWrite(pino, LED_OFF);
}
