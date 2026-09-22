// led_control.h -- atuacao nos LEDs: cada evento de deteccao PISCA o LED
// correspondente LED_PISCADAS vezes (2) e apaga. A decisao de QUANDO
// piscar (uma vez por episodio de fala) fica na Task 3 (detect_task.cpp);
// aqui so a camada de atuacao.
#pragma once
#include <stdint.h>

typedef enum { LED_ESTADO_APAGADO, LED_ESTADO_VERDE, LED_ESTADO_VERMELHO } led_estado_t;

void led_control_init();

// Pisca o LED do estado (verde = voz feminina, vermelho = voz masculina)
// LED_PISCADAS vezes e apaga. Bloqueia a Task 3 durante o pisca (baixa prioridade; a fila
// absorve o atraso). LED_ESTADO_APAGADO nao faz nada.
void led_control_piscar(led_estado_t estado);
