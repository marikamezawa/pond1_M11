// bringup_led.ino -- teste de bring-up: so pisca os 2 LEDs alternadamente.
// Objetivo: confirmar que GPIO2/GPIO32 + resistores + fiacao estao corretos,
// antes de integrar qualquer logica de deteccao.
//
// Esperado: LED verde e vermelho alternando a cada 500ms, sem os dois
// acesos ao mesmo tempo e sem os dois apagados ao mesmo tempo.

#define LED_VERDE    2
#define LED_VERMELHO 32

void setup() {
  pinMode(LED_VERDE, OUTPUT);
  pinMode(LED_VERMELHO, OUTPUT);
}

void loop() {
  digitalWrite(LED_VERDE, HIGH);
  digitalWrite(LED_VERMELHO, LOW);
  delay(500);

  digitalWrite(LED_VERDE, LOW);
  digitalWrite(LED_VERMELHO, HIGH);
  delay(500);
}
