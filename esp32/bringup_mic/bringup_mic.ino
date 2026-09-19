// bringup_mic.ino -- teste de bring-up: le o INMP441 via I2S e imprime o
// RMS de cada bloco de amostras no Serial Monitor.
// Objetivo: confirmar que o microfone esta cabeado certo e entregando
// audio de verdade (nao ruido/lixo/silencio constante), antes de integrar
// a logica de extracao de features e deteccao.
//
// Abra o Serial Monitor a 115200 baud. Esperado:
//   - em silencio, rms baixo e razoavelmente estavel (algumas centenas)
//   - falando perto do mic, rms sobe bem visivelmente (milhares+)
//   - se rms ficar sempre ~0 ou sempre no maximo (constante), ha problema
//     de fiacao (checar SCK/WS/SD, L/R no GND, alimentacao do mic)

#include <driver/i2s.h>

#define I2S_SCK 26
#define I2S_WS  25
#define I2S_SD  27
#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE 16000
#define BLOCK_SIZE  1024

int32_t samples_raw[BLOCK_SIZE];

void setup() {
  Serial.begin(115200);
  delay(500);

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = BLOCK_SIZE,
    .use_apll = false,
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD,
  };

  esp_err_t err;
  err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("Erro ao instalar driver I2S: %d\n", err);
  }
  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("Erro ao configurar pinos I2S: %d\n", err);
  }
}

void loop() {
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, samples_raw, sizeof(samples_raw), &bytes_read, portMAX_DELAY);
  int n = bytes_read / sizeof(int32_t);

  double soma_quadrados = 0;
  int32_t pico = 0;
  for (int i = 0; i < n; i++) {
    // INMP441 entrega dado de 24 bits alinhado a esquerda num slot de 32 bits.
    // Shift aritmetico >> 8 preserva o sinal e devolve o valor de 24 bits real.
    int32_t amostra = samples_raw[i] >> 8;
    int32_t abs_amostra = amostra < 0 ? -amostra : amostra;
    if (abs_amostra > pico) pico = abs_amostra;
    soma_quadrados += (double)amostra * (double)amostra;
  }
  double rms = sqrt(soma_quadrados / n);

  Serial.printf("amostras=%d  rms=%.1f  pico=%ld\n", n, rms, (long)pico);
}
