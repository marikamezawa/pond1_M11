// audio_dump.ino -- grava trechos de audio reais do INMP441 e transmite
// pela Serial (baud alto) para o computador salvar como .wav.
//
// Objetivo: capturar audio REAL do hardware (INMP441 + protoboard + sala)
// para incluir no treino do modelo, corrigindo o descompasso de dominio
// (resposta em frequencia do mic, ruido eletrico) que a normalizacao de
// energia por frame sozinha nao resolve.
//
// Uso:
//   1. Grave este sketch no ESP32.
//   2. Rode scripts/08_capturar_audio_hardware.py no computador (mesma
//      porta serial, 921600 baud).
//   3. O script manda um gatilho; o ESP32 grava RECORD_SECONDS de audio
//      e transmite. Repete a cada novo gatilho -- da pra capturar varios
//      trechos sem regravar o firmware.

#include <driver/i2s.h>

#define I2S_SCK 26
#define I2S_WS  25
#define I2S_SD  27
#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE 16000
#define DEFAULT_SECONDS 5
#define MAX_SECONDS 120

// Nao da pra bufferizar os 5s inteiros em RAM estatica (160000 bytes --
// estoura o DRAM junto com os buffers do driver I2S/WiFi). Transmite em
// blocos de FRAME_SIZE conforme chega do I2S.
#define FRAME_SIZE 1024
static int16_t g_frame[FRAME_SIZE];

void i2s_iniciar() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
  };
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD,
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void setup() {
  Serial.begin(921600);
  delay(500);
  i2s_iniciar();
  Serial.println("PRONTO");
}

void gravar_e_enviar(int segundos) {
  int32_t raw[FRAME_SIZE];
  int pos = 0;
  const int N_AMOSTRAS = SAMPLE_RATE * segundos;

  // Descarta ~0.5s iniciais: o INMP441 tem transiente de partida (pop/apito)
  // e o DMA do I2S ainda guarda dados velhos de antes do gatilho.
  i2s_zero_dma_buffer(I2S_PORT);
  for (int i = 0; i < 8; i++) {
    size_t descartados = 0;
    i2s_read(I2S_PORT, raw, FRAME_SIZE * sizeof(int32_t), &descartados, portMAX_DELAY);
  }

  Serial.println("START");

  while (pos < N_AMOSTRAS) {
    size_t bytes_lidos = 0;
    int n_pedir = min(FRAME_SIZE, N_AMOSTRAS - pos);
    i2s_read(I2S_PORT, raw, n_pedir * sizeof(int32_t), &bytes_lidos, portMAX_DELAY);
    int n_lido = bytes_lidos / sizeof(int32_t);
    for (int i = 0; i < n_lido; i++) {
      int32_t amostra24 = raw[i] >> 8;
      int32_t amostra16 = amostra24 >> 8;
      if (amostra16 > 32767) amostra16 = 32767;
      if (amostra16 < -32768) amostra16 = -32768;
      g_frame[i] = (int16_t)amostra16;
    }
    Serial.write((uint8_t *)g_frame, n_lido * sizeof(int16_t));
    pos += n_lido;
  }

  Serial.println();
  Serial.println("DONE");
}

void loop() {
  if (Serial.available() > 0) {
    // 1o byte do gatilho = duracao em segundos (1..MAX_SECONDS); outro valor = padrao
    int segundos = Serial.read();
    if (segundos < 1 || segundos > MAX_SECONDS) segundos = DEFAULT_SECONDS;
    while (Serial.available() > 0) Serial.read();  // limpa o resto
    Serial.println("GRAVANDO");
    gravar_e_enviar(segundos);
    Serial.println("PRONTO");
  }
}
