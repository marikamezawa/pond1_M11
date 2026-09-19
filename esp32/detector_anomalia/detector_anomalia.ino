// detector_anomalia.ino -- entry point. Cria os mecanismos de
// sincronizacao FreeRTOS e as 3 tasks (Arduino framework, nao ESP-IDF).
//
// Arquitetura (ver planejamento.md para o desenho completo):
//   Task 1 (captura, prioridade 5/alta)   -> I2S + buffer circular + mutex
//   Task 2 (features, prioridade 3/media) -> acumula janela + MFCC + fila
//   Task 3 (deteccao, prioridade 1/baixa) -> SVM + LED (3 estados + retencao)
#include "sync.h"
#include "config.h"
#include "audio_capture.h"
#include "feature_task.h"
#include "detect_task.h"

SemaphoreHandle_t xMutexBuffer;
SemaphoreHandle_t xSemSlotCheio;
QueueHandle_t xFilaFeatures;

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("Detector de Anomalias Acusticas -- iniciando...");

    xMutexBuffer = xSemaphoreCreateMutex();
    xSemSlotCheio = xSemaphoreCreateBinary();
    xFilaFeatures = xQueueCreate(5, sizeof(FeatureVector));

    if (xMutexBuffer == NULL || xSemSlotCheio == NULL || xFilaFeatures == NULL) {
        Serial.println("ERRO: falha ao criar mecanismos de sincronizacao FreeRTOS");
        while (1) { delay(1000); }
    }

    // Stacks: os buffers grandes (janela de 3s, support vectors do SVM,
    // etc.) sao estaticos/globais, nao alocados na stack da task -- os
    // valores abaixo cobrem variaveis locais + margem para Serial.printf.
    xTaskCreate(task_captura,  "captura",  4096,  NULL, 5, NULL);
    xTaskCreate(task_features, "features", 8192,  NULL, 3, NULL);
    xTaskCreate(task_deteccao, "deteccao", 8192,  NULL, 1, NULL);

    Serial.println("3 tasks criadas. Sistema em execucao.");
}

void loop() {
    // Toda a logica roda nas 3 tasks FreeRTOS; loop() fica ocioso.
    vTaskDelay(pdMS_TO_TICKS(1000));
}
