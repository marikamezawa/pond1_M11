// sync.h -- handles de sincronizacao FreeRTOS compartilhados entre as 3
// tasks. Definidos no .ino principal, declarados aqui como extern.
#pragma once
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/queue.h>

extern SemaphoreHandle_t xMutexBuffer;   // protege o buffer circular (Task1 <-> Task2)
extern SemaphoreHandle_t xSemSlotCheio;  // Task1 -> Task2: "tem um novo frame pronto"
extern QueueHandle_t     xFilaFeatures;  // Task2 -> Task3: FeatureVector
