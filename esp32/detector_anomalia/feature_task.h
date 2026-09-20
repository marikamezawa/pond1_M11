// feature_task.h -- Task 2: acumula frames do buffer circular (Task 1) em
// segmentos de 1s; a cada segmento envia um FeatureVector (janela de 3s
// deslizante, calculada em dsp.cpp) pela fila para a Task 3.
#pragma once

void task_features(void *pvParameters);
