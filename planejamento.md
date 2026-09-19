# Ponderada — Detector de Anomalias Acústicas

**Data de entrega:** 18/09/2026 · **Checkpoint:** 21/09/2026

---

## Visão geral

O sistema monitora um ambiente esperando apenas vozes femininas. Qualquer voz masculina captada pelo microfone INMP441 é classificada como **anomalia** e acende o LED vermelho; voz feminina mantém o LED verde. A aplicação prática é monitoramento de ambientes restritos — sala privada, dormitório, ou qualquer espaço onde uma entrada não autorizada deva ser sinalizada acusticamente.

### Resumo dos entregáveis

| Entregável | Formato | Critério coberto |
|---|---|---|
| Código FreeRTOS | Repositório GitHub | 25% captura + 15% concorrência |
| Diagrama RTOS | SVG/PNG | Documentação (5%) |
| Modelo treinado | `.onnx` | 20% acurácia |
| Relatório técnico | Markdown/PDF | 5% documentação |
| Script de teste | Python | 15% latência |

### Cronograma sugerido

| Dia | Tarefa |
|---|---|
| Hoje (17/09) | Configurar ambiente Python, baixar dataset |
| 17–18/09 | Treinar e exportar modelo `.onnx` |
| 17–18/09 | Implementar Tasks 1 e 2 no ESP32 |
| 18/09 | Integrar Task 3 + inferência + LED |
| 18/09 | Medir latência, escrever relatório, fechar repositório |

---

## Arquitetura RTOS

O sistema usa **três tarefas FreeRTOS** sincronizadas por um mutex, um semáforo binário e uma fila.

```
[INMP441 / I2S] → Task1 (alta) → [Buffer circular + Mutex] → Task2 (média) → [Fila xQueue] → Task3 (baixa) → [LED]
```

### Task 1 — Captura de áudio (prioridade alta: 5)

- Roda continuamente, acordada pela interrupção I2S
- Lê um bloco de amostras do periférico I2S (`i2s_read`)
- Adquire o **mutex** do buffer circular, grava as amostras, libera o mutex
- Chama `xSemaphoreGive(semSlotCheio)` para acordar a Task 2
- **Watchdog:** se não completar um ciclo em 50 ms, reinicia a tarefa
- Stack sugerida: 4096 bytes

### Task 2 — Extração de features (prioridade média: 3)

- Fica bloqueada em `xSemaphoreTake(semSlotCheio, portMAX_DELAY)`
- Ao acordar: adquire mutex, lê frame do buffer, libera mutex
- Calcula as três features (ver seção Features Acústicas)
- Empacota num struct `FeatureVector` e envia com `xQueueSend(filaFeatures, &fv, 0)`
- Stack sugerida: 8192 bytes (FFT usa bastante stack)

### Task 3 — Detecção de anomalia (prioridade baixa: 1)

- Fica bloqueada em `xQueueReceive(filaFeatures, &fv, portMAX_DELAY)`
- Gate de silêncio: se `fv.rms` abaixo do threshold, trata como "sem fala" (ver lógica de LED abaixo) sem rodar inferência
- Caso contrário, roda inferência no modelo `.onnx`/`.tflite`(ou SVM manual em C, ver seção de conversão) com o vetor de features recebido
- Se `P(masculino) > 0.65` → LED vermelho (anomalia); caso contrário → LED verde (normal)
- Stack sugerida: 16384 bytes (runtime ONNX/TFLite)

**Lógica de LED — 3 estados com retenção (hold time):**

Importante: isso é só a camada de atuação (o que fazer com o GPIO); não muda em nada a lógica de detecção de anomalia (SVM + threshold 0.65) descrita acima.

| Situação | LED |
|---|---|
| Voz masculina detectada | 🔴 vermelho aceso |
| Voz feminina detectada | 🟢 verde aceso |
| Silêncio (sem fala) por mais que o tempo de retenção | ⚫ os dois apagados |

Para evitar "flicker" do LED durante pausas curtas de respiração no meio de uma fala contínua, o LED da última detecção de voz permanece aceso por um **tempo de retenção (hold, ~1.5s)** mesmo que um frame intermediário seja classificado como silêncio. Só depois de `hold` segundos sem nenhuma fala detectada é que os dois LEDs apagam.

```c
#define LED_HOLD_MS 1500

// Estado global da Task 3
uint32_t t_ultima_voz_ms = 0;
enum { LED_APAGADO, LED_VERDE, LED_VERMELHO } led_atual = LED_APAGADO;

// A cada FeatureVector recebido da fila:
if (fv.rms >= THRESHOLD_SILENCIO) {
    led_atual = (prob_masculino > THRESHOLD_ANOMALIA) ? LED_VERMELHO : LED_VERDE;
    t_ultima_voz_ms = esp_timer_get_time() / 1000;
} else if ((esp_timer_get_time() / 1000) - t_ultima_voz_ms > LED_HOLD_MS) {
    led_atual = LED_APAGADO;
}
// senao: mantem led_atual (dentro da janela de retencao)

gpio_set_level(LED_VERDE,    led_atual == LED_VERDE);
gpio_set_level(LED_VERMELHO, led_atual == LED_VERMELHO);
```

Essa lógica de 3 estados + retenção foi validada em Python no `scripts/06_test_live_mic.py` (simulação via terminal, sem hardware) antes de ir para o ESP32.

### Mecanismos de sincronização

| Mecanismo | API FreeRTOS | Protege |
|---|---|---|
| Mutex binário | `xSemaphoreCreateMutex()` | Acesso ao buffer circular (Tasks 1 e 2) |
| Semáforo binário | `xSemaphoreCreateBinary()` | Sinaliza slot cheio (T1 → T2) |
| Fila | `xQueueCreate(5, sizeof(FeatureVector))` | Transfere features (T2 → T3) |

### Struct de dados

```c
// Buffer circular
#define FRAME_SIZE     1024   // amostras por frame
#define BUFFER_SLOTS   4      // número de slots no buffer circular

typedef struct {
    int16_t samples[FRAME_SIZE];
} AudioFrame;

// Vetor de features (enviado pela fila)
typedef struct {
    float rms;
    float spectral_centroid;
    float mfccs[13];
    uint32_t timestamp_ms;
} FeatureVector;
```

---

## Pipeline de dados

```
Microfone → [amostras I2S] → Buffer circular → [frame de 1024 amostras]
         → RMS + FFT + MFCCs → FeatureVector → Modelo SVM → Probabilidade
         → threshold 0.65 → LED vermelho (anomalia) ou LED verde (normal)
```

### Parâmetros de áudio

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Sample rate | 16.000 Hz | Suficiente para voz humana (até 8 kHz útil) |
| Bits por amostra | 16 bits | Padrão I2S do INMP441 |
| Tamanho do frame | 1024 amostras | ~64 ms — bom compromisso latência/resolução |
| Janela de MFCC | 25 ms (400 amostras) | Padrão para reconhecimento de fala |
| Hop size | 10 ms (160 amostras) | Overlap de 60% para suavizar transições |

---

## Modelo de detecção

### Por que SVM com kernel RBF?

Para o contexto desta ponderada, o **SVM (Support Vector Machine) com kernel RBF** é a melhor escolha pelos seguintes motivos:

1. **Tamanho do modelo exportado:** 10–50 KB em `.onnx` — cabe facilmente na flash do ESP32
2. **Velocidade de inferência:** < 5 ms no ESP32 com 13 MFCCs — dentro do orçamento de latência
3. **Acurácia:** 92–96% para classificação binária de gênero com MFCCs, sem precisar de GPU
4. **Facilidade de exportação:** `skl2onnx` converte diretamente de scikit-learn para `.onnx`
5. **Sem dependência de GPU para treino:** roda no seu notebook em minutos

**Alternativas consideradas e descartadas:**

| Modelo | Problema para este contexto |
|---|---|
| MLP / rede neural | Maior, mais lento para inferir, requer TFLite ou ONNX Runtime pesado |
| Random Forest | Arquivo `.onnx` maior, inferência mais lenta no ESP32 |
| CNN em áudio | Exige espectrograma completo, memória insuficiente no ESP32 |
| GMM | Acurácia inferior ao SVM para este problema binário |

### Threshold de decisão

O modelo SVM retorna uma probabilidade via `predict_proba`. O threshold padrão é **0.65** para classe masculina — isto é, só acende o LED vermelho se o modelo tiver pelo menos 65% de confiança. Isso reduz falsos positivos (vozes andróginas ou ruído sendo classificados como anomalia).

---

## Features acústicas

### RMS (Root Mean Square)

Mede a energia do sinal. Vozes têm RMS significativamente maior que silêncio, servindo como **gate de atividade de voz (VAD)** — se RMS < threshold_silencio, a Task 3 não roda inferência, economizando processamento.

```c
float calcular_rms(int16_t *samples, int n) {
    float soma = 0;
    for (int i = 0; i < n; i++) {
        float s = samples[i] / 32768.0f;
        soma += s * s;
    }
    return sqrtf(soma / n);
}
```

### Spectral Centroid

"Centro de gravidade" do espectro — vozes femininas têm centróide mais alto que masculinas. Calculado após FFT.

```c
// Após computar magnitude spectrum mag[N/2]:
float centroid = 0, total_mag = 0;
for (int k = 0; k < N/2; k++) {
    float freq = k * sample_rate / (float)N;
    centroid  += freq * mag[k];
    total_mag += mag[k];
}
centroid /= total_mag;
```

### MFCCs (Mel-Frequency Cepstral Coefficients)

Os 13 primeiros coeficientes capturam o timbre vocal — diferenças de frequência fundamental e formantes entre gêneros. Pipeline de cálculo:

1. **Pré-ênfase:** `y[n] = x[n] - 0.97 * x[n-1]` — reforça altas frequências
2. **Janelamento:** aplica janela de Hamming para reduzir vazamento espectral
3. **FFT:** 512 pontos
4. **Banco de filtros Mel:** 26 filtros triangulares entre 0 e 8000 Hz
5. **Log da energia:** `log(energia_filtro)`
6. **DCT:** Discrete Cosine Transform, retém os 13 primeiros coeficientes

**No ESP32:** usar a biblioteca `esp-dsp` para FFT acelerada (disponível no ESP-IDF). Os filtros Mel e DCT são implementados como matrizes pré-computadas em flash.

---

## Dataset e preparação dos dados

### Dataset recomendado: facebook/voxpopuli (via Hugging Face Hub)

> **Mozilla Common Voice foi descartado.** O site commonvoice.mozilla.org não
> disponibiliza mais os pacotes de áudio para download, e os espelhos oficiais
> no Hugging Face (`mozilla-foundation/common_voice_13_0`,
> `common_voice_17_0`, etc.) também estão vazios — contêm apenas o `README.md`,
> sem nenhum arquivo de áudio (verificado em 17/09/2026). Um mirror
> comunitário (`fsicoli/common_voice_17_0`) tem os áudios, mas seu script de
> carregamento é incompatível com a versão atual da lib `datasets` (bug ao
> ler `n_shards.json`).

- **Fonte:** `facebook/voxpopuli` no Hugging Face Hub — https://huggingface.co/datasets/facebook/voxpopuli
- **Por quê:** discursos do Parlamento Europeu com campo `gender` nativo (`"male"` / `"female"`), áudio já em 16 kHz, distribuído em Parquet (sem loading script customizado — mais estável via streaming que os mirrors do Common Voice)
- **Requisitos:** um token de acesso (`HF_TOKEN` ou `huggingface-cli login`)
- **Coleta:** 3000 exemplos masculinos + 3000 femininos, filtrando pelo campo `gender`, parando o streaming assim que as classes estiverem balanceadas (ver `scripts/01_prepare_dataset.py`)
- **Ressalva:** é fala formal/parlamentar (não conversacional como o Common Voice) — bom o suficiente para o sinal de gênero via MFCC/pitch que o modelo usa, mas vale mencionar no relatório como limitação de domínio

### Preparação dos dados (script Python)

> Nota: os trechos abaixo são o pseudocódigo original de referência. A
> implementação real (streaming do Hugging Face Hub, sem pasta local
> `pasta_cv`) está em `scripts/01_prepare_dataset.py` e
> `scripts/02_extract_features.py`.

```python
import pandas as pd
import librosa
import numpy as np
from pathlib import Path

def extrair_mfccs(caminho_audio, sr=16000, n_mfcc=13):
    y, _ = librosa.load(caminho_audio, sr=sr)
    # Pré-ênfase
    y = librosa.effects.preemphasis(y)
    # MFCCs: média sobre os frames (vetor fixo de 13 números)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.mean(axis=1)  # shape: (13,)

def extrair_rms(caminho_audio, sr=16000):
    y, _ = librosa.load(caminho_audio, sr=sr)
    return float(np.sqrt(np.mean(y**2)))

def extrair_centroid(caminho_audio, sr=16000):
    y, _ = librosa.load(caminho_audio, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(centroid.mean())

def preparar_dataset(pasta_cv, csv_metadados, label_col='gender'):
    df = pd.read_csv(csv_metadados, sep='\t')
    df = df[df[label_col].isin(['male', 'female'])].dropna()

    X, y = [], []
    for _, row in df.iterrows():
        caminho = pasta_cv / 'clips' / row['path']
        if not caminho.exists():
            continue
        mfccs   = extrair_mfccs(caminho)
        rms     = extrair_rms(caminho)
        centroid = extrair_centroid(caminho)
        features = np.concatenate([[rms, centroid], mfccs])  # vetor de 15 números
        X.append(features)
        y.append(1 if row[label_col] == 'male' else 0)  # 1=anomalia, 0=normal

    return np.array(X), np.array(y)
```

### Balanceamento e divisão

```python
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Balancear classes (mesmo número de exemplos masculinos e femininos)
idx_fem  = np.where(y == 0)[0]
idx_masc = np.where(y == 1)[0]
n = min(len(idx_fem), len(idx_masc))
idx_bal = np.concatenate([idx_fem[:n], idx_masc[:n]])
X_bal, y_bal = X[idx_bal], y[idx_bal]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
```

---

## Treinamento do modelo

### Script completo de treino e exportação

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

# 1. Montar pipeline (escalonamento é obrigatório para SVM)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        C=10.0,          # penalidade — ajustar com GridSearchCV se necessário
        gamma='scale',   # gamma automático baseado no número de features
        probability=True # necessário para predict_proba e threshold customizado
    ))
])

# 2. Treinar
pipeline.fit(X_train, y_train)

# 3. Avaliar
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['feminino', 'masculino']))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# 4. Exportar para ONNX
n_features = X_train.shape[1]  # 15
tipo_entrada = [('float_input', FloatTensorType([None, n_features]))]
modelo_onnx = convert_sklearn(pipeline, initial_types=tipo_entrada)

with open('detector_genero_voz.onnx', 'wb') as f:
    f.write(modelo_onnx.SerializeToString())

print("Modelo exportado: detector_genero_voz.onnx")
print(f"Tamanho: {len(modelo_onnx.SerializeToString()) / 1024:.1f} KB")
```

### Ajuste de hiperparâmetros (opcional, mas recomendado)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svm__C':     [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01]
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print("Melhores parâmetros:", grid.best_params_)
pipeline = grid.best_estimator_
```

### Dependências Python

```txt
# requirements.txt
librosa==0.10.2
numpy==1.26.4
scikit-learn==1.5.2
skl2onnx==1.17.0
onnx==1.16.2
pandas==2.2.3
soundfile==0.12.1
```

Instalar com: `pip install -r requirements.txt`

---

## Implementação no ESP32

### Montagem física (confirmada)

**Placa:** ESP32-WROOM-32U (DevKitC, antena externa via U.FL — pinout igual ao DevKit padrão)

**Componentes:** ESP32-WROOM-32U, INMP441 (microfone I2S), 1 LED verde, 1 LED vermelho, 2 resistores (~220-330Ω, um por LED), 1 capacitor cerâmico de disco, 1 capacitor eletrolítico, jumpers. Botão e buzzer disponíveis mas **não usados** neste projeto (fora do escopo do enunciado).

| Componente | Pino | Conecta em |
|---|---|---|
| INMP441 VDD | — | 3.3V |
| INMP441 GND | — | GND |
| INMP441 SCK (BCLK) | — | GPIO26 |
| INMP441 WS (LRCLK) | — | GPIO25 |
| INMP441 SD (dados) | — | GPIO27 |
| INMP441 L/R | — | GND (seleciona canal esquerdo, mono) |
| LED verde (normal/feminino) | GPIO2 | resistor em série → GND |
| LED vermelho (anomalia/masculino) | GPIO32 | resistor em série → GND |

**Decoupling:**
- Capacitor cerâmico de disco (100nF) entre VDD e GND do INMP441, o mais próximo possível do módulo — recomendação do datasheet para reduzir ruído na alimentação do ADC do mic (ruído aqui afeta diretamente RMS/MFCC).
- Capacitor eletrolítico entre 3.3V e GND na trilha geral de alimentação da protoboard (não em um componente específico) — buffer contra picos de corrente do ESP32/I2S, boa prática de estabilidade em protoboard.

**Nota:** pinagem ajustada durante a montagem física real (restrição de espaço na protoboard — só um lado do ESP32 acessível). GPIO22 (SD) virou GPIO27, e GPIO4 (LED vermelho) virou GPIO32 — ambos GPIOs de uso geral, sem função de boot/strapping, equivalentes aos originais. **Evitado:** GPIO3, cogitado inicialmente para o LED vermelho, mas é o pino RXD0 da UART0 (usado pela USB-serial para gravação de firmware e `Serial`/debug) — não pode ser reaproveitado como saída digital.

Esses GPIOs substituem os placeholders usados no pseudocódigo abaixo (`audio_capture.c` e `anomaly_detector.c`) — ajustar as `#define`/`gpio_cfg` para os valores confirmados acima antes de compilar.

### Estrutura do repositório

```
detector-anomalia-acustica/
├── main/
│   ├── main.c              ← entry point, criação das tasks e handles
│   ├── audio_capture.c     ← Task 1: I2S + buffer circular
│   ├── audio_capture.h
│   ├── feature_extractor.c ← Task 2: RMS, centróide, MFCCs
│   ├── feature_extractor.h
│   ├── anomaly_detector.c  ← Task 3: inferência ONNX + LED
│   ├── anomaly_detector.h
│   ├── model/
│   │   └── detector_genero_voz.onnx  ← modelo embarcado
│   └── CMakeLists.txt
├── test/
│   └── test_pipeline.py    ← script de teste de performance
├── docs/
│   └── diagrama_rtos.svg
├── README.md
└── CMakeLists.txt
```

### Esqueleto do main.c

```c
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include "audio_capture.h"
#include "feature_extractor.h"
#include "anomaly_detector.h"

// Handles globais de sincronização
SemaphoreHandle_t xMutexBuffer;
SemaphoreHandle_t xSemSlotCheio;
QueueHandle_t     xFilaFeatures;

void app_main(void) {
    // Criar mecanismos de sincronização
    xMutexBuffer   = xSemaphoreCreateMutex();
    xSemSlotCheio  = xSemaphoreCreateBinary();
    xFilaFeatures  = xQueueCreate(5, sizeof(FeatureVector));

    configASSERT(xMutexBuffer  != NULL);
    configASSERT(xSemSlotCheio != NULL);
    configASSERT(xFilaFeatures != NULL);

    // Criar as três tasks
    xTaskCreate(task_captura,   "captura",  4096,  NULL, 5, NULL);
    xTaskCreate(task_features,  "features", 8192,  NULL, 3, NULL);
    xTaskCreate(task_deteccao,  "deteccao", 16384, NULL, 1, NULL);
}
```

### Esqueleto da Task 1 (audio_capture.c)

```c
#include "driver/i2s_std.h"
#include "freertos/semphr.h"

#define SAMPLE_RATE  16000
#define FRAME_SIZE   1024
#define BUFFER_SLOTS 4

static int16_t buffer[BUFFER_SLOTS][FRAME_SIZE];
static int     slot_escrita = 0;

extern SemaphoreHandle_t xMutexBuffer;
extern SemaphoreHandle_t xSemSlotCheio;

void task_captura(void *pvParameters) {
    // Configurar I2S para INMP441
    i2s_chan_handle_t rx_handle;
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    i2s_new_channel(&chan_cfg, NULL, &rx_handle);

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = GPIO_NUM_NC,
            .bclk = GPIO_NUM_26,   // ajustar para seu hardware
            .ws   = GPIO_NUM_25,
            .dout = GPIO_NUM_NC,
            .din  = GPIO_NUM_22,
        },
    };
    i2s_channel_init_std_mode(rx_handle, &std_cfg);
    i2s_channel_enable(rx_handle);

    size_t bytes_lidos;
    while (1) {
        int16_t temp[FRAME_SIZE];
        i2s_channel_read(rx_handle, temp, FRAME_SIZE * sizeof(int16_t), &bytes_lidos, pdMS_TO_TICKS(100));

        xSemaphoreTake(xMutexBuffer, portMAX_DELAY);
        memcpy(buffer[slot_escrita], temp, FRAME_SIZE * sizeof(int16_t));
        slot_escrita = (slot_escrita + 1) % BUFFER_SLOTS;
        xSemaphoreGive(xMutexBuffer);

        xSemaphoreGive(xSemSlotCheio);  // acorda Task 2
    }
}
```

### Esqueleto da Task 2 (feature_extractor.c)

```c
#include "esp_dsp.h"  // FFT acelerada do ESP-IDF
#include "math.h"

extern SemaphoreHandle_t xMutexBuffer;
extern SemaphoreHandle_t xSemSlotCheio;
extern QueueHandle_t     xFilaFeatures;

// Matriz de filtros Mel pré-computada (26 filtros, 257 bins de frequência)
static float mel_filterbank[26][257];  // inicializada em setup

void task_features(void *pvParameters) {
    inicializar_mel_filterbank(mel_filterbank, SAMPLE_RATE, 257, 26);

    while (1) {
        xSemaphoreTake(xSemSlotCheio, portMAX_DELAY);

        // Ler frame do buffer
        int16_t frame[FRAME_SIZE];
        xSemaphoreTake(xMutexBuffer, portMAX_DELAY);
        // (ler do slot de leitura do buffer circular)
        xSemaphoreGive(xMutexBuffer);

        FeatureVector fv;
        fv.timestamp_ms = esp_timer_get_time() / 1000;

        // RMS
        fv.rms = calcular_rms(frame, FRAME_SIZE);

        // FFT → Spectral Centroid e MFCCs
        float espectro[FRAME_SIZE];
        computar_fft(frame, espectro, FRAME_SIZE);
        fv.spectral_centroid = calcular_centroid(espectro, FRAME_SIZE, SAMPLE_RATE);
        calcular_mfccs(espectro, mel_filterbank, fv.mfccs, 13);

        xQueueSend(xFilaFeatures, &fv, 0);  // não bloqueia: descarta se fila cheia
    }
}
```

### Esqueleto da Task 3 (anomaly_detector.c)

```c
#include "driver/gpio.h"

#define LED_VERDE    GPIO_NUM_2
#define LED_VERMELHO GPIO_NUM_4
#define THRESHOLD_ANOMALIA  0.65f
#define THRESHOLD_SILENCIO  0.01f
#define LED_HOLD_MS         1500   // retencao: evita flicker em pausas curtas de fala

extern QueueHandle_t xFilaFeatures;

// Função de inferência ONNX (implementar com runtime leve)
// Opções: ONNX Runtime for ESP32, ou converter para TFLite e usar TFLite Micro
float inferir_probabilidade_masculino(FeatureVector *fv);

typedef enum { LED_APAGADO, LED_VERDE_ON, LED_VERMELHO_ON } led_estado_t;

void task_deteccao(void *pvParameters) {
    gpio_set_direction(LED_VERDE,    GPIO_MODE_OUTPUT);
    gpio_set_direction(LED_VERMELHO, GPIO_MODE_OUTPUT);

    led_estado_t led_atual = LED_APAGADO;
    uint32_t t_ultima_voz_ms = 0;

    FeatureVector fv;
    while (1) {
        xQueueReceive(xFilaFeatures, &fv, portMAX_DELAY);
        uint32_t agora_ms = esp_timer_get_time() / 1000;

        // Gate de silêncio: não roda inferência se energia muito baixa
        if (fv.rms >= THRESHOLD_SILENCIO) {
            uint32_t t_inicio = esp_timer_get_time();
            float prob_masculino = inferir_probabilidade_masculino(&fv);
            uint32_t t_inferencia = esp_timer_get_time() - t_inicio;

            led_atual = (prob_masculino > THRESHOLD_ANOMALIA) ? LED_VERMELHO_ON : LED_VERDE_ON;
            t_ultima_voz_ms = agora_ms;

            if (led_atual == LED_VERMELHO_ON) {
                ESP_LOGI("detector", "ANOMALIA: voz masculina (%.2f) | inferência: %lu µs",
                         prob_masculino, t_inferencia);
            }
        } else if (agora_ms - t_ultima_voz_ms > LED_HOLD_MS) {
            // silencio ha mais tempo que a retencao: apaga os dois
            led_atual = LED_APAGADO;
        }
        // senao: silencio momentaneo dentro da janela de retencao -> mantem led_atual

        gpio_set_level(LED_VERDE,    led_atual == LED_VERDE_ON);
        gpio_set_level(LED_VERMELHO, led_atual == LED_VERMELHO_ON);
    }
}
```

---

## Latência e análise de performance

### Orçamento de latência (meta)

| Etapa | Tempo alvo | Como medir |
|---|---|---|
| Captura (1 frame) | ~64 ms | Determinístico: FRAME_SIZE / SAMPLE_RATE |
| Extração de features | 30–50 ms | `esp_timer_get_time()` antes/depois da Task 2 |
| Inferência SVM | 5–20 ms | `esp_timer_get_time()` antes/depois de `inferir_probabilidade_masculino` |
| **Total ponta a ponta** | **~100–140 ms** | Timestamp na captura e no acionamento do LED |

A latência total de ~100–140 ms é aceitável para monitoramento ambiental — o ouvido humano leva ~150 ms para processar e reagir a um som, então o sistema responde tão rápido quanto um observador humano.

### Como documentar no relatório

```c
// No início da Task 1, ao capturar o frame:
uint32_t t_captura = esp_timer_get_time();

// No início da Task 2, ao receber o semáforo:
uint32_t t_features_inicio = esp_timer_get_time();
// ... cálculos ...
uint32_t t_features_fim = esp_timer_get_time();
uint32_t latencia_features = t_features_fim - t_features_inicio;

// Na Task 3, antes e depois da inferência:
uint32_t t_inf_inicio = esp_timer_get_time();
float prob = inferir_probabilidade_masculino(&fv);
uint32_t latencia_inf = esp_timer_get_time() - t_inf_inicio;

// Latência total (do início da captura até o LED acender):
uint32_t latencia_total = esp_timer_get_time() - t_captura;

ESP_LOGI("latencia", "features=%luµs | inferencia=%luµs | total=%luµs",
         latencia_features, latencia_inf, latencia_total);
```

---

## Script de teste de performance

```python
#!/usr/bin/env python3
"""
test_pipeline.py — Testa o pipeline offline com áudios reais
Simula frames de áudio e mede latência de cada etapa
"""
import numpy as np
import librosa
import time
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
import joblib

# Carregar modelo e scaler
sess = ort.InferenceSession('detector_genero_voz.onnx')
# (o scaler está embutido no pipeline ONNX — não precisa carregar separado)

def extrair_features_frame(frame: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extrai RMS + Spectral Centroid + 13 MFCCs de um frame."""
    rms      = float(np.sqrt(np.mean(frame**2)))
    centroid = float(librosa.feature.spectral_centroid(y=frame, sr=sr).mean())
    mfccs    = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13).mean(axis=1)
    return np.concatenate([[rms, centroid], mfccs]).astype(np.float32)

def inferir(features: np.ndarray) -> dict:
    """Roda inferência e retorna probabilidades."""
    entrada = features.reshape(1, -1)
    saida   = sess.run(None, {'float_input': entrada})
    # saida[1] = array de probabilidades [P(feminino), P(masculino)]
    prob_masc = float(saida[1][0][1])
    return {'prob_masculino': prob_masc, 'anomalia': prob_masc > 0.65}

def testar_arquivo(caminho_audio: str, genero_real: str):
    """Testa um arquivo de áudio frame a frame e mede latência."""
    y, sr = librosa.load(caminho_audio, sr=16000)
    FRAME_SIZE = 1024
    n_frames = len(y) // FRAME_SIZE

    resultados = []
    for i in range(n_frames):
        frame = y[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]

        t0 = time.perf_counter()
        features = extrair_features_frame(frame, sr)
        t1 = time.perf_counter()
        resultado = inferir(features)
        t2 = time.perf_counter()

        resultados.append({
            'frame': i,
            'anomalia': resultado['anomalia'],
            'prob_masc': resultado['prob_masculino'],
            'lat_features_ms': (t1 - t0) * 1000,
            'lat_inferencia_ms': (t2 - t1) * 1000,
        })

    # Calcular métricas
    acertos = sum(1 for r in resultados
                  if (r['anomalia'] and genero_real == 'masculino')
                  or (not r['anomalia'] and genero_real == 'feminino'))
    acuracia = acertos / len(resultados) * 100
    lat_feat = np.mean([r['lat_features_ms'] for r in resultados])
    lat_inf  = np.mean([r['lat_inferencia_ms'] for r in resultados])

    print(f"Arquivo: {caminho_audio} | Gênero real: {genero_real}")
    print(f"  Acurácia por frame: {acuracia:.1f}%")
    print(f"  Latência features: {lat_feat:.2f} ms")
    print(f"  Latência inferência: {lat_inf:.2f} ms")
    print(f"  Latência total (sem captura): {lat_feat + lat_inf:.2f} ms")
    return resultados

if __name__ == '__main__':
    # Testar com arquivos de exemplo (substituir pelos seus)
    testar_arquivo('audio_teste_feminino.wav', 'feminino')
    testar_arquivo('audio_teste_masculino.wav', 'masculino')
```

---

## Checklist de avaliação

### Antes de entregar (18/09)

- [ ] Repositório público no GitHub com README claro
- [ ] `main.c` com três tasks criadas e prioridades corretas
- [ ] Mutex e semáforo documentados no código (comentários explicando o conflito resolvido)
- [ ] Fila `xQueueCreate` com tamanho definido e justificado
- [ ] Arquivo `detector_genero_voz.onnx` no repositório
- [ ] Diagrama RTOS (SVG ou PNG) na pasta `docs/`
- [ ] Relatório técnico com tabela de latências medidas
- [ ] Script `test_pipeline.py` funcional com métricas de acurácia
- [ ] LED respondendo corretamente (verde = feminino, vermelho = masculino, ambos apagados = silêncio, com retenção de ~1.5s para evitar flicker)
- [ ] Seção no README explicando a aplicação prática e justificativa da anomalia

### Para o checkpoint oral (21/09)

Duas perguntas prováveis e respostas-chave:

**"Por que usar semáforo binário entre Task 1 e Task 2 em vez de notificação direta?"**
→ O semáforo desacopla produtor e consumidor — a Task 1 pode dar `Give` mesmo se a Task 2 ainda não acordou, e a Task 2 fica bloqueada sem consumir CPU enquanto não há frame disponível. Notificação de task (`xTaskNotify`) seria equivalente, mas o semáforo é mais explícito e legível para sincronização produtor-consumidor.

**"O que acontece se a Task 3 (inferência) demorar mais que a Task 2 está produzindo features?"**
→ A fila de 5 posições funciona como buffer. Se encher, o `xQueueSend` com timeout 0 na Task 2 descarta o frame mais antigo — comportamento aceitável porque estamos processando áudio em tempo real e um frame atrasado é menos útil que o frame atual. Isso é documentado na análise de latência.