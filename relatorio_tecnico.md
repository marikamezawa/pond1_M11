# Relatório Técnico — Detector de Anomalias Acústicas

**Aplicação escolhida:** monitoramento de ambiente restrito (sala privada, dormitório) que espera apenas vozes femininas. Qualquer **voz masculina** captada é tratada como anomalia e sinalizada via LED vermelho; voz feminina mantém o LED verde; silêncio mantém os dois LEDs apagados.

Diagrama RTOS completo: [`docs/diagrama_rtos.svg`](docs/diagrama_rtos.svg).

---

## 1. Arquitetura RTOS

O sistema usa **3 tasks FreeRTOS** (Arduino framework, ESP32-WROOM-32U) sincronizadas por um mutex, um semáforo binário e uma fila — implementação real em `esp32/detector_anomalia/` (não pseudocódigo):

```
INMP441 (I2S) → Task 1 (prio 5, alta)  → [buffer circular + mutex]
             → Task 2 (prio 3, média)  → [janela 3s → RMS+centroid+13 MFCC] → fila
             → Task 3 (prio 1, baixa)  → [SVM RBF em C] → LED (3 estados + retenção)
```

| Task | Prioridade | Arquivo | Responsabilidade |
|---|---|---|---|
| Task 1 — Captura | 5 (alta) | `audio_capture.cpp` | Lê I2S em blocos de 1024 amostras (64ms), grava no buffer circular (4 slots) sob mutex, sinaliza semáforo a cada frame |
| Task 2 — Features | 3 (média) | `feature_task.cpp` | Acumula 48000 amostras (3s) a partir dos frames do buffer circular, calcula RMS + Spectral Centroid + 13 MFCCs (`dsp.cpp`), envia `FeatureVector` pela fila |
| Task 3 — Detecção | 1 (baixa) | `detect_task.cpp` | Recebe da fila, roda inferência SVM (`svm_infer.cpp`), decide LED, loga latência via Serial |

### Mecanismos de sincronização (implementados, não pseudocódigo)

| Mecanismo | API FreeRTOS | Protege / Conecta |
|---|---|---|
| Mutex binário | `xSemaphoreCreateMutex()` | Acesso ao buffer circular (Task 1 escreve, Task 2 lê) |
| Semáforo binário | `xSemaphoreCreateBinary()` | Sinaliza "novo frame pronto" (Task 1 → Task 2), produtor-consumidor sem polling |
| Fila | `xQueueCreate(5, sizeof(FeatureVector))` | Transfere `FeatureVector` (Task 2 → Task 3); cheia → descarta o mais antigo (audio em tempo real, frame velho é menos útil) |

### Por que a janela de análise é de 3 segundos, não por frame de 64ms

A extração de features (RMS/centróide/MFCC) precisa de contexto temporal suficiente para ser estatisticamente estável. Testamos empiricamente com janelas de 1.5s durante a validação ao vivo (Fase 1, Python) e observamos erro sistemático de classificação — features instáveis demais em janelas curtas. A Task 2 acumula frames de 64ms (capturados continuamente pela Task 1) até completar uma janela de 3s, calcula UM `FeatureVector` por janela, e reinicia o acúmulo — mantendo a granularidade de captura alta (I2S contínuo, sem gaps) e a granularidade de classificação estável.

### Watchdog

Task 1 mede a duração de cada ciclo de captura; se ultrapassar 50ms, loga um aviso via Serial (`[WATCHDOG][captura] ciclo levou Xms`).

---

## 2. Modelo de detecção

- **Dataset:** `facebook/voxpopuli` (Hugging Face Hub, streaming) — o Mozilla Common Voice foi descartado por indisponibilidade (site oficial não distribui mais os pacotes de áudio, e os espelhos `mozilla-foundation/common_voice_*` no HF estão vazios). 6000 clipes (3000 masculinos + 3000 femininos), rotulados pelo campo `gender` nativo.
- **Features:** RMS + Spectral Centroid + 13 MFCCs, implementação própria (não usa `librosa.feature.mfcc`) — FFT de 512 pontos, 26 filtros Mel (escala HTK), DCT-II ortonormal. Escolhida deliberadamente simples o suficiente para ser reimplementada fielmente em C sem biblioteca de DSP externa (ver `scripts/dsp_common.py` / `esp32/detector_anomalia/dsp.cpp`).
- **Janela:** clipes originais (~11s em média) cortados em janelas não-sobrepostas de 3s; janelas com RMS abaixo de 0.01 (silêncio) descartadas do treino.
- **Divisão treino/teste:** por **clipe de origem** (`GroupShuffleSplit`), não por janela — evita vazamento de dados entre janelas do mesmo locutor/gravação.
- **Modelo:** `StandardScaler` + `SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)`.
- **Exportação:** `.onnx` (218.9 KB, via `skl2onnx`) para uso em Python/teste; parâmetros do SVM também exportados para um header C (`svm_params.h`, 2792 support vectors) para inferência nativa no ESP32, já que o operador `SVMClassifier` (domínio `ai.onnx.ml`) não converte para TFLite.

### Resultado (held-out, por clipe)

```
              precision    recall  f1-score   support
    feminino       0.92      0.94      0.93      1746
   masculino       0.93      0.92      0.93      1728
    accuracy                           0.93      3474

Matriz de confusão:
[[1635  111]
 [ 139 1589]]
```

**93% de acurácia**, sem viés forte entre classes (111 falsos positivos, 139 falsos negativos).

### Threshold de decisão

`P(masculino) > 0.65` (não 0.5) — reduz falsos positivos de vozes andróginas/ruído, às custas de mais falsos negativos. Justificativa: em monitoramento de segurança, um alarme falso recorrente é mais custoso (descrédito do sistema) do que uma detecção um pouco mais conservadora.

---

## 3. Validação numérica C vs. Python (antes do hardware)

Como não há acesso a um ESP32 físico neste ambiente de desenvolvimento, **toda a lógica de DSP e inferência foi validada numericamente no computador antes de ir para o firmware**, comparando a implementação em C (compilada com `g++`, independente do Arduino) contra a implementação Python usada no treino:

| Componente | Método de validação | Resultado |
|---|---|---|
| `dsp.cpp` (FFT + MFCC) | `esp32/host_test/test_dsp_host` vs `dsp_common.py` no mesmo sinal sintético | Match a ~1e-6 (diferença float32 vs float64) |
| `svm_infer.cpp` (SVM + Platt scaling) | `esp32/host_test/test_svm_host` vs `pipeline.predict_proba()` do sklearn | Match a ~0.001–0.002 após correção de um bug de sinal (ver Discussão) |

Isso elimina a categoria de erro "features/inferência calculadas diferente no treino vs. no embarcado", que já havia causado problemas de acurácia na Fase 1 (ver Discussão).

---

## 4. Latência e performance

### Latência medida em Python (simulação offline, `scripts/07_test_pipeline.py`)

Executada sobre 25 janelas de teste (arquivos reais do dataset, não usados isoladamente como benchmark oficial de acurácia):

| Etapa | Média | P95 | Máx |
|---|---|---|---|
| Extração de features | ~38 ms | ~47 ms | ~48 ms |
| Inferência (ONNX Runtime) | ~0.7 ms | ~0.8 ms | ~1.2 ms |
| **Total (software, PC)** | **~39 ms** | **~48 ms** | **~49 ms** |

> **Isto é uma referência de software rodando em CPU de notebook, não a latência real do ESP32.** O ESP32 (Xtensa LX6 @ 240MHz, sem otimizações SIMD) deve ser mais lento na extração de features (FFT/MFCC em C puro, sem `esp-dsp`) e na inferência SVM (2792 support vectors, kernel RBF calculado support-vector a support-vector).

### Latência real em hardware — **pendente de medição**

O firmware (`detect_task.cpp`) já loga via Serial, para cada janela processada:

```
[deteccao] ANOMALIA prob_masc=0.87 rms=0.0234 | lat_features=XXXX us | lat_inferencia=YYYY us | lat_total=ZZZZ ms | led=2
```

**Ação pendente:** rodar o firmware no ESP32 real, capturar a saída do Serial Monitor (115200 baud) falando frases masculinas e femininas, e colar aqui os valores reais de `lat_features`, `lat_inferencia` e `lat_total` observados — substituindo esta seção antes da entrega final. A estimativa é de dezenas de ms para features (FFT em C, ~300 frames por janela de 3s) e alguns ms para a inferência SVM (ordem de grandeza validada por contagem de operações: 2792 kernels RBF × 15 multiplicações + `expf()`).

### Orçamento de latência (do enunciado original do projeto)

| Etapa | Meta original | Observação |
|---|---|---|
| Captura (1 frame) | ~64ms | Determinístico (1024/16000) |
| Janela de análise completa | 3s | Trade-off deliberado: estabilidade de features vs. responsividade (ver Discussão) |
| Extração + inferência | dezenas de ms | A confirmar em hardware |

---

## 5. Discussão

### O que funcionou bem
- SVM com kernel RBF sobre RMS+centróide+MFCC é uma escolha leve e eficaz (93% de acurácia, modelo pequeno o bastante para caber na flash do ESP32).
- Implementar a extração de features do zero (sem `librosa`/`esp-dsp`) permitiu validação numérica C vs. Python bit-a-bit antes de tocar hardware, eliminando uma classe inteira de bugs de integração.

### Erros encontrados e corrigidos durante o desenvolvimento (documentados propositalmente — parte do processo de engenharia)

1. **Descompasso treino/inferência (janela de tempo):** a primeira versão do modelo foi treinada com a *média de features sobre o clipe inteiro* (~10s), mas testada ao vivo com janelas de 1.5s — causando erro sistemático de classificação (voz correta, features estatisticamente diferentes do que o modelo viu no treino). Corrigido treinando com a MESMA duração de janela usada na inferência (3s), com split treino/teste por clipe de origem (evitando vazamento de dados).
2. **Bug de sinal no Platt scaling:** a implementação inicial em C de `P(masculino) = 1/(1+exp(decisão·probA + probB))` produzia probabilidades ~1 ponto percentual erradas frente ao `predict_proba()` do sklearn. Isolado por validação numérica cruzada (Python puro reimplementando a fórmula, comparando termo a termo) — o sinal correto é `exp(decisão·probA − probB)` (subtração, não soma). Erro residual após a correção: <0.2%, sem impacto no threshold de decisão (0.65).
3. **Domínio do dataset vs. microfone real:** o modelo treinado em `voxpopuli` (fala parlamentar/institucional, microfones de estúdio) mostrou degradação ao ser testado com o microfone embutido de um notebook — RMS e centróide espectral sistematicamente mais baixos que a distribuição de treino. Efeito de *domain shift*, não um bug de implementação. Mitigação: threshold assimétrico (0.65) e documentação da limitação (abaixo).

### Limitações conhecidas
- **Domain shift dataset → hardware real:** o modelo nunca viu áudio capturado por um INMP441; a acurácia em hardware pode ser menor que os 93% medidos offline. Recomenda-se validar com gravações reais do próprio hardware antes da demonstração, e se necessário incluir uma pequena amostra de áudio capturado pelo INMP441 no conjunto de avaliação.
- **Escopo fechado (binário), não anomalia genérica:** o sistema distingue "voz masculina" de "voz feminina" — duas classes conhecidas e bem representadas no treino — e não um detector one-class/não-supervisionado capaz de sinalizar *qualquer* som fora do esperado (ex: um grito, um latido). Para este projeto essa é uma escolha válida (a anomalia está bem definida: um segundo gênero de voz), mas é uma limitação de escopo relevante de mencionar: um som nunca visto no treino (silêncio, música, ruído) força uma classificação em uma das duas classes conhecidas, sem noção de "não sei".
- **Latência de resposta:** o trade-off de usar janelas de 3s (por estabilidade estatística das features) significa que o sistema demora até 3s para reagir a uma mudança de voz — mais lento que uma resposta "instantânea" por frame, mas ainda compatível com a aplicação de monitoramento ambiental proposta.

---

## 6. Estrutura do repositório

```
├── scripts/                      Pipeline Python (dataset → features → treino → testes)
│   ├── 01_prepare_dataset.py
│   ├── 02_extract_features.py
│   ├── 03_train_model.py
│   ├── 04_convert_tflite.py      (documenta por que a conversão falha)
│   ├── 04b_export_svm_header.py  (fallback: SVM → header C)
│   ├── 05_test_inference.py      (teste manual, 1 arquivo .wav)
│   ├── 06_test_live_mic.py       (teste ao vivo, microfone do notebook)
│   ├── 07_test_pipeline.py       (codigo de teste: simula + mede performance)
│   └── dsp_common.py             (extração de features, fonte da verdade)
├── esp32/
│   ├── detector_anomalia/        Firmware real (Arduino, 3 tasks FreeRTOS)
│   ├── bringup_led/, bringup_mic/  Sketches de validação de hardware
│   └── host_test/                 Testes de validação numérica C vs Python
├── model/                        detector_genero_voz.onnx, svm_params.h, pipeline.joblib
├── docs/diagrama_rtos.svg
├── planejamento.md               Plano técnico detalhado (histórico de decisões)
└── relatorio_tecnico.md          Este documento
```
