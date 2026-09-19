# Detector de Anomalias Acústicas

Sistema embarcado (ESP32 + INMP441 + FreeRTOS) que monitora um ambiente esperando apenas vozes femininas; qualquer **voz masculina** é tratada como anomalia (LED vermelho). Ver [planejamento.md](planejamento.md) para o histórico completo de decisões técnicas e [relatorio_tecnico.md](relatorio_tecnico.md) para arquitetura, resultados e discussão.

## Fase 1 — Modelo (Python, sem hardware)

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. `scripts/01_prepare_dataset.py` — coleta `facebook/voxpopuli` via streaming do Hugging Face Hub (requer `HF_TOKEN`). O Mozilla Common Voice foi descartado (site e espelhos no HF sem áudio disponível). Gera `data/dataset.csv`.
2. `scripts/02_extract_features.py` — corta cada clipe em janelas de 3s e extrai RMS + Spectral Centroid + 13 MFCCs (`dsp_common.py`, implementação própria — sem `librosa.feature.mfcc`, pensada para ser espelhada em C). Gera `data/X.npy`, `data/y.npy`, `data/groups.npy`.
3. `scripts/03_train_model.py` — treina `StandardScaler + SVC(kernel='rbf')` com split treino/teste **por clipe de origem** (evita vazamento de dados), exporta `model/detector_genero_voz.onnx` e `model/pipeline.joblib`. Acurácia atual: **93%**.
4. `scripts/04_convert_tflite.py` — tenta converter para `.tflite`; espera-se que falhe (SVMClassifier é `ai.onnx.ml`, não suportado — ver docstring).
   - `scripts/04b_export_svm_header.py` — fallback: exporta os parâmetros do SVM para `model/svm_params.h`, usado na inferência em C no ESP32.
5. `scripts/05_test_inference.py caminho/audio.wav` — classifica um `.wav`.
6. `scripts/06_test_live_mic.py` — monitoramento ao vivo pelo microfone (via `arecord`), com a mesma lógica de LED de 3 estados + retenção usada no firmware.
7. `scripts/07_test_pipeline.py` — **código de teste**: simula detecção sobre um conjunto de áudios rotulados e mede latência (features/inferência) e acurácia funcional do pipeline completo.

## Fase 2 — Firmware ESP32 (Arduino framework)

**Hardware:** ESP32-WROOM-32U + INMP441 + LED verde (GPIO2) + LED vermelho (GPIO32). Pinagem completa e notas de montagem em [planejamento.md](planejamento.md#montagem-física-confirmada).

- `esp32/bringup_led/`, `esp32/bringup_mic/` — sketches de validação de hardware (piscar LEDs / ler RMS do mic no Serial Monitor).
- `esp32/detector_anomalia/` — **firmware real**: 3 tasks FreeRTOS (mutex + semáforo + fila), extração de features em C (`dsp.cpp`, sem dependência de `esp-dsp`/TFLite) e inferência do SVM em C (`svm_infer.cpp`, via `svm_params.h`). Ver `relatorio_tecnico.md` para detalhes da arquitetura.
- `esp32/host_test/` — testes de validação numérica (compilam com `g++` no computador, comparam a saída de `dsp.cpp`/`svm_infer.cpp` contra `dsp_common.py`/`sklearn`, sem precisar do hardware).

### Pendente antes da entrega

- [ ] Compilar e gravar `esp32/detector_anomalia/` no ESP32 real, capturar latência real via Serial Monitor e preencher a seção correspondente em `relatorio_tecnico.md`
- [ ] Validar acurácia com áudio capturado pelo INMP441 real (domain shift esperado — dataset é `voxpopuli`, não gravações do próprio hardware)
- [ ] Commit do repositório
