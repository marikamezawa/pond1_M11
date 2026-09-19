#!/usr/bin/env python3
"""
06_test_live_mic.py -- Monitoramento ao vivo pelo microfone do notebook.

Nao grava nada em disco: captura audio continuamente do microfone via
`arecord` (ALSA, ja vem no Linux -- sem precisar instalar nada), corta em
janelas curtas, extrai as mesmas features do pipeline (RMS + Spectral
Centroid + 13 MFCCs) e roda a inferencia do modelo ONNX em cada janela,
imprimindo o resultado no terminal em tempo real -- similar ao que a Task 3
vai fazer no ESP32 (Fase 2), so que aqui em Python.

Simula tambem a logica de LED de 3 estados que sera usada na Task 3:
  - Voz masculina  -> LED VERMELHO aceso
  - Voz feminina   -> LED VERDE aceso
  - Silencio       -> os dois apagados
Com retencao (--hold): o LED da ultima deteccao de voz continua aceso por
alguns segundos mesmo que uma janela intermediaria caia em silencio (evita
"piscar" em pausas curtas de respiracao no meio de uma fala continua). Isso
e so a camada de atuacao (o que fazer com o LED) -- a logica de deteccao de
anomalia (SVM + threshold) e a mesma, nao muda em nada.

Uso:
    python scripts/06_test_live_mic.py
    python scripts/06_test_live_mic.py --window 3.0 --threshold 0.65 --hold 1.5
    python scripts/06_test_live_mic.py --device plughw:0,6   # forcar dispositivo

Ctrl+C para parar.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dsp_common import SAMPLE_RATE, WINDOW_SEC, extrair_features as _extrair_features

THRESHOLD_PADRAO = 0.65
SILENCIO_RMS = 0.005


def extrair_features(y: np.ndarray, sr: int = SAMPLE_RATE):
    rms = float(np.sqrt(np.mean(y ** 2)))
    return _extrair_features(y, sr), rms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("model/detector_genero_voz.onnx"))
    parser.add_argument("--window", type=float, default=WINDOW_SEC, help="Duracao de cada janela analisada, em segundos")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_PADRAO)
    parser.add_argument("--device", default="default", help="Dispositivo ALSA (ver `arecord -l`), ex: plughw:0,6")
    parser.add_argument("--silence-rms", type=float, default=SILENCIO_RMS,
                         help="RMS abaixo do qual a janela e tratada como silencio")
    parser.add_argument("--hold", type=float, default=1.5,
                         help="Segundos que o LED continua aceso apos a ultima deteccao de voz")
    args = parser.parse_args()

    if shutil.which("arecord") is None:
        sys.exit("`arecord` nao encontrado. Instale alsa-utils: sudo apt install alsa-utils")
    if not args.model.exists():
        sys.exit(f"Modelo nao encontrado: {args.model} (rode 03_train_model.py antes)")

    sess = ort.InferenceSession(str(args.model))
    entrada_nome = sess.get_inputs()[0].name

    # dsp_common nao usa mais librosa/numba para features -- so numpy puro,
    # entao nao ha mais custo de compilacao JIT na primeira chamada.

    window_samples = int(args.window * SAMPLE_RATE)
    bytes_por_janela = window_samples * 2  # 16 bits = 2 bytes/amostra

    cmd = [
        "arecord", "-D", args.device,
        "-f", "S16_LE", "-c", "1", "-r", str(SAMPLE_RATE),
        "-t", "raw", "-q",
    ]
    print(f"Escutando o microfone ({args.device}) em janelas de {args.window:.1f}s "
          f"(retencao do LED: {args.hold:.1f}s). Ctrl+C para parar.\n")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    led_atual = "APAGADO"       # "VERDE" | "VERMELHO" | "APAGADO"
    led_expira_em = 0.0         # timestamp (time.monotonic()) ate quando o LED atual deve continuar aceso
    try:
        while True:
            raw = proc.stdout.read(bytes_por_janela)
            if len(raw) < bytes_por_janela:
                break  # arecord encerrou ou pipe fechou

            amostras = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            features, rms = extrair_features(amostras)
            agora = time.monotonic()
            ts = time.strftime("%H:%M:%S")

            if rms >= args.silence_rms:
                saida = sess.run(None, {entrada_nome: features.reshape(1, -1)})
                prob_masculino = float(saida[1][0][1])

                if prob_masculino > args.threshold:
                    led_atual = "VERMELHO"
                    print(f"[{ts}] [ANOMALIA] Voz masculina detectada ({prob_masculino:.2f}, rms={rms:.4f}) -> LED VERMELHO")
                else:
                    led_atual = "VERDE"
                    print(f"[{ts}] [NORMAL] Voz feminina detectada ({1 - prob_masculino:.2f}, rms={rms:.4f}) -> LED VERDE")
                led_expira_em = agora + args.hold
            elif agora < led_expira_em:
                # silencio momentaneo, mas dentro da janela de retencao: mantem o LED da ultima deteccao
                print(f"[{ts}] silencio (rms={rms:.4f}) -> LED {led_atual} (retido)")
            else:
                led_atual = "APAGADO"
                print(f"[{ts}] silencio (rms={rms:.4f}) -> LEDS APAGADOS")
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuario.")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
