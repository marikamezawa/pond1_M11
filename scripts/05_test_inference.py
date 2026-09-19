#!/usr/bin/env python3
"""
05_test_inference.py -- Recebe um arquivo .wav e classifica a voz usando o
modelo ONNX treinado.

Uso:
    python scripts/05_test_inference.py caminho/para/audio.wav
    python scripts/05_test_inference.py caminho/para/audio.wav --threshold 0.7

Saida:
    [NORMAL] Voz feminina detectada (0.92)
    [ANOMALIA] Voz masculina detectada (0.87)
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort

from dsp_common import SAMPLE_RATE, extrair_features

THRESHOLD_PADRAO = 0.65


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav_path", type=Path, help="Arquivo .wav a classificar")
    parser.add_argument("--model", type=Path, default=Path("model/detector_genero_voz.onnx"))
    parser.add_argument("--threshold", type=float, default=THRESHOLD_PADRAO,
                         help="Limiar de P(masculino) para disparar anomalia")
    args = parser.parse_args()

    if not args.wav_path.exists():
        sys.exit(f"Arquivo nao encontrado: {args.wav_path}")
    if not args.model.exists():
        sys.exit(f"Modelo nao encontrado: {args.model} (rode 03_train_model.py antes)")

    sess = ort.InferenceSession(str(args.model))

    y, _ = librosa.load(str(args.wav_path), sr=SAMPLE_RATE)
    features = extrair_features(y, SAMPLE_RATE).reshape(1, -1)
    entrada_nome = sess.get_inputs()[0].name
    saida = sess.run(None, {entrada_nome: features})

    # saida[0] = label predito (0/1), saida[1] = probabilidades (N, 2) com zipmap=False
    probs = saida[1][0]
    prob_masculino = float(probs[1])

    if prob_masculino > args.threshold:
        print(f"[ANOMALIA] Voz masculina detectada ({prob_masculino:.2f})")
    else:
        print(f"[NORMAL] Voz feminina detectada ({1 - prob_masculino:.2f})")


if __name__ == "__main__":
    main()
