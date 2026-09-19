#!/usr/bin/env python3
"""
02_extract_features.py -- Extrai RMS + Spectral Centroid + 13 MFCCs de cada
clipe listado em data/dataset.csv e salva X.npy / y.npy prontos para treino.

Cada clipe original (~10s em media, no caso do voxpopuli) e cortado em
janelas nao-sobrepostas de WINDOW_SEC (dsp_common.py, hoje 3s) e uma
feature e extraida por janela -- essa e a MESMA duracao usada na inferencia
ao vivo (06) e no ESP32 (Task 2 acumula WINDOW_SAMPLES antes de calcular um
FeatureVector), para nao repetir o descompasso treino/inferencia que ja
causou erro sistematico quando testamos janelas mais curtas (1.5s) ao vivo.

Janelas com RMS abaixo de SILENCIO_RMS sao descartadas (o pipeline de
producao tambem nao roda inferencia em silencio).

Vetor de features (15 dimensoes): [rms, centroid, mfcc_0..mfcc_12]
Label: 1 = masculino (anomalia), 0 = feminino (normal)

Saida:
  X.npy      -- (N, 15) features por janela
  y.npy      -- (N,) label por janela
  groups.npy -- (N,) indice do clipe de origem de cada janela -- usado no
                script 03 para dividir treino/teste por CLIPE (nao por
                janela), evitando vazamento de dados entre janelas do
                mesmo audio/locutor.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from dsp_common import SAMPLE_RATE, WINDOW_SAMPLES, extrair_features

SILENCIO_RMS = 0.01


def processar_clipe(caminho_audio: str):
    try:
        y, _ = librosa.load(caminho_audio, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"  [aviso] falha ao carregar {caminho_audio}: {e}")
        return []

    if y.size < WINDOW_SAMPLES:
        return []

    n_janelas = len(y) // WINDOW_SAMPLES
    resultados = []
    for i in range(n_janelas):
        trecho = y[i * WINDOW_SAMPLES: (i + 1) * WINDOW_SAMPLES]
        rms = float(np.sqrt(np.mean(trecho ** 2)))
        if rms < SILENCIO_RMS:
            continue
        resultados.append(extrair_features(trecho, SAMPLE_RATE))
    return resultados


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} nao encontrado. Rode 01_prepare_dataset.py antes.")

    df = pd.read_csv(args.csv)

    X, y, groups = [], [], []
    for clip_idx, row in tqdm(df.iterrows(), total=len(df), desc="Extraindo features"):
        label = 1 if row["label"] == "male" else 0
        for features in processar_clipe(row["path"]):
            X.append(features)
            y.append(label)
            groups.append(clip_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "X.npy", X)
    np.save(args.out_dir / "y.npy", y)
    np.save(args.out_dir / "groups.npy", groups)

    print(f"\nX.npy: {X.shape}  y.npy: {y.shape}  groups.npy: {groups.shape}")
    print(f"Janelas de {WINDOW_SAMPLES/SAMPLE_RATE:.1f}s a partir de {df.shape[0]} clipes originais")
    print(f"Classe 0 (feminino): {(y == 0).sum()}  Classe 1 (masculino): {(y == 1).sum()}")
    print(f"Salvo em {args.out_dir}/")


if __name__ == "__main__":
    main()
