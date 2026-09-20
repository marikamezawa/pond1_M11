#!/usr/bin/env python3
"""
09_extrair_features_hardware.py -- extrai features dos audios gravados pelo
INMP441 real (data/hardware/fem_*.wav, masc_*.wav; ver 08_capturar_audio_hardware.py)
e salva X_hw.npy / y_hw.npy / groups_hw.npy / holdout_hw.npy em data/.

- Descarta os primeiros TRIM_SEC de cada arquivo (apito de partida do mic).
- Janelas de WINDOW_SEC com passo HOP_SEC (sobrepostas, so pra ter mais amostras
  a partir de gravacoes curtas -- todas do mesmo arquivo ficam no mesmo grupo).
- holdout_hw = 1 marca arquivos reservados para AVALIAR (nao treinar) o modelo
  no hardware real: masc_06..10 (segundo locutor masculino, nunca visto no
  treino) e fem_9, 10, 11 e 22.
"""
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp_common import SAMPLE_RATE, WINDOW_SAMPLES, calcular_rms, extrair_features

HW_DIR = Path("data/hardware")
OUT_DIR = Path("data")
TRIM_SEC = 0.5
HOP_SEC = 0.5
SILENCIO_RMS = 0.01

HOLDOUT = {("masc", n) for n in range(6, 11)} | {("fem", n) for n in (9, 10, 11, 22)}


def main():
    arquivos = []
    for f in HW_DIR.glob("*.wav"):
        m = re.match(r"(fem|masc)_(\d+)", f.stem)
        if m:
            arquivos.append((m.group(1), int(m.group(2)), f))
    arquivos.sort()

    trim, hop = int(TRIM_SEC * SAMPLE_RATE), int(HOP_SEC * SAMPLE_RATE)
    X, y, groups, holdout = [], [], [], []
    for gid, (tipo, n, f) in enumerate(arquivos):
        audio, sr = sf.read(f)
        assert sr == SAMPLE_RATE, f"{f}: sr={sr}"
        audio = audio[trim:].astype(np.float32)
        for s in range(0, len(audio) - WINDOW_SAMPLES + 1, hop):
            w = audio[s:s + WINDOW_SAMPLES]
            if calcular_rms(w) < SILENCIO_RMS:
                continue
            X.append(extrair_features(w, SAMPLE_RATE))
            y.append(1 if tipo == "masc" else 0)
            groups.append(100000 + gid)
            holdout.append(1 if (tipo, n) in HOLDOUT else 0)

    # Janelas de fala PARCIAL: no detector real a janela de 3s costuma pegar so
    # um pedaco de fala (inicio/fim de frase) no meio de silencio. Sem esses
    # exemplos o modelo classificava esses casos como masculino (~0.89).
    # Monta cada janela com um trecho de fala + silencio real do mic (sil_*.wav).
    sil = []
    for f in sorted(HW_DIR.glob("sil_*.wav")):
        s, _ = sf.read(f)
        sil.append(s[trim:].astype(np.float32))
    n_sinteticas = 0
    if sil:
        sil = np.concatenate(sil)
        rng = np.random.default_rng(0)
        for gid, (tipo, n, f) in enumerate(arquivos):
            audio, _ = sf.read(f)
            audio = audio[trim:].astype(np.float32)
            for dur in (0.75, 1.5, 2.25):
                for pos in ("ini", "meio", "fim"):
                    for _ in range(2):
                        L = int(dur * SAMPLE_RATE)
                        ini = int(rng.integers(0, len(audio) - L))
                        fala = audio[ini:ini + L]
                        o = int(rng.integers(0, len(sil) - WINDOW_SAMPLES))
                        w = sil[o:o + WINDOW_SAMPLES].copy()
                        p = {"ini": 0, "meio": (WINDOW_SAMPLES - L) // 2, "fim": WINDOW_SAMPLES - L}[pos]
                        w[p:p + L] = fala
                        if calcular_rms(w) < 0.015:
                            continue
                        X.append(extrair_features(w, SAMPLE_RATE))
                        y.append(1 if tipo == "masc" else 0)
                        groups.append(100000 + gid)
                        holdout.append(1 if (tipo, n) in HOLDOUT else 0)
                        n_sinteticas += 1
    else:
        print("[aviso] nenhum sil_*.wav em data/hardware -- sem janelas de fala parcial.")
    print(f"Janelas sinteticas (fala parcial + silencio real): {n_sinteticas}")

    X = np.array(X, dtype=np.float32)
    np.save(OUT_DIR / "X_hw.npy", X)
    np.save(OUT_DIR / "y_hw.npy", np.array(y, dtype=np.int64))
    np.save(OUT_DIR / "groups_hw.npy", np.array(groups, dtype=np.int64))
    np.save(OUT_DIR / "holdout_hw.npy", np.array(holdout, dtype=np.int64))
    y, holdout = np.array(y), np.array(holdout)
    print(f"{len(arquivos)} arquivos -> {X.shape[0]} janelas "
          f"(fem={int((y==0).sum())}, masc={int((y==1).sum())}, holdout={int(holdout.sum())})")


if __name__ == "__main__":
    main()
