#!/usr/bin/env python3
"""
07_test_pipeline.py -- "Codigo de teste" exigido no enunciado: simula
deteccao de anomalias sobre audios reais rotulados e mede performance
(latencia de extracao de features + inferencia) e acuracia funcional do
pipeline completo, end-to-end, do jeito que ele vai rodar (janelas de 3s,
mesma extracao de features de dsp_common.py, mesmo modelo .onnx).

NAO substitui a avaliacao oficial de acuracia (essa vem do split
treino/teste por clipe em 03_train_model.py, que e o numero reportado no
relatorio). Este script e um teste funcional/smoke-test do pipeline
completo + medicao de latencia, para detectar regressao e documentar
performance.

Uso:
    python scripts/07_test_pipeline.py --n-por-classe 15
    python scripts/07_test_pipeline.py --manifest meu_teste.csv   # path,label
"""
import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd

from dsp_common import SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_SEC, extrair_features

THRESHOLD_ANOMALIA = 0.65
SILENCIO_RMS = 0.01


def montar_manifest_padrao(dataset_csv: Path, n_por_classe: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    partes = []
    for label, grupo in df.groupby("label"):
        partes.append(grupo.sample(n=min(n_por_classe, len(grupo)), random_state=seed))
    return pd.concat(partes, ignore_index=True)


def testar_arquivo(sess, entrada_nome, caminho_audio: str, label_real: str, threshold: float):
    """Corta o audio em janelas de WINDOW_SEC, roda o pipeline completo em
    cada uma, e retorna a lista de resultados por janela."""
    y, _ = librosa.load(caminho_audio, sr=SAMPLE_RATE)
    n_janelas = len(y) // WINDOW_SAMPLES

    resultados = []
    for i in range(n_janelas):
        trecho = y[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES]
        rms = float(np.sqrt(np.mean(trecho ** 2)))

        t0 = time.perf_counter()
        features = extrair_features(trecho, SAMPLE_RATE)
        t1 = time.perf_counter()

        if rms < SILENCIO_RMS:
            resultados.append({
                "arquivo": caminho_audio, "janela": i, "label_real": label_real,
                "silencio": True, "anomalia_prevista": None,
                "prob_masculino": None,
                "lat_features_ms": (t1 - t0) * 1000, "lat_inferencia_ms": 0.0,
            })
            continue

        saida = sess.run(None, {entrada_nome: features.reshape(1, -1)})
        t2 = time.perf_counter()
        prob_masculino = float(saida[1][0][1])
        anomalia_prevista = prob_masculino > threshold

        resultados.append({
            "arquivo": caminho_audio, "janela": i, "label_real": label_real,
            "silencio": False, "anomalia_prevista": anomalia_prevista,
            "prob_masculino": prob_masculino,
            "lat_features_ms": (t1 - t0) * 1000, "lat_inferencia_ms": (t2 - t1) * 1000,
        })
    return resultados


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("model/detector_genero_voz.onnx"))
    parser.add_argument("--manifest", type=Path, default=None,
                         help="CSV com colunas path,label. Se omitido, amostra do data/dataset.csv")
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--n-por-classe", type=int, default=15,
                         help="Quantos arquivos por classe amostrar quando --manifest nao for passado")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_ANOMALIA)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-json", type=Path, default=Path("data/test_pipeline_resultados.json"))
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"{args.model} nao encontrado. Rode 03_train_model.py antes.")

    if args.manifest is not None:
        manifest = pd.read_csv(args.manifest)
    else:
        if not args.dataset_csv.exists():
            raise SystemExit(f"{args.dataset_csv} nao encontrado e --manifest nao foi passado.")
        manifest = montar_manifest_padrao(args.dataset_csv, args.n_por_classe, args.seed)

    sess = ort.InferenceSession(str(args.model))
    entrada_nome = sess.get_inputs()[0].name

    todos_resultados = []
    print(f"Testando {len(manifest)} arquivos (janelas de {WINDOW_SEC}s cada)...\n")
    for _, row in manifest.iterrows():
        resultados = testar_arquivo(sess, entrada_nome, row["path"], row["label"], args.threshold)
        todos_resultados.extend(resultados)

    df = pd.DataFrame(todos_resultados)
    df_com_voz = df[~df["silencio"]].copy()
    df_com_voz["genero_real_masc"] = df_com_voz["label_real"] == "male"
    df_com_voz["acerto"] = df_com_voz["anomalia_prevista"] == df_com_voz["genero_real_masc"]

    acuracia = df_com_voz["acerto"].mean() * 100
    lat_features = df_com_voz["lat_features_ms"]
    lat_inferencia = df_com_voz["lat_inferencia_ms"]
    lat_total = lat_features + lat_inferencia

    print("=" * 60)
    print(f"Janelas totais: {len(df)}  (com voz: {len(df_com_voz)}, silencio: {(df['silencio']).sum()})")
    print(f"Acuracia (janelas com voz): {acuracia:.1f}%")
    print()
    print("Latencia por etapa (ms):")
    print(f"  features:   media={lat_features.mean():.2f}  p95={lat_features.quantile(0.95):.2f}  max={lat_features.max():.2f}")
    print(f"  inferencia: media={lat_inferencia.mean():.2f}  p95={lat_inferencia.quantile(0.95):.2f}  max={lat_inferencia.max():.2f}")
    print(f"  total:      media={lat_total.mean():.2f}  p95={lat_total.quantile(0.95):.2f}  max={lat_total.max():.2f}")
    print("=" * 60)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    resumo = {
        "n_janelas_total": len(df),
        "n_janelas_com_voz": len(df_com_voz),
        "n_janelas_silencio": int(df["silencio"].sum()),
        "acuracia_pct": float(acuracia),
        "latencia_features_ms": {"media": float(lat_features.mean()), "p95": float(lat_features.quantile(0.95)), "max": float(lat_features.max())},
        "latencia_inferencia_ms": {"media": float(lat_inferencia.mean()), "p95": float(lat_inferencia.quantile(0.95)), "max": float(lat_inferencia.max())},
        "latencia_total_ms": {"media": float(lat_total.mean()), "p95": float(lat_total.quantile(0.95)), "max": float(lat_total.max())},
    }
    with open(args.out_json, "w") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"\nResumo salvo em {args.out_json}")


if __name__ == "__main__":
    main()
