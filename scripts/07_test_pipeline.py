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

Modo --cenario: monta uma linha do tempo com audios REAIS do INMP441
(data/hardware: silencio, voz feminina e voz masculina alternadas) e passa
por ela a mesma logica de decisao do firmware (janela de 3s reavaliada a
cada 1s, gate de silencio, limiar, debounce simetrico, LED que pisca 2x uma vez por episodio). Reporta falsos alarmes, deteccoes e tempo ate o alarme. Para uma
avaliacao honesta, use um modelo treinado SEM esses arquivos (holdout):
    python scripts/03_train_model.py --hardware --out-dir model/holdout
    python scripts/07_test_pipeline.py --cenario --model model/holdout/detector_genero_voz.onnx

Uso:
    python scripts/07_test_pipeline.py --n-por-classe 15
    python scripts/07_test_pipeline.py --manifest meu_teste.csv   # path,label
    python scripts/07_test_pipeline.py --cenario
"""
import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd

import soundfile as sf

from dsp_common import SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_SEC, calcular_rms, extrair_features

THRESHOLD_ANOMALIA = 0.75
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


# ---- Simulacao do firmware (mesmos valores de esp32/detector_anomalia/config.h) ----
FW_THRESHOLD_SILENCIO = 0.03
FW_PISO_JANELAS = 60
FW_FATOR_PISO = 2.5
FW_PISO_MAX = 0.05
FW_ANOMALIA_JANELAS_SEGUIDAS = 2
FW_NORMAL_JANELAS_SEGUIDAS = 3
FW_HOP_S = 1.0
TRIM_SEC = 0.5

# Arquivos reservados (holdout) em 09_extrair_features_hardware.py
CENARIO = ["sil_02", "fem_22", "sil_03", "masc_06", "sil_02", "masc_07", "fem_9", "masc_08",
           "sil_03", "masc_09", "fem_10", "masc_10", "fem_11", "masc_15", "sil_01"]


def simular_firmware(sess, entrada_nome, y, threshold):
    """Reproduz a logica da Task 3 sobre janelas de 3s a cada 1s (mesmos valores
    de config.h). Retorna lista de (t_fim_s, prob, estado, pisca): estado em
    {silencio, normal, incerto, anomalia}; pisca em {None, "verde", "vermelho"}.
    O LED pisca (2 piscadas) uma vez por episodio de fala (ver detect_task.cpp)."""
    saida = []
    masc, norm, episodio = 0, 0, "nenhum"
    historico = []
    passo, janela = int(FW_HOP_S * SAMPLE_RATE), WINDOW_SAMPLES
    for fim in range(janela, len(y) + 1, passo):
        w = y[fim - janela:fim]
        t = fim / SAMPLE_RATE
        rms = calcular_rms(w)
        # gate adaptativo (igual ao firmware): max(minimo fixo, fator * piso de ruido)
        historico = (historico + [rms])[-FW_PISO_JANELAS:]
        gate = max(FW_THRESHOLD_SILENCIO, FW_FATOR_PISO * min(min(historico), FW_PISO_MAX))
        if rms < gate:
            masc = norm = 0
            episodio = "nenhum"
            saida.append((t, None, "silencio", None))
            continue
        prob = float(sess.run(None, {entrada_nome: extrair_features(w, SAMPLE_RATE).reshape(1, -1)})[1][0][1])
        if prob > threshold:
            masc, norm = masc + 1, 0
        else:
            norm, masc = norm + 1, 0
        pisca = None
        if masc >= FW_ANOMALIA_JANELAS_SEGUIDAS and episodio != "anomalia":
            pisca, episodio = "vermelho", "anomalia"
        elif norm >= FW_NORMAL_JANELAS_SEGUIDAS:
            if episodio == "nenhum":
                pisca, episodio = "verde", "normal"
            elif episodio == "anomalia":
                episodio = "normal"   # rearma o vermelho sem piscar verde
        estado = "anomalia" if masc >= FW_ANOMALIA_JANELAS_SEGUIDAS else ("incerto" if masc > 0 else "normal")
        saida.append((t, prob, estado, pisca))
    return saida


def rodar_cenario(args):
    sess = ort.InferenceSession(str(args.model))
    entrada_nome = sess.get_inputs()[0].name
    trechos, pos = [], 0
    partes = []
    for nome in CENARIO:
        y, sr = sf.read(Path("data/hardware") / f"{nome}.wav")
        assert sr == SAMPLE_RATE
        y = y[int(TRIM_SEC * SAMPLE_RATE):].astype(np.float32)
        tipo = "masc" if nome.startswith("masc") else ("fem" if nome.startswith("fem") else "sil")
        trechos.append((nome, tipo, pos / SAMPLE_RATE, (pos + len(y)) / SAMPLE_RATE))
        pos += len(y)
        partes.append(y)
    audio = np.concatenate(partes)
    sim = simular_firmware(sess, entrada_nome, audio, args.threshold)

    print(f"Cenario: {len(CENARIO)} trechos, {len(audio)/SAMPLE_RATE:.0f}s de audio real do INMP441; "
          f"modelo {args.model}\n")
    print(f"{'trecho':<10} {'tipo':<5} {'inicio':>7} {'fim':>7}  resultado")
    n_masc = n_masc_det = n_fa_seg = n_nao_masc = 0
    atrasos = []
    for nome, tipo, ini, fim in trechos:
        if tipo == "masc":
            n_masc += 1
            reds = [t for (t, _, e, pi) in sim if pi == "vermelho" and ini <= t <= fim + FW_HOP_S]
            verdes_errados = [t for (t, _, e, pi) in sim if pi == "verde" and ini <= t <= fim + FW_HOP_S]
            if reds:
                n_masc_det += 1
                atrasos.append(reds[0] - ini)
                res = f"DETECTADA (pisca vermelho {reds[0]-ini:.0f}s apos o inicio)" + (
                    f"; pisca VERDE antes/junto ({len(verdes_errados)}x)" if verdes_errados else "")
            else:
                res = "NAO detectada"
        else:
            # so avalia janelas totalmente dentro do trecho (sem audio do trecho anterior)
            dentro = [(t, pi) for (t, _, e, pi) in sim if t - WINDOW_SEC >= ini and t <= fim]
            n_nao_masc += 1
            fa = [t for (t, pi) in dentro if pi == "vermelho"]
            if fa:
                n_fa_seg += 1
                res = f"FALSO ALARME (pisca vermelho em {len(fa)} de {len(dentro)} janelas)"
            else:
                res = f"ok (0 piscas vermelhos em {len(dentro)} janelas)"
        print(f"{nome:<10} {tipo:<5} {ini:7.1f} {fim:7.1f}  {res}")

    print("\n" + "=" * 60)
    print(f"Vozes masculinas detectadas: {n_masc_det}/{n_masc}")
    if atrasos:
        print(f"Tempo ate o alarme: media={np.mean(atrasos):.1f}s  max={np.max(atrasos):.1f}s")
    print(f"Trechos femininos/silencio com falso alarme: {n_fa_seg}/{n_nao_masc}")
    print("=" * 60)
    return {"masc_detectadas": n_masc_det, "masc_total": n_masc,
            "tempo_ate_alarme_s": [float(a) for a in atrasos],
            "trechos_nao_masc_com_falso_alarme": n_fa_seg, "trechos_nao_masc_total": n_nao_masc}


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
    parser.add_argument("--cenario", action="store_true",
                         help="Simula o firmware sobre uma linha do tempo de audios reais do INMP441")
    parser.add_argument("--out-json", type=Path, default=Path("data/test_pipeline_resultados.json"))
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"{args.model} nao encontrado. Rode 03_train_model.py antes.")

    if args.cenario:
        rodar_cenario(args)
        return

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
