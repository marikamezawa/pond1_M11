#!/usr/bin/env python3
"""
01_prepare_dataset.py -- Coleta um dataset de voz com rotulo de genero para
o detector de genero de voz (Fase 1), via streaming do Hugging Face Hub.

O Mozilla Common Voice foi descartado: o site commonvoice.mozilla.org nao
disponibiliza mais os pacotes de audio para download, e os espelhos oficiais
no Hugging Face (mozilla-foundation/common_voice_*) tambem estao vazios --
so tem README, sem arquivos de audio (verificado em 2026-09-17).

Dataset usado agora: facebook/voxpopuli -- discursos do Parlamento Europeu,
com campo "gender" nativo ("male"/"female"), audio ja em 16 kHz, distribuido
em parquet (sem loading script customizado, mais estavel via streaming).
https://huggingface.co/datasets/facebook/voxpopuli

Requisitos: um token de acesso (`huggingface-cli login` ou variavel de
ambiente HF_TOKEN) -- alguns configs/idiomas podem exigir autenticacao.

O dataset e consumido com streaming=True (nao baixa o dataset inteiro): o
script para assim que coletar N clipes de cada classe (--n-per-class,
padrao 3000+3000), filtrando pelo campo "gender".

Saida: data/dataset.csv com colunas ["path", "label"], label em
{"male", "female"}, classes balanceadas.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

GENDER_MAP = {
    "male": "male",
    "male_masculine": "male",
    "masculine": "male",
    "female": "female",
    "female_feminine": "female",
    "feminine": "female",
}


def normalizar_genero(valor) -> str | None:
    if valor is None:
        return None
    return GENDER_MAP.get(str(valor).strip().lower())


def preparar_via_hf(dataset_name: str, language: str, n_per_class: int,
                     out_dir: Path, seed: int) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "Pacote 'datasets' nao instalado. Rode: pip install datasets huggingface_hub"
        )
    import soundfile as sf

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print(f"Conectando ao Hugging Face Hub: {dataset_name} ({language}) [streaming]...")
    ds = load_dataset(dataset_name, language, split="train", streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    counts = {"male": 0, "female": 0}
    linhas = []

    for row in ds:
        if counts["male"] >= n_per_class and counts["female"] >= n_per_class:
            break

        genero = normalizar_genero(row.get("gender"))
        if genero is None or counts[genero] >= n_per_class:
            continue

        audio = row.get("audio")
        if not audio or audio.get("array") is None:
            continue

        clip_id = f"{genero}_{counts[genero]:05d}"
        caminho_wav = clips_dir / f"{clip_id}.wav"
        sf.write(caminho_wav, audio["array"], audio["sampling_rate"])

        linhas.append({"path": str(caminho_wav.resolve()), "label": genero})
        counts[genero] += 1

        if sum(counts.values()) % 200 == 0:
            print(f"  progresso: male={counts['male']} female={counts['female']}")

    if counts["male"] == 0 or counts["female"] == 0:
        sys.exit("Nao foi possivel coletar amostras das duas classes. Verifique o token/idioma.")

    print(f"Coleta finalizada: male={counts['male']} female={counts['female']}")
    return pd.DataFrame(linhas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-dataset", default="facebook/voxpopuli")
    parser.add_argument("--language", default="en", help="Codigo de idioma/config do dataset (ex: en, pt, de)")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-per-class", type=int, default=3000, help="Numero de clipes por classe (balanceado)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = preparar_via_hf(args.hf_dataset, args.language, args.n_per_class, args.out_dir, args.seed)

    csv_path = args.out_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDataset salvo em {csv_path} ({len(df)} linhas)")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
