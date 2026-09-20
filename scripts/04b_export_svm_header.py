#!/usr/bin/env python3
"""
04b_export_svm_header.py -- Fallback para a conversao .onnx -> .tflite (ver
README.md): exporta o SVM como header C.

Extrai os parametros do pipeline (StandardScaler + SVC kernel RBF) treinado
em 03_train_model.py e gera um header C (model/svm_params.h) com tudo que e
necessario para reimplementar a inferencia do SVM manualmente em C/C++ no
ESP32 (Fase 2), sem depender de TFLite Micro nem de um runtime ONNX:

  1. normalizar a entrada: x_norm[i] = (x[i] - mean[i]) / scale[i]
  2. kernel RBF entre x_norm e cada vetor de suporte:
       K(x, sv) = exp(-gamma * ||x_norm - sv||^2)
  3. decisao: sum(dual_coef[i] * K(x_norm, sv[i])) + intercept
  4. probabilidade (Platt scaling, igual ao libsvm/sklearn):
       P(masculino) = 1 / (1 + exp(decision * probA - probB))
       (sinal de probB SUBTRAIDO -- verificado numericamente contra
       pipeline.predict_proba() do sklearn; ver esp32/detector_anomalia/svm_infer.cpp)
"""
import argparse
from pathlib import Path

import joblib
import numpy as np


def formatar_array_c(nome: str, valores: np.ndarray) -> str:
    valores = np.asarray(valores, dtype=np.float64).ravel()
    itens = ", ".join(f"{v:.10e}f" for v in valores)
    return f"static const float {nome}[{len(valores)}] = {{ {itens} }};"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=Path, default=Path("model/pipeline.joblib"))
    parser.add_argument("--out", type=Path, default=Path("model/svm_params.h"))
    args = parser.parse_args()

    if not args.pipeline.exists():
        raise SystemExit(f"{args.pipeline} nao encontrado. Rode 03_train_model.py antes.")

    pipeline = joblib.load(args.pipeline)
    scaler = pipeline.named_steps["scaler"]
    svm = pipeline.named_steps["svm"]

    if len(svm.classes_) != 2:
        raise SystemExit("Este exportador so suporta classificacao binaria.")

    n_features = scaler.mean_.shape[0]
    support_vectors = svm.support_vectors_          # (n_SV, n_features)
    dual_coef = svm.dual_coef_.ravel()               # (n_SV,)
    intercept = float(svm.intercept_[0])
    n_sv = support_vectors.shape[0]

    if isinstance(svm.gamma, str):
        # 'scale' (default) = 1 / (n_features * X.var()); sklearn guarda o
        # valor ja resolvido em _gamma apos o fit.
        gamma = float(svm._gamma)
    else:
        gamma = float(svm.gamma)

    prob_a = float(svm.probA_[0])
    prob_b = float(svm.probB_[0])

    linhas = [
        "// Gerado automaticamente por scripts/04b_export_svm_header.py",
        "// NAO editar a mao. Regenere a partir do pipeline treinado.",
        "//",
        "// Fallback para a inferencia SVM em C, ja que SVMClassifier (ai.onnx.ml)",
        "// nao converte para TFLite -- ver README.md.",
        "#pragma once",
        "",
        f"#define SVM_N_FEATURES {n_features}",
        f"#define SVM_N_SUPPORT_VECTORS {n_sv}",
        "",
        formatar_array_c("svm_scaler_mean", scaler.mean_),
        formatar_array_c("svm_scaler_scale", scaler.scale_),
        "",
        f"static const float svm_gamma = {gamma:.10e}f;",
        f"static const float svm_intercept = {intercept:.10e}f;",
        f"static const float svm_prob_a = {prob_a:.10e}f;",
        f"static const float svm_prob_b = {prob_b:.10e}f;",
        "",
        formatar_array_c("svm_dual_coef", dual_coef),
        "",
        "// support_vectors[SVM_N_SUPPORT_VECTORS][SVM_N_FEATURES], linha a linha",
        f"static const float svm_support_vectors[{n_sv}][{n_features}] = {{",
    ]
    for sv in support_vectors:
        itens = ", ".join(f"{v:.10e}f" for v in sv)
        linhas.append(f"    {{ {itens} }},")
    linhas.append("};")
    linhas.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(linhas))

    print(f"Header gerado: {args.out}")
    print(f"  n_features={n_features}  n_support_vectors={n_sv}  gamma={gamma:.6f}")


if __name__ == "__main__":
    main()
