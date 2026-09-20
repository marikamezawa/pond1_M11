#!/usr/bin/env python3
"""
03_train_model.py -- Treina o SVM (kernel RBF) em cima de X.npy/y.npy,
opcionalmente ajusta hiperparametros com GridSearchCV, avalia e exporta:

  model/detector_genero_voz.onnx  -- para inferencia (Python, ESP32 via ONNX Runtime)
  model/pipeline.joblib           -- pipeline sklearn bruto (usado pelo script
                                      04b_export_svm_header.py para gerar o
                                      fallback em C, ja que SVMClassifier nao
                                      converte para TFLite -- ver README.md)
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("model"))
    parser.add_argument("--grid-search", action="store_true",
                         help="Roda GridSearchCV para ajustar C e gamma (mais lento)")
    parser.add_argument("--hardware", action="store_true",
                         help="Inclui audio real do INMP441 (data/X_hw.npy) no treino")
    parser.add_argument("--hw-weight", type=float, default=20.0,
                         help="Peso de cada janela de hardware no treino")
    parser.add_argument("--hw-final", action="store_true",
                         help="Treina com TODO o hardware (sem holdout) -- use no modelo final")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X = np.load(args.data_dir / "X.npy")
    y = np.load(args.data_dir / "y.npy")

    groups_path = args.data_dir / "groups.npy"
    if groups_path.exists():
        # Divide por CLIPE de origem, nao por janela: janelas do mesmo clipe
        # (mesmo locutor/gravacao) nunca aparecem ao mesmo tempo em treino e
        # teste, o que inflaria a acuracia artificialmente (vazamento de dados).
        groups = np.load(groups_path)
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        print(f"Split por clipe de origem: {len(set(groups[train_idx]))} clipes treino / "
              f"{len(set(groups[test_idx]))} clipes teste")
    else:
        print("[aviso] groups.npy nao encontrado -- dividindo por janela (pode vazar dados "
              "entre treino/teste).")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        groups_train = None

    sample_weight = np.ones(len(y_train))
    X_hw_test = y_hw_test = None
    if args.hardware:
        # Audio real do INMP441 (09_extrair_features_hardware.py) misturado ao
        # voxpopuli, com peso maior: sao poucas janelas mas representam o
        # dominio de deploy (mic + protoboard + sala) que o voxpopuli nao cobre.
        X_hw = np.load(args.data_dir / "X_hw.npy")
        y_hw = np.load(args.data_dir / "y_hw.npy")
        holdout = np.load(args.data_dir / "holdout_hw.npy").astype(bool)
        usar_treino = np.ones(len(y_hw), dtype=bool) if args.hw_final else ~holdout
        if not args.hw_final:
            X_hw_test, y_hw_test = X_hw[holdout], y_hw[holdout]
        X_train = np.vstack([X_train, X_hw[usar_treino]])
        y_train = np.concatenate([y_train, y_hw[usar_treino]])
        w_hw = np.full(int(usar_treino.sum()), args.hw_weight)
        # Balanceia as classes dentro do hardware (ha mais janelas femininas que
        # masculinas); sem isso o modelo enviesa para "feminino".
        y_uso = y_hw[usar_treino]
        n_f, n_m = int((y_uso == 0).sum()), int((y_uso == 1).sum())
        if n_f and n_m:
            w_hw[y_uso == 1] *= n_f / n_m
        sample_weight = np.concatenate([sample_weight, w_hw])
        groups_train = None  # grid search por grupo nao se aplica com hardware misturado
        print(f"Hardware: {int(usar_treino.sum())} janelas no treino (peso {args.hw_weight}), "
              f"{0 if X_hw_test is None else len(y_hw_test)} janelas reservadas p/ avaliacao")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=args.seed)),
    ])

    if args.grid_search:
        param_grid = {
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto", 0.001, 0.01],
        }
        cv = GroupKFold(n_splits=5) if groups_train is not None else 5
        fit_params = {"groups": groups_train} if groups_train is not None else {}
        print("Rodando GridSearchCV (cv=5, scoring=f1)...")
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train, **fit_params)
        print("Melhores parametros:", grid.best_params_)
        pipeline = grid.best_estimator_
    else:
        pipeline.fit(X_train, y_train, svm__sample_weight=sample_weight)

    y_pred = pipeline.predict(X_test)
    print("\n[voxpopuli - teste]")
    print(classification_report(y_test, y_pred, target_names=["feminino", "masculino"]))
    print("Matriz de confusao:")
    print(confusion_matrix(y_test, y_pred))

    if X_hw_test is not None:
        y_hw_pred = pipeline.predict(X_hw_test)
        print("\n[hardware real - holdout (locutor masculino B + femininas 9-11 e 22)]")
        print(classification_report(y_hw_test, y_hw_pred, target_names=["feminino", "masculino"], zero_division=0))
        print("Matriz de confusao:")
        print(confusion_matrix(y_hw_test, y_hw_pred, labels=[0, 1]))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = args.out_dir / "pipeline.joblib"
    joblib.dump(pipeline, joblib_path)
    print(f"\nPipeline sklearn salvo em {joblib_path}")

    n_features = X_train.shape[1]
    tipo_entrada = [("float_input", FloatTensorType([None, n_features]))]
    # zipmap=False: saida[1] vira um array float (N, 2) em vez de lista de dicts,
    # mais facil de consumir tanto no script de teste quanto em C depois.
    modelo_onnx = convert_sklearn(
        pipeline,
        initial_types=tipo_entrada,
        options={id(pipeline): {"zipmap": False}},
    )

    onnx_path = args.out_dir / "detector_genero_voz.onnx"
    with open(onnx_path, "wb") as f:
        f.write(modelo_onnx.SerializeToString())

    tamanho_kb = onnx_path.stat().st_size / 1024
    print(f"Modelo exportado: {onnx_path} ({tamanho_kb:.1f} KB)")


if __name__ == "__main__":
    main()
