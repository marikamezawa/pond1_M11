#!/usr/bin/env python3
"""
03_train_model.py -- Treina o SVM (kernel RBF) em cima de X.npy/y.npy,
opcionalmente ajusta hiperparametros com GridSearchCV, avalia e exporta:

  model/detector_genero_voz.onnx  -- para inferencia (Python, ESP32 via ONNX Runtime)
  model/pipeline.joblib           -- pipeline sklearn bruto (usado pelo script
                                      04b_export_svm_header.py para gerar o
                                      fallback em C, ja que SVMClassifier nao
                                      converte para TFLite -- ver 04_convert_tflite.py)
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
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=["feminino", "masculino"]))
    print("Matriz de confusao:")
    print(confusion_matrix(y_test, y_pred))

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
