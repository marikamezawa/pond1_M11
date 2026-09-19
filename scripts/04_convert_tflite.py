#!/usr/bin/env python3
"""
04_convert_tflite.py -- Tenta converter model/detector_genero_voz.onnx para
.tflite usando onnx2tf.

AVISO DE COMPATIBILIDADE
-------------------------
O pipeline (StandardScaler + SVC) e exportado pelo skl2onnx usando operadores
do dominio ai.onnx.ml (Scaler, SVMClassifier, ZipMap). Esses operadores sao
suportados pelo ONNX Runtime, mas conversores ONNX->TFLite genericos como
onnx2tf/onnx-tf implementam apenas o dominio padrao ai.onnx (voltado a
grafos de redes neurais) e normalmente NAO sabem converter SVMClassifier.

Ou seja: e esperado que esta conversao falhe para este modelo. Este script
tenta mesmo assim (pode funcionar em versoes futuras dos conversores) e, se
falhar, informa a alternativa adotada neste projeto: rodar
04b_export_svm_header.py para extrair os parametros do SVM treinado (vetores
de suporte, coeficientes, gamma, media/desvio do scaler) para um header C
(model/svm_params.h), que a Fase 2 usa para implementar a inferencia do SVM
diretamente em C no ESP32 -- sem depender de TFLite Micro.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-path", type=Path, default=Path("model/detector_genero_voz.onnx"))
    parser.add_argument("--out-dir", type=Path, default=Path("model/tflite"))
    args = parser.parse_args()

    if not args.onnx_path.exists():
        sys.exit(f"{args.onnx_path} nao encontrado. Rode 03_train_model.py antes.")

    try:
        import onnx2tf
    except ImportError:
        sys.exit(
            "Pacote 'onnx2tf' nao instalado (e opcional -- ver requirements.txt).\n"
            "Instale com: pip install onnx2tf tensorflow\n"
            "Mesmo instalado, a conversao provavelmente falha para este modelo "
            "(SVMClassifier e um operador ai.onnx.ml, nao suportado por onnx2tf).\n"
            "Use scripts/04b_export_svm_header.py como alternativa."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tentando converter {args.onnx_path} -> {args.out_dir} ...")
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(args.onnx_path),
            output_folder_path=str(args.out_dir),
            output_signaturedefs=True,
        )
    except Exception as e:
        print("\n[FALHA NA CONVERSAO]")
        print(f"  {type(e).__name__}: {e}")
        print(
            "\nIsso e esperado: o operador SVMClassifier (dominio ai.onnx.ml) "
            "nao e suportado por onnx2tf.\n"
            "Use o fallback: python scripts/04b_export_svm_header.py\n"
            "Ele gera model/svm_params.h com os parametros do SVM treinado "
            "para inferencia manual em C na Fase 2 (ESP32)."
        )
        sys.exit(1)

    print(f"\nConversao concluida. Arquivos .tflite em {args.out_dir}/")


if __name__ == "__main__":
    main()
