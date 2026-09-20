// svm_infer.h -- inferencia do SVM (kernel RBF) treinado em Python,
// usando os parametros exportados em svm_params.h (scripts/04b_export_svm_header.py).
//
// Motivo de nao usar ONNX/TFLite aqui: o skl2onnx exporta o pipeline usando
// operadores do dominio ai.onnx.ml (Scaler, SVMClassifier), que nao
// convertem para TFLite (ver README.md).
// Como SVM+RBF e so algebra simples (normalizacao + kernel + soma
// ponderada + sigmoid), reimplementar em C e direto e leve.
#pragma once

// Retorna P(masculino), a mesma probabilidade que
// pipeline.predict_proba(x)[0][1] retornaria em Python, para um vetor de
// entrada `features` com SVM_N_FEATURES valores (na ordem
// [rms, centroid, mfcc_0..mfcc_12], SEM normalizar -- a normalizacao
// (StandardScaler) e aplicada aqui dentro).
float svm_prob_masculino(const float *features);
