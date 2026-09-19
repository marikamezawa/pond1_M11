// svm_infer.cpp -- ver svm_infer.h
#include "svm_infer.h"
#include "svm_params.h"
#include <math.h>

float svm_prob_masculino(const float *features) {
    // 1. Normalizacao (StandardScaler): x_norm[i] = (x[i]-mean[i])/scale[i]
    float x_norm[SVM_N_FEATURES];
    for (int i = 0; i < SVM_N_FEATURES; i++) {
        x_norm[i] = (features[i] - svm_scaler_mean[i]) / svm_scaler_scale[i];
    }

    // 2. Funcao de decisao: soma dos kernels RBF ponderados pelos dual_coef
    //    K(x, sv) = exp(-gamma * ||x_norm - sv||^2)
    double decisao = svm_intercept;
    for (int s = 0; s < SVM_N_SUPPORT_VECTORS; s++) {
        float dist_sq = 0.0f;
        for (int i = 0; i < SVM_N_FEATURES; i++) {
            float d = x_norm[i] - svm_support_vectors[s][i];
            dist_sq += d * d;
        }
        float kernel = expf(-svm_gamma * dist_sq);
        decisao += (double)svm_dual_coef[s] * (double)kernel;
    }

    // 3. Platt scaling (libsvm/sklearn). ATENCAO ao sinal de probB: o par
    //    (probA, probB) do libsvm foi ajustado para estimar P(classe 0),
    //    nao P(classe 1/masculino) diretamente -- por isso o sinal de
    //    probB e SUBTRAIDO aqui (verificado numericamente contra
    //    pipeline.predict_proba() do sklearn, erro residual < 0.002):
    //    P(masculino) = 1 / (1 + exp(decisao*probA - probB))
    double expoente = decisao * (double)svm_prob_a - (double)svm_prob_b;
    double prob = 1.0 / (1.0 + exp(expoente));

    return (float)prob;
}
