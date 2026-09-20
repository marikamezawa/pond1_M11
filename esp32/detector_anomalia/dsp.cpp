// dsp.cpp -- ver dsp.h. Espelha scripts/dsp_common.py formula por formula.
//
// dsp_extrair_features (batch) e dsp_stream_* (streaming) compartilham o
// mesmo helper processar_frame() -- garante que os dois caminhos
// calculam EXATAMENTE a mesma coisa (validado em
// esp32/host_test/test_dsp_stream_host.cpp).
#include "dsp.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define DSP_N_BINS (DSP_N_FFT / 2 + 1)   // 257

static bool g_inicializado = false;

static float g_hann[DSP_N_FFT];

// Filtros Mel representados de forma compacta (sem matriz densa 26x257):
// cada filtro m tem [bin_left, bin_center, bin_right), pesos calculados na hora.
static int g_mel_left[DSP_N_MELS];
static int g_mel_center[DSP_N_MELS];
static int g_mel_right[DSP_N_MELS];

// Base da DCT-II ortonormal, precomputada: DCT_BASIS[k][m], k=0..12, m=0..25
static float g_dct_basis[DSP_N_MFCC][DSP_N_MELS];

// Buffers de trabalho (estaticos -- pequenos, ~5KB total; nao
// reentrante, ok porque so a Task 2 chama isso, uma vez por vez).
static float g_frame_real[DSP_N_FFT];
static float g_frame_imag[DSP_N_FFT];
static float g_power[DSP_N_BINS];
static float g_frame_norm[DSP_N_FFT];

static float hz_to_mel(float f) {
    return 2595.0f * log10f(1.0f + f / 700.0f);
}

static float mel_to_hz(float m) {
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

static void construir_mel_filterbank() {
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel((float)DSP_SAMPLE_RATE / 2.0f);

    float mel_points[DSP_N_MELS + 2];
    for (int i = 0; i < DSP_N_MELS + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (DSP_N_MELS + 1);
    }

    int bin_points[DSP_N_MELS + 2];
    for (int i = 0; i < DSP_N_MELS + 2; i++) {
        float hz = mel_to_hz(mel_points[i]);
        int b = (int)floorf((DSP_N_FFT + 1) * hz / DSP_SAMPLE_RATE);
        if (b < 0) b = 0;
        if (b > DSP_N_BINS - 1) b = DSP_N_BINS - 1;
        bin_points[i] = b;
    }

    for (int m = 1; m <= DSP_N_MELS; m++) {
        int f_left = bin_points[m - 1];
        int f_center = bin_points[m];
        int f_right = bin_points[m + 1];
        if (f_center == f_left) f_center++;
        if (f_right == f_center) f_right++;
        g_mel_left[m - 1] = f_left;
        g_mel_center[m - 1] = f_center;
        g_mel_right[m - 1] = f_right;
    }
}

static void construir_dct_basis() {
    for (int k = 0; k < DSP_N_MFCC; k++) {
        float escala = (k == 0) ? sqrtf(1.0f / (4.0f * DSP_N_MELS))
                                 : sqrtf(1.0f / (2.0f * DSP_N_MELS));
        for (int m = 0; m < DSP_N_MELS; m++) {
            float v = cosf((float)M_PI / DSP_N_MELS * (m + 0.5f) * k) * 2.0f;
            g_dct_basis[k][m] = v * escala;
        }
    }
}

static void dsp_init() {
    if (g_inicializado) return;

    for (int n = 0; n < DSP_N_FFT; n++) {
        g_hann[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / DSP_N_FFT);
    }

    construir_mel_filterbank();
    construir_dct_basis();

    g_inicializado = true;
}

// FFT radix-2 Cooley-Tukey, in-place, iterativa. n deve ser potencia de 2.
static void fft_radix2(float *real, float *imag, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cur_wr = 1.0f, cur_wi = 0.0f;
            int half = len / 2;
            for (int k = 0; k < half; k++) {
                int ie = i + k, io = i + k + half;
                float ur = real[ie], ui = imag[ie];
                float vr = real[io] * cur_wr - imag[io] * cur_wi;
                float vi = real[io] * cur_wi + imag[io] * cur_wr;
                real[ie] = ur + vr;
                imag[ie] = ui + vi;
                real[io] = ur - vr;
                imag[io] = ui - vi;

                float next_wr = cur_wr * wr - cur_wi * wi;
                float next_wi = cur_wr * wi + cur_wi * wr;
                cur_wr = next_wr;
                cur_wi = next_wi;
            }
        }
    }
}

// Processa UM frame de DSP_N_FFT amostras JA PRE-ENFATIZADAS (mas ainda
// nao normalizadas): normaliza a energia do frame para DSP_TARGET_FRAME_RMS
// (pula o frame -- nao acumula nada -- se estiver abaixo do piso de
// energia DSP_MIN_FRAME_RMS), janela, FFT, centroid, mel+log, DCT.
// Retorna true se o frame contribuiu (nao foi pulado).
static bool processar_frame(const float *frame, double *centroid_sum_acc, float *mfcc_sum_acc) {
    double soma_sq = 0.0;
    for (int n = 0; n < DSP_N_FFT; n++) soma_sq += (double)frame[n] * (double)frame[n];
    float frame_rms = sqrtf((float)(soma_sq / DSP_N_FFT));

    if (frame_rms < DSP_MIN_FRAME_RMS) {
        return false;  // quase silencio -- nao amplifica ruido, nao contribui
    }

    float escala = DSP_TARGET_FRAME_RMS / (frame_rms > 1e-10f ? frame_rms : 1e-10f);
    if (escala > DSP_MAX_SCALE) escala = DSP_MAX_SCALE;
    for (int n = 0; n < DSP_N_FFT; n++) g_frame_norm[n] = frame[n] * escala;

    float freq_por_bin = (float)DSP_SAMPLE_RATE / (float)DSP_N_FFT;

    for (int n = 0; n < DSP_N_FFT; n++) {
        g_frame_real[n] = g_frame_norm[n] * g_hann[n];
        g_frame_imag[n] = 0.0f;
    }

    fft_radix2(g_frame_real, g_frame_imag, DSP_N_FFT);

    double soma_mag = 0.0, soma_mag_freq = 0.0;
    for (int k = 0; k < DSP_N_BINS; k++) {
        float re = g_frame_real[k], im = g_frame_imag[k];
        float mag = sqrtf(re * re + im * im);
        g_power[k] = mag * mag;
        soma_mag += mag;
        soma_mag_freq += (double)mag * (double)(k * freq_por_bin);
    }
    float centroid_frame = (soma_mag > 1e-10) ? (float)(soma_mag_freq / soma_mag) : 0.0f;
    *centroid_sum_acc += centroid_frame;

    float log_mel[DSP_N_MELS];
    for (int m = 0; m < DSP_N_MELS; m++) {
        float energia = 0.0f;
        int fl = g_mel_left[m], fc = g_mel_center[m], fr = g_mel_right[m];
        for (int k = fl; k < fc; k++) {
            float peso = (float)(k - fl) / (float)(fc - fl);
            energia += peso * g_power[k];
        }
        for (int k = fc; k < fr && k < DSP_N_BINS; k++) {
            float peso = (float)(fr - k) / (float)(fr - fc);
            energia += peso * g_power[k];
        }
        if (energia < 1e-10f) energia = 1e-10f;
        log_mel[m] = logf(energia);
    }

    for (int kdct = 0; kdct < DSP_N_MFCC; kdct++) {
        float soma = 0.0f;
        for (int m = 0; m < DSP_N_MELS; m++) {
            soma += g_dct_basis[kdct][m] * log_mel[m];
        }
        mfcc_sum_acc[kdct] += soma;
    }

    return true;
}

// ================= API batch =================

void dsp_extrair_features(float *amostras, int n_amostras, float *rms_janela, float *saida_features_svm) {
    dsp_init();

    // RMS da janela ANTES da pre-enfase (gatilho de silencio, nao entra
    // no classificador).
    double soma_sq_bruto = 0.0;
    for (int n = 0; n < n_amostras; n++) {
        soma_sq_bruto += (double)amostras[n] * (double)amostras[n];
    }
    *rms_janela = sqrtf((float)(soma_sq_bruto / n_amostras));

    // Pre-enfase in-place: y[n] = y[n] - 0.97*y[n-1], preservando y[0].
    float anterior = amostras[0];
    for (int n = 1; n < n_amostras; n++) {
        float atual = amostras[n];
        amostras[n] = atual - 0.97f * anterior;
        anterior = atual;
    }

    int n_frames_total = 0;
    if (n_amostras >= DSP_N_FFT) {
        n_frames_total = 1 + (n_amostras - DSP_N_FFT) / DSP_HOP;
    }
    if (n_frames_total < 1) n_frames_total = 1;

    double centroid_sum = 0.0;
    float mfcc_sum[DSP_N_MFCC];
    memset(mfcc_sum, 0, sizeof(mfcc_sum));
    int n_contribuiram = 0;

    static float frame_tmp[DSP_N_FFT];
    for (int fidx = 0; fidx < n_frames_total; fidx++) {
        int inicio = fidx * DSP_HOP;
        for (int n = 0; n < DSP_N_FFT; n++) {
            int idx = inicio + n;
            frame_tmp[n] = (idx < n_amostras) ? amostras[idx] : 0.0f;
        }
        if (processar_frame(frame_tmp, &centroid_sum, mfcc_sum)) {
            n_contribuiram++;
        }
    }

    int divisor = (n_contribuiram > 0) ? n_contribuiram : 1;
    saida_features_svm[0] = (float)(centroid_sum / divisor);
    for (int k = 0; k < DSP_N_MFCC; k++) {
        saida_features_svm[1 + k] = mfcc_sum[k] / divisor;
    }
}

// ================= API streaming =================

void dsp_stream_reset(dsp_stream_t *st) {
    dsp_init();
    st->buf_count = 0;
    st->preemph_anterior = 0.0f;
    st->tem_anterior = 0;
    st->soma_sq_bruto = 0.0;
    st->n_amostras_total = 0;
    st->centroid_sum = 0.0;
    memset(st->mfcc_sum, 0, sizeof(st->mfcc_sum));
    st->n_frames = 0;
}

void dsp_stream_push(dsp_stream_t *st, const int16_t *amostras, int n) {
    for (int i = 0; i < n; i++) {
        float bruto = amostras[i] / 32768.0f;

        st->soma_sq_bruto += (double)bruto * (double)bruto;
        st->n_amostras_total++;

        float pre;
        if (!st->tem_anterior) {
            pre = bruto;
            st->tem_anterior = 1;
        } else {
            pre = bruto - 0.97f * st->preemph_anterior;
        }
        st->preemph_anterior = bruto;

        if (st->buf_count < DSP_STREAM_BUF_CAP) {
            st->buf[st->buf_count++] = pre;
        }
    }

    while (st->buf_count >= DSP_N_FFT) {
        if (processar_frame(st->buf, &st->centroid_sum, st->mfcc_sum)) {
            st->n_frames++;
        }

        int restante = st->buf_count - DSP_HOP;
        memmove(st->buf, st->buf + DSP_HOP, restante * sizeof(float));
        st->buf_count = restante;
    }
}

void dsp_stream_finalizar(const dsp_stream_t *st, float *rms_janela, float *saida_features_svm) {
    *rms_janela = (st->n_amostras_total > 0)
                      ? sqrtf((float)(st->soma_sq_bruto / st->n_amostras_total))
                      : 0.0f;

    int divisor = (st->n_frames > 0) ? st->n_frames : 1;
    saida_features_svm[0] = (float)(st->centroid_sum / divisor);
    for (int k = 0; k < DSP_N_MFCC; k++) {
        saida_features_svm[1 + k] = st->mfcc_sum[k] / divisor;
    }
}

void dsp_stream_fechar_segmento(dsp_stream_t *st, dsp_segmento_t *seg) {
    seg->soma_sq_bruto = st->soma_sq_bruto;
    seg->n_amostras = st->n_amostras_total;
    seg->centroid_sum = st->centroid_sum;
    memcpy(seg->mfcc_sum, st->mfcc_sum, sizeof(seg->mfcc_sum));
    seg->n_frames = st->n_frames;

    st->soma_sq_bruto = 0.0;
    st->n_amostras_total = 0;
    st->centroid_sum = 0.0;
    memset(st->mfcc_sum, 0, sizeof(st->mfcc_sum));
    st->n_frames = 0;
}

void dsp_segmentos_finalizar(const dsp_segmento_t *segs, int n_segs, float *rms_janela, float *saida_features_svm) {
    double soma_sq = 0.0, centroid = 0.0;
    long n_amostras = 0;
    int n_frames = 0;
    float mfcc[DSP_N_MFCC] = {0};
    for (int i = 0; i < n_segs; i++) {
        soma_sq += segs[i].soma_sq_bruto;
        n_amostras += segs[i].n_amostras;
        centroid += segs[i].centroid_sum;
        n_frames += segs[i].n_frames;
        for (int k = 0; k < DSP_N_MFCC; k++) mfcc[k] += segs[i].mfcc_sum[k];
    }
    *rms_janela = (n_amostras > 0) ? sqrtf((float)(soma_sq / n_amostras)) : 0.0f;
    int divisor = (n_frames > 0) ? n_frames : 1;
    saida_features_svm[0] = (float)(centroid / divisor);
    for (int k = 0; k < DSP_N_MFCC; k++) saida_features_svm[1 + k] = mfcc[k] / divisor;
}
