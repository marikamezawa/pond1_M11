// dsp.cpp -- ver dsp.h. Espelha scripts/dsp_common.py formula por formula.
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

// Buffers de trabalho (estaticos -- evita estourar stack da task; nao
// reentrante, ok porque so a Task 2 chama isso, uma vez por vez).
static float g_frame_real[DSP_N_FFT];
static float g_frame_imag[DSP_N_FFT];

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
    // bit-reversal
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

    // butterflies
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

void dsp_extrair_features(float *amostras, int n_amostras, float *saida_features) {
    dsp_init();

    // Pre-enfase in-place: y[n] = y[n] - 0.97*y[n-1], preservando y[0].
    float anterior = amostras[0];
    for (int n = 1; n < n_amostras; n++) {
        float atual = amostras[n];
        amostras[n] = atual - 0.97f * anterior;
        anterior = atual;
    }

    // RMS sobre o sinal inteiro (ja pre-enfatizado)
    double soma_sq = 0.0;
    for (int n = 0; n < n_amostras; n++) {
        soma_sq += (double)amostras[n] * (double)amostras[n];
    }
    float rms = sqrtf((float)(soma_sq / n_amostras));

    int n_frames = 0;
    if (n_amostras >= DSP_N_FFT) {
        n_frames = 1 + (n_amostras - DSP_N_FFT) / DSP_HOP;
    }
    if (n_frames < 1) n_frames = 1;  // fallback minimo (nao deve ocorrer em uso normal)

    double centroid_sum = 0.0;
    float mfcc_sum[DSP_N_MFCC];
    memset(mfcc_sum, 0, sizeof(mfcc_sum));

    float freq_por_bin = (float)DSP_SAMPLE_RATE / (float)DSP_N_FFT;

    for (int fidx = 0; fidx < n_frames; fidx++) {
        int inicio = fidx * DSP_HOP;

        for (int n = 0; n < DSP_N_FFT; n++) {
            int idx = inicio + n;
            float s = (idx < n_amostras) ? amostras[idx] : 0.0f;
            g_frame_real[n] = s * g_hann[n];
            g_frame_imag[n] = 0.0f;
        }

        fft_radix2(g_frame_real, g_frame_imag, DSP_N_FFT);

        float magnitude[DSP_N_BINS];
        float power[DSP_N_BINS];
        double soma_mag = 0.0, soma_mag_freq = 0.0;
        for (int k = 0; k < DSP_N_BINS; k++) {
            float re = g_frame_real[k], im = g_frame_imag[k];
            float mag = sqrtf(re * re + im * im);
            magnitude[k] = mag;
            power[k] = mag * mag;
            soma_mag += mag;
            soma_mag_freq += (double)mag * (double)(k * freq_por_bin);
        }
        float centroid_frame = (soma_mag > 1e-10) ? (float)(soma_mag_freq / soma_mag) : 0.0f;
        centroid_sum += centroid_frame;

        float log_mel[DSP_N_MELS];
        for (int m = 0; m < DSP_N_MELS; m++) {
            float energia = 0.0f;
            int fl = g_mel_left[m], fc = g_mel_center[m], fr = g_mel_right[m];
            for (int k = fl; k < fc; k++) {
                float peso = (float)(k - fl) / (float)(fc - fl);
                energia += peso * power[k];
            }
            for (int k = fc; k < fr && k < DSP_N_BINS; k++) {
                float peso = (float)(fr - k) / (float)(fr - fc);
                energia += peso * power[k];
            }
            if (energia < 1e-10f) energia = 1e-10f;
            log_mel[m] = logf(energia);
        }

        for (int kdct = 0; kdct < DSP_N_MFCC; kdct++) {
            float soma = 0.0f;
            for (int m = 0; m < DSP_N_MELS; m++) {
                soma += g_dct_basis[kdct][m] * log_mel[m];
            }
            mfcc_sum[kdct] += soma;
        }
    }

    saida_features[0] = rms;
    saida_features[1] = (float)(centroid_sum / n_frames);
    for (int k = 0; k < DSP_N_MFCC; k++) {
        saida_features[2 + k] = mfcc_sum[k] / n_frames;
    }
}
