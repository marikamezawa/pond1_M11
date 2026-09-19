"""
dsp_common.py -- extracao de features (RMS + Spectral Centroid + 13 MFCCs)
usada por TODOS os scripts do pipeline (02, 05, 06), para garantir que
treino e inferencia usem exatamente a mesma formula.

Implementacao propria de MFCC (nao usa librosa.feature.mfcc/melspectrogram),
propositalmente simples o suficiente para ser espelhada fielmente em C no
ESP32 (ver esp32/detector_anomalia/dsp.cpp) sem depender de nenhuma
biblioteca de DSP externa. Os parametros batem com o que o planejamento.md
ja descrevia para o embarcado: FFT de 512 pontos, banco de 26 filtros Mel,
janela de Hamming... aqui usamos Hann (mais simples e igualmente padrao),
sem centralizacao/padding (frames simplesmente nao se sobrepoem a borda).

Pipeline por frame:
  1. Janelamento Hann, sem overlap-add especial (so windowing simples)
  2. FFT real de N_FFT pontos -> espectro de potencia (so os N_FFT/2+1 bins)
  3. Banco de filtros Mel (escala HTK, triangulares, pico=1, nao normalizados
     pela area -- versao classica/mais simples que a Slaney do librosa)
  4. Log da energia de cada filtro (log natural, com piso para evitar log(0))
  5. DCT-II ortonormal -> pega os 13 primeiros coeficientes
  6. Media dos vetores de MFCC sobre todos os frames da janela de audio

RMS e Spectral Centroid sao calculados separadamente (RMS no dominio do
tempo; centroid por frame usando o espectro de MAGNITUDE, depois media).
"""
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160          # 10ms @ 16kHz
N_MELS = 26
N_MFCC = 13
FMIN = 0.0
FMAX = SAMPLE_RATE / 2.0
EPS = 1e-10

# Duracao da janela de analise usada em TODO o pipeline (treino, teste,
# mic ao vivo, e no ESP32 -- Task 2 acumula esse tanto de amostras antes de
# calcular UM FeatureVector). Escolhido apos validar ao vivo que 1.5s
# causava erros sistematicos (features instaveis, poucas silabas por
# janela); 3s da mais estabilidade mantendo resposta "quase em tempo real".
WINDOW_SEC = 3.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)

_hann_window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _build_mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Retorna matriz (n_mels, n_fft//2+1) com filtros triangulares HTK, pico=1."""
    n_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    fbank = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if f_center == f_left:
            f_center += 1
        if f_right == f_center:
            f_right += 1
        for k in range(f_left, f_center):
            fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if k < n_bins:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fbank.astype(np.float32)


_MEL_FILTERBANK = _build_mel_filterbank()


def _build_dct_basis(n_in=N_MELS, n_out=N_MFCC):
    """Base da DCT-II ortonormal, pre-computada uma unica vez: (n_out, n_in)."""
    k = np.arange(n_out)[:, None]
    nidx = np.arange(n_in)[None, :]
    basis = np.cos(np.pi / n_in * (nidx + 0.5) * k) * 2.0
    basis[0] *= np.sqrt(1.0 / (4.0 * n_in))
    if n_out > 1:
        basis[1:] *= np.sqrt(1.0 / (2.0 * n_in))
    return basis.astype(np.float64)


_DCT_BASIS = _build_dct_basis()  # (N_MFCC, N_MELS)


def extrair_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extrai [rms, spectral_centroid, mfcc_0..mfcc_12] de um sinal 1D float32
    normalizado em [-1, 1] (mesma convencao usada em todo o projeto).
    """
    y = y.astype(np.float64)

    # Pre-enfase (mesma formula usada desde o inicio do projeto)
    y_preemph = np.append(y[0], y[1:] - 0.97 * y[:-1])

    rms = float(np.sqrt(np.mean(y_preemph ** 2)))

    n_frames = 1 + (len(y_preemph) - N_FFT) // HOP if len(y_preemph) >= N_FFT else 0
    if n_frames <= 0:
        # audio curto demais para 1 frame: usa zero-padding de um unico frame
        frame = np.zeros(N_FFT, dtype=np.float64)
        frame[:len(y_preemph)] = y_preemph[:N_FFT]
        frames = frame[None, :]
        n_frames = 1
    else:
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
        frames = y_preemph[idx]

    windowed = frames * _hann_window[None, :]
    spectrum = np.fft.rfft(windowed, n=N_FFT, axis=1)   # (n_frames, N_FFT/2+1)
    magnitude = np.abs(spectrum)
    power = magnitude ** 2

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    soma_mag = magnitude.sum(axis=1)
    soma_mag_safe = np.where(soma_mag > EPS, soma_mag, EPS)
    centroid_por_frame = (magnitude @ freqs) / soma_mag_safe
    centroid = float(centroid_por_frame.mean())

    mel_energy = power @ _MEL_FILTERBANK.T          # (n_frames, n_mels)
    log_mel = np.log(np.maximum(mel_energy, EPS))

    mfccs = log_mel @ _DCT_BASIS.T                   # (n_frames, N_MFCC)
    mfcc_mean = mfccs.mean(axis=0)

    return np.concatenate([[rms, centroid], mfcc_mean]).astype(np.float32)
