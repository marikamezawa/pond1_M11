"""
dsp_common.py -- extracao de features (Spectral Centroid + 13 MFCCs) usada
por TODOS os scripts do pipeline (02, 05, 06), para garantir que treino e
inferencia usem exatamente a mesma formula. Espelhada em C no ESP32 (ver
esp32/detector_anomalia/dsp.cpp).

Historico importante (por que o RMS bruto NAO entra mais no classificador):
o RMS absoluto captado varia MUITO entre hardwares/distancias/ganhos de
microfone -- o dataset de treino (voxpopuli, ~0.028 de RMS medio), o
microfone do notebook (~0.01-0.03) e o INMP441 real (0.04-0.12 falando
perto) ficam em escalas bem diferentes. Como RMS era uma das features
originais, isso enviesava a classificacao (energia fora da distribuicao de
treino, nao o timbre da voz, dominava a decisao).

Correcao: cada FRAME (512 amostras) e normalizado para uma energia alvo
antes do MFCC -- a FORMA espectral (que carrega a informacao de timbre/
formantes, ligada ao genero da voz) fica independente do volume absoluto
de captura. O RMS da janela inteira continua sendo calculado (funcao
`calcular_rms`), mas so como gatilho de silencio (VAD) -- nao alimenta
mais o classificador.

Pipeline por frame:
  1. Normaliza a energia do frame para TARGET_FRAME_RMS (frames quase
     silenciosos, abaixo de MIN_FRAME_RMS, sao ignorados -- nao
     contribuem pra media, evita amplificar ruido de fundo)
  2. Janelamento Hann
  3. FFT real de N_FFT pontos -> espectro de potencia (N_FFT/2+1 bins)
  4. Banco de filtros Mel (escala HTK, triangulares, pico=1)
  5. Log da energia de cada filtro (log natural, com piso)
  6. DCT-II ortonormal -> 13 primeiros coeficientes
  7. Media do centroid e dos MFCCs sobre todos os frames que contribuiram
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
# calcular UM FeatureVector).
WINDOW_SEC = 3.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)

# Normalizacao de energia por frame (ver docstring do modulo).
TARGET_FRAME_RMS = 0.1
MIN_FRAME_RMS = 0.003
MAX_SCALE = 15.0

# Dimensao do vetor de features que alimenta o classificador:
# [centroid, mfcc_0..mfcc_12]. RMS NAO esta aqui -- ver calcular_rms().
FEATURE_DIM = 1 + N_MFCC

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


def calcular_rms(y: np.ndarray) -> float:
    """RMS da janela inteira (sinal cru, sem pre-enfase) -- usado so como
    gatilho de silencio (VAD), NAO alimenta o classificador."""
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))


def extrair_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extrai [centroid, mfcc_0..mfcc_12] (FEATURE_DIM=14) de um sinal 1D
    float, normalizado em [-1, 1]. Cada frame de 512 amostras e
    normalizado individualmente para uma energia alvo antes do MFCC (ver
    docstring do modulo) -- torna o resultado robusto ao ganho/distancia
    do microfone usado na captura.
    """
    y = y.astype(np.float64)
    y_preemph = np.append(y[0], y[1:] - 0.97 * y[:-1])

    n_frames_total = 1 + (len(y_preemph) - N_FFT) // HOP if len(y_preemph) >= N_FFT else 0
    if n_frames_total <= 0:
        frame = np.zeros(N_FFT, dtype=np.float64)
        frame[:len(y_preemph)] = y_preemph[:N_FFT]
        frames = frame[None, :]
        n_frames_total = 1
    else:
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames_total)[:, None]
        frames = y_preemph[idx]

    # Normalizacao de energia por frame + filtro de frames quase-silenciosos
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    validos = frame_rms >= MIN_FRAME_RMS
    if not np.any(validos):
        # janela inteira abaixo do piso -- usa todos os frames sem filtrar
        # para nao devolver um vetor vazio (caller ja deveria ter feito o
        # gate de silencio pelo RMS da janela antes de chegar aqui)
        validos = np.ones(n_frames_total, dtype=bool)

    escala = TARGET_FRAME_RMS / np.maximum(frame_rms, EPS)
    escala = np.minimum(escala, MAX_SCALE)
    frames_norm = frames * escala[:, None]

    frames_validos = frames_norm[validos]
    windowed = frames_validos * _hann_window[None, :]
    spectrum = np.fft.rfft(windowed, n=N_FFT, axis=1)
    magnitude = np.abs(spectrum)
    power = magnitude ** 2

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    soma_mag = magnitude.sum(axis=1)
    soma_mag_safe = np.where(soma_mag > EPS, soma_mag, EPS)
    centroid_por_frame = (magnitude @ freqs) / soma_mag_safe
    centroid = float(centroid_por_frame.mean())

    mel_energy = power @ _MEL_FILTERBANK.T
    log_mel = np.log(np.maximum(mel_energy, EPS))

    mfccs = log_mel @ _DCT_BASIS.T
    mfcc_mean = mfccs.mean(axis=0)

    return np.concatenate([[centroid], mfcc_mean]).astype(np.float32)
