#!/usr/bin/env python3
"""
08_capturar_audio_hardware.py -- recebe audio gravado pelo INMP441 real
(via esp32/audio_dump/audio_dump.ino) pela porta serial e salva como .wav.

Objetivo: capturar audio de VERDADE do hardware (mic + protoboard + sala)
para incluir no treino do modelo -- corrige o descompasso de dominio
(resposta em frequencia do mic, ruido eletrico) que a normalizacao de
energia por frame sozinha nao resolve, ja que o dataset original
(voxpopuli) e gravado com equipamento de estudio/transmissao.

Uso:
    1. Grave esp32/audio_dump/audio_dump.ino no ESP32 (compile + upload).
    2. python scripts/08_capturar_audio_hardware.py --port /dev/ttyUSB0 --out data/hardware/fem_01.wav
    3. O script aguarda o ESP32 ficar "PRONTO", manda um gatilho, e salva
       os 5s gravados.
    4. Repita o passo 1 (recompilar + regravar o firmware) antes de CADA
       captura -- isso garante um reset fisico de verdade a cada vez, o que
       se mostrou mais confiavel do que reusar o boot anterior.

Depois de capturar alguns arquivos, rode scripts/09_incluir_audio_hardware.py
para incorporar-los ao dataset de treino.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import serial
import soundfile as sf

SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Porta serial do ESP32, ex: /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--out", type=Path, required=True, help="Arquivo .wav de saida")
    parser.add_argument("--seconds", type=int, default=5, help="Duracao da gravacao (1-120)")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    RECORD_SECONDS = args.seconds
    N_BYTES = SAMPLE_RATE * RECORD_SECONDS * 2  # int16

    print(f"Conectando em {args.port} @ {args.baud}...")
    # timeout curto por leitura -- se deixar igual ao prazo total, uma unica
    # chamada de readline() travada em lixo de boot (bootloader ROM roda a
    # 74880 baud, sai garbled quando lido a 921600) consome o orcamento
    # inteiro de uma vez so, mesmo que o ESP32 va mandar "PRONTO" logo depois.
    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    time.sleep(2)  # da tempo pro ESP32 rebootar (upload sempre causa reset)
    ser.reset_input_buffer()  # descarta lixo de boot do bootloader ROM

    def esperar_linha(alvo, prazo):
        t0 = time.time()
        while time.time() - t0 < prazo:
            linha = ser.readline().decode(errors="ignore").strip()
            if linha:
                print(f"  esp32> {linha}")
            if linha == alvo:
                return True
        return False

    # Nao depende de ver "PRONTO": a placa pode ja ter rebootado antes de abrirmos
    # a porta. Reenvia o gatilho a cada 2s ate ela responder GRAVANDO/START.
    print(f"Disparando gravacao de {RECORD_SECONDS}s...")
    t0 = time.time()
    ultimo_gatilho = 0.0
    gravando = False
    while True:
        agora = time.time()
        if agora - t0 > args.timeout:
            sys.exit("Timeout: o ESP32 nao respondeu ao gatilho (audio_dump.ino esta gravado?).")
        if not gravando and agora - ultimo_gatilho >= 2.0:
            ser.write(bytes([RECORD_SECONDS]))
            ultimo_gatilho = agora
        linha = ser.readline().decode(errors="ignore").strip()
        if linha:
            print(f"  esp32> {linha}")
        if linha == "GRAVANDO" and not gravando:
            gravando = True
            print(f"  (preparando... fale daqui a ~0.5s, por {RECORD_SECONDS - 1}s)")
        if linha == "START":
            break

    dados = bytearray()
    t0 = time.time()
    prazo_dados = max(args.timeout, RECORD_SECONDS + N_BYTES / (args.baud / 10) + 10)  # margem generosa
    while len(dados) < N_BYTES:
        if time.time() - t0 > prazo_dados:
            sys.exit(f"Recebi so {len(dados)} de {N_BYTES} bytes esperados -- tente de novo "
                      f"(porta serial lenta demais ou desconexao).")
        pedaco = ser.read(N_BYTES - len(dados))
        dados.extend(pedaco)
    dados = bytes(dados)

    # consome o resto ate DONE
    if not esperar_linha("DONE", args.timeout):
        sys.exit("Timeout esperando o ESP32 confirmar DONE.")

    amostras = np.frombuffer(dados, dtype="<i2").astype(np.float32) / 32768.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, amostras, SAMPLE_RATE)
    print(f"\nSalvo: {args.out} ({len(amostras)/SAMPLE_RATE:.1f}s)")
    print(f"RMS: {np.sqrt(np.mean(amostras**2)):.4f}  pico: {np.max(np.abs(amostras)):.4f}")


if __name__ == "__main__":
    main()
