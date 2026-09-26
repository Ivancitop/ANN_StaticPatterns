"""
capturar.py
===========
Registro y anotacion del dataset REAL con los potenciometros de la tarjeta.

El firmware calcula las 9 caracteristicas con el mismo preprocesado que se
usa en inferencia (comando 'C') y las devuelve en hex junto con la lectura
cruda. El script pide al operador que coloque cada patron y guarda N muestras
por clase con su etiqueta. Con ese archivo, evaluar.py compara la
clasificacion de la PC contra la del MCU sobre EXACTAMENTE los mismos datos.

Uso:
    python capturar.py --puerto COM6 --n 300
Salida:
    dataset_real.csv
"""

import argparse
import csv
import time

import mlp_core as core
from mlp_core import hex_to_f32

# Patron a colocar -> (clase, subtipo). La clase default se captura con varias
# combinaciones para que quede representada como en el dataset sintetico.
SESIONES = [
    (1, "HHL", "pot1 ALTO, pot2 ALTO, pot3 BAJO  (CW)"),
    (2, "LHH", "pot1 BAJO, pot2 ALTO, pot3 ALTO  (CCW)"),
    (3, "HLH", "pot1 ALTO, pot2 BAJO, pot3 ALTO  (paro)"),
    (0, "HHH", "los tres ALTOS"),
    (0, "LLL", "los tres BAJOS"),
    (0, "HLL", "pot1 ALTO, pot2 BAJO, pot3 BAJO"),
    (0, "LHL", "pot1 BAJO, pot2 ALTO, pot3 BAJO"),
    (0, "LLH", "pot1 BAJO, pot2 BAJO, pot3 ALTO"),
    (0, "medio", "al menos un pote a media escala (40-60 %)"),
    (0, "movimiento", "GIRA lentamente un pote durante toda la captura"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=300, help="muestras por clase activa")
    ap.add_argument("--periodo", type=float, default=0.17,
                    help="s entre capturas (>= ventana de 160 ms: sin traslape)")
    ap.add_argument("--salida", default="dataset_real.csv")
    args = ap.parse_args()

    import serial
    ser = serial.Serial(args.puerto, args.baud, timeout=2.0)
    time.sleep(0.5); ser.write(b"X\n"); ser.readline(); ser.reset_input_buffer()

    # La clase default reparte sus N muestras entre sus 7 variantes
    n_def = max(1, args.n // sum(1 for s in SESIONES if s[0] == 0))

    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "subtipo", "raw1", "raw2", "raw3"] + core.FEAT_NAMES
                   + [f"{n}_hex" for n in core.FEAT_NAMES])
        for clase, sub, texto in SESIONES:
            n = args.n if clase != 0 else n_def
            input(f"\n[{core.CLASES[clase]}] Coloca: {texto}\n  Enter para capturar {n} muestras...")
            k = 0
            while k < n:
                ser.write(b"C\n")
                r = ser.readline().decode("ascii", errors="replace").strip()
                if not r or r in ("N", "E") or r.startswith("D,"):
                    time.sleep(args.periodo); continue
                p = r.split(",")
                hx = p[3:3 + core.N_FEAT]
                w.writerow([clase, sub] + p[:3] + [f"{hex_to_f32(h):.6f}" for h in hx] + hx)
                k += 1
                if k % 25 == 0:
                    print(f"  {k}/{n}  crudo={p[:3]}")
                time.sleep(args.periodo)
    ser.close()
    print(f"\nDataset anotado guardado en {args.salida}")


if __name__ == "__main__":
    main()
