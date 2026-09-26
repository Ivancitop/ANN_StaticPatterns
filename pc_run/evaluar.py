"""
evaluar.py
==========
Clasificacion PC vs MCU sobre el dataset REAL (dataset_real.csv).

  - PC : MLP en float64 con los pesos que entreno el MCU (resultados_mcu.json)
  - MCU: el mismo vector de caracteristicas (hex exacto) enviado en modo 'V'

Como ambos usan los mismos pesos y las mismas entradas bit a bit, cualquier
discrepancia de clase se debe solo a la aritmetica float32 del MCU.
Ademas reporta la exactitud del modelo entrenado con datos sinteticos al
enfrentarse a datos reales (la brecha sintetico -> real).

Uso:
    python evaluar.py --puerto COM6        # el MCU debe tener la red entrenada
    python evaluar.py --solo-pc            # sin hardware
Salida:
    resultados_real.json
"""

import argparse
import csv
import os

import numpy as np

import mlp_core as core
from mlp_core import (MLP, load_json, save_json, matriz_confusion, metricas_clase,
                      clasificador_umbral)


def cargar(path):
    X, Xh, y, sub = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            Xh.append([row[f"{n}_hex"] for n in core.FEAT_NAMES])
            X.append([core.hex_to_f32(h) for h in Xh[-1]])
            y.append(int(row["clase"])); sub.append(row["subtipo"])
    return np.array(X), Xh, np.array(y), np.array(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--datos", default="dataset_real.csv")
    ap.add_argument("--solo-pc", action="store_true")
    args = ap.parse_args()

    X, Xh, y, sub = cargar(args.datos)
    fuente = "resultados_mcu.json" if os.path.exists("resultados_mcu.json") else "resultados_pc.json"
    net = MLP(dtype=np.float64); net.set_weights(load_json(fuente)["weights_final"])
    print(f"{len(X)} muestras reales; pesos de {fuente}")

    P_pc = net.predict_proba(X)
    yp_pc = np.where(P_pc.max(1) >= 0.70, P_pc.argmax(1), 0)   # mismo umbral que el firmware
    out = {"n": len(X), "pesos": fuente,
           "cm_pc": matriz_confusion(y, yp_pc).tolist(),
           "metricas_pc": metricas_clase(matriz_confusion(y, yp_pc)),
           "cm_umbral": matriz_confusion(y, clasificador_umbral(X)).tolist()}
    out["acc_por_subtipo_pc"] = {s: float(np.mean(yp_pc[sub == s] == y[sub == s]))
                                 for s in np.unique(sub)}
    print(f"PC  : exactitud {100 * out['metricas_pc']['accuracy']:.1f} %")

    if not args.solo_pc:
        import serial
        ser = serial.Serial(args.puerto, args.baud, timeout=2.0)
        ser.write(b"X\n"); ser.readline(); ser.reset_input_buffer()
        P_mcu = []
        for hx, yy in zip(Xh, y):
            ser.write(("V," + ",".join(hx) + f",{yy}\n").encode("ascii"))
            r = ser.readline().decode("ascii").strip().split(",")
            P_mcu.append([core.hex_to_f32(h) for h in r[1:5]])
        ser.close()
        P_mcu = np.array(P_mcu)
        yp_mcu = np.where(P_mcu.max(1) >= 0.70, P_mcu.argmax(1), 0)
        M = matriz_confusion(y, yp_mcu)
        out.update({"cm_mcu": M.tolist(), "metricas_mcu": metricas_clase(M),
                    "acuerdo_pc_mcu": float(np.mean(yp_mcu == yp_pc)),
                    "max_dif_prob": float(np.max(np.abs(P_mcu - P_pc)))})
        print(f"MCU : exactitud {100 * out['metricas_mcu']['accuracy']:.1f} %   "
              f"acuerdo PC-MCU {100 * out['acuerdo_pc_mcu']:.2f} %   "
              f"max|dp| {out['max_dif_prob']:.2e}")

    save_json("resultados_real.json", out)
    print("-> resultados_real.json")


if __name__ == "__main__":
    main()
