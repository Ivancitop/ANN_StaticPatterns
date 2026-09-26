"""
run_pc.py
=========
Entrena el modelo de referencia en la PC y deja todo listo para comparar
contra el microcontrolador.

Genera (en la carpeta actual):
  dataset_sintetico.csv       dataset anotado (niveles, subtipo, 9 features, clase)
  resultados_pc.json          corrida principal float64 (referencia)
  resultados_pc_f32.json      misma corrida en float32 (efecto de la precision)
  resultados_reduccion.json   topologias reducidas, cuantizacion int8, linea base
  ../firmware/pesos_iniciales.h

Uso:
    python run_pc.py
"""

import csv
import os
import time

import numpy as np

import mlp_core as core
from mlp_core import (MLP, split_dataset, epoch_orders, save_json,
                      matriz_confusion, metricas_clase, clasificador_umbral,
                      cuantizar_int8, gradient_check)

# ---------------- configuracion del experimento ----------------
# Deben ser IDENTICAS en run_mcu.py
EPOCHS = 30
LR = 0.05
SEED_WEIGHTS = 1
SEED_DATA = 0
SEED_ORDER = 7
N_POR_CLASE = 300
FIRMWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware")


def train(net, d, orders, lr, log=True):
    hist = {"loss_train": [], "loss_val": [], "acc_train": [], "acc_val": [],
            "prob_log": [], "loss_log": []}
    Xtr, Ytr = d["X_train"], d["y_train"]
    t0 = time.perf_counter()
    for ep, order in enumerate(orders):
        acc = 0.0
        for idx in order:
            p, loss = net.train_sample(Xtr[idx], Ytr[idx], lr)
            acc += loss
            if log:
                hist["prob_log"].append([float(v) for v in p])
                hist["loss_log"].append(loss)
        hist["loss_train"].append(acc / len(order))
        lv, av, _ = net.evaluate(d["X_val"], d["y_val"])
        _, at, _ = net.evaluate(Xtr, Ytr)
        hist["loss_val"].append(lv); hist["acc_val"].append(av); hist["acc_train"].append(at)
        if log:
            print(f"  epoca {ep:3d}  loss_tr={hist['loss_train'][-1]:.5f}  "
                  f"loss_val={lv:.5f}  acc_val={av:5.1f}%")
    hist["wall_time_s"] = time.perf_counter() - t0
    return hist


def time_forward(net, X, reps=20):
    t0 = time.perf_counter()
    for _ in range(reps):
        for x in X:
            net.forward(x)
    return 1e6 * (time.perf_counter() - t0) / (reps * len(X))


def frontera(net, patron=(1.0, 1.0, 0.0), canal=2, n=121, seed=3):
    """Barre el nivel de un potenciometro manteniendo los otros fijos y
    devuelve P(clase) contra el nivel: la frontera de decision real."""
    r = np.random.default_rng(seed)
    niveles = np.linspace(0.0, 1.0, n)
    probs = []
    for v in niveles:
        lv = list(patron); lv[canal] = v
        raw = np.clip(np.rint(np.asarray(lv) * core.ADC_FS +
                              r.normal(0, 5.0, size=(core.MA_LEN + core.WIN - 1, 3))),
                      0, core.ADC_FS)
        probs.append(net.forward(core.features_from_raw(raw))[0].tolist())
    return {"niveles": niveles.tolist(), "probs": probs, "patron": list(patron),
            "canal": canal}


def run(dtype, tag, d, orders):
    net = MLP(seed=SEED_WEIGHTS, dtype=dtype)
    w0 = net.get_weights()
    print(f"\n[{tag}] dtype={np.dtype(dtype).name}  topologia={net.sizes}  "
          f"{net.n_params()} parametros")
    hist = train(net, d, orders, LR)
    _, acc_te, Pte = net.evaluate(d["X_test"], d["y_test"])
    yp = Pte.argmax(axis=1)
    M = matriz_confusion(d["y_test"], yp)
    hist.update({
        "acc_test": acc_te, "cm_test": M.tolist(), "metricas_test": metricas_clase(M),
        "prob_test": Pte.tolist(),
        "time_fwd_us": time_forward(net, d["X_test"][:50]),
        "time_per_train_sample_us": 1e6 * hist["wall_time_s"] / (EPOCHS * len(d["X_train"])),
        "weights_init": w0, "weights_final": net.get_weights(),
        "config": {"epochs": EPOCHS, "lr": LR, "dtype": np.dtype(dtype).name,
                   "seed_weights": SEED_WEIGHTS, "seed_data": SEED_DATA,
                   "seed_order": SEED_ORDER, "n_por_clase": N_POR_CLASE,
                   "topologia": list(net.sizes), "n_params": net.n_params(),
                   "ma_len": core.MA_LEN, "win": core.WIN},
        "frontera": frontera(net),
    })
    # Exactitud por subtipo en test (donde falla una regla de umbral)
    sub = d["sub_test"]
    hist["acc_por_subtipo"] = {s: float(np.mean(yp[sub == s] == d["y_test"][sub == s]))
                               for s in np.unique(sub)}
    return net, hist


def estudio_reduccion(d, orders, net_ref):
    """Tamano del modelo contra exactitud: justifica la topologia elegida y
    responde a 'reducir el modelo o cuantizar si es necesario'."""
    casos = [
        ("9-8-6-4 (firmware)", (9, 8, 6, 4), None),
        ("9-6-4", (9, 6, 4), None),
        ("9-4 (sin ocultas)", (9, 4), None),
        ("3-8-6-4 (solo medias)", (3, 8, 6, 4), [0, 1, 2]),
        ("6-8-6-4 (media+desv)", (6, 8, 6, 4), [0, 1, 2, 3, 4, 5]),
        ("3-4-4 (medias, mínima)", (3, 4, 4), [0, 1, 2]),
    ]
    out = []
    for nombre, topo, idx in casos:
        net = MLP(topologia=topo, seed=SEED_WEIGHTS, feat_idx=idx)
        train(net, d, orders, LR, log=False)
        _, acc, P = net.evaluate(d["X_test"], d["y_test"])
        M = matriz_confusion(d["y_test"], P.argmax(axis=1))
        macs = sum(a * b for a, b in zip(topo[:-1], topo[1:]))
        out.append({"nombre": nombre, "topologia": list(topo), "n_params": net.n_params(),
                    "bytes_f32": 4 * net.n_params(), "macs": macs, "acc_test": acc,
                    "macro_f1": metricas_clase(M)["macro_f1"],
                    "acc_default": float(M[0, 0] / M[0].sum())})
        print(f"  {nombre:26s} {net.n_params():4d} params  acc_test={acc:5.1f}%  "
              f"F1={out[-1]['macro_f1']:.3f}")

    # Cuantizacion int8 post-entrenamiento del modelo del firmware
    q = cuantizar_int8(net_ref)
    _, acc_q, Pq = q.evaluate(d["X_test"], d["y_test"])
    _, acc_f, Pf = net_ref.evaluate(d["X_test"], d["y_test"])
    quant = {"acc_f32": acc_f, "acc_int8": acc_q,
             "max_dif_prob": float(np.max(np.abs(Pq - Pf))),
             "acuerdo_clase": float(np.mean(Pq.argmax(1) == Pf.argmax(1))),
             "bytes_f32": 4 * net_ref.n_params(),
             "bytes_int8": net_ref.n_params() + 4 * 2 * len(net_ref.W)}
    print(f"  int8: acc={acc_q:.1f}% (f32 {acc_f:.1f}%), max|dp|={quant['max_dif_prob']:.2e}")

    yb = clasificador_umbral(d["X_test"])
    Mb = matriz_confusion(d["y_test"], yb)
    base = {"acc_test": 100.0 * float(np.mean(yb == d["y_test"])), "cm_test": Mb.tolist(),
            "metricas": metricas_clase(Mb)}
    sub = d["sub_test"]
    base["acc_por_subtipo"] = {s: float(np.mean(yb[sub == s] == d["y_test"][sub == s]))
                               for s in np.unique(sub)}
    print(f"  regla de umbral: acc_test={base['acc_test']:.1f}%")
    return {"topologias": out, "int8": quant, "umbral": base}


def guardar_csv(d, path="dataset_sintetico.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "nombre_clase", "subtipo", "nivel1", "nivel2", "nivel3"]
                   + core.FEAT_NAMES)
        for x, y, s, lv in zip(d["X_all"], d["y_all"], d["sub_all"], d["lv_all"]):
            w.writerow([int(y), core.CLASES[int(y)], s] + [f"{v:.4f}" for v in lv]
                       + [f"{v:.6f}" for v in x])
    print(f"Dataset anotado: {path}")


def main():
    print("Gradient check (analitico vs diferencias centradas)")
    gc = gradient_check()
    print(f"  error relativo maximo = {gc:.3e}")

    d = split_dataset(N_POR_CLASE, SEED_DATA)
    guardar_csv(d)
    orders = epoch_orders(len(d["X_train"]), EPOCHS, SEED_ORDER)

    net64, h64 = run(np.float64, "PC float64", d, orders)
    h64["gradient_check"] = gc
    h64["split"] = {k: d[k] for k in ("X_train", "y_train", "X_val", "y_val",
                                       "X_test", "y_test", "sub_test")}
    h64["orders"] = orders
    save_json("resultados_pc.json", h64)

    _, h32 = run(np.float32, "PC float32", d, orders)
    save_json("resultados_pc_f32.json", h32)

    print("\nEstudio de reduccion del modelo")
    save_json("resultados_reduccion.json", estudio_reduccion(d, orders, net64))

    os.makedirs(FIRMWARE_DIR, exist_ok=True)
    path = os.path.join(FIRMWARE_DIR, "pesos_iniciales.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write(MLP(seed=SEED_WEIGHTS).export_c("INIT"))
    print(f"\nPesos iniciales para el firmware: {path}")


if __name__ == "__main__":
    main()
