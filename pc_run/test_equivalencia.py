"""
test_equivalencia.py
====================
Pruebas en PC del codigo C del firmware (preproc.c, mlp.c) contra el modelo
de referencia en Python, usando el gemelo tests/host_mcu.

  1) Preprocesado: se inyectan series crudas identicas (comando A) y se
     comparan las 9 caracteristicas que calcula el C contra mlp_core.
  2) Red: se entrena una epoca por el protocolo y se compara la salida del C
     contra MLP(float32) muestra a muestra.

Uso (desde tests/):
    gcc -O2 -std=c99 -I../firmware host_mcu.c ../firmware/preproc.c \
        ../firmware/mlp.c ../firmware/pi_ctrl.c -lm -o host_mcu
    python test_equivalencia.py
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import mlp_core as core  # noqa: E402

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "host_mcu")


def run(cmds):
    p = subprocess.run([EXE], input="\n".join(cmds + ["Q"]) + "\n",
                       capture_output=True, text=True, check=True)
    return [l for l in p.stdout.splitlines() if l]


def test_preproc(n=200, seed=11):
    r = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n):
        T = core.MA_LEN + core.WIN - 1 + int(r.integers(0, 40))
        lv = r.random(3)
        raw = np.clip(np.rint(lv * 4095 + r.normal(0, 30, (T, 3))), 0, 4095).astype(int)
        out = run([f"A,{a},{b},{c}" for a, b, c in raw] + ["C"])
        f_c = np.array([core.hex_to_f32(h) for h in out[-1].split(",")[3:]])
        f_py = core.features_from_raw(raw)
        max_err = max(max_err, float(np.max(np.abs(f_c - f_py))))
    return max_err


def test_red(n=300):
    d = core.split_dataset()
    net = core.MLP(dtype=np.float32)
    cmds, ref = [], []
    for x, y in zip(d["X_train"][:n].astype(np.float32), d["y_train"][:n]):
        cmds.append("T," + ",".join(core.f32_to_hex(v) for v in x) + f",{y}")
        p, _ = net.train_sample(x, y, 0.05)
        ref.append(p)
    out = run(cmds)
    pc = np.array([[core.hex_to_f32(h) for h in l.split(",")[1:5]] for l in out])
    return float(np.max(np.abs(pc - np.array(ref))))


if __name__ == "__main__":
    e1 = test_preproc()
    print(f"Preprocesado  C vs Python : error max = {e1:.2e}")
    e2 = test_red()
    print(f"Red (300 pasos de SGD)    : error max en p = {e2:.2e}")
    ok = e1 < 1e-5 and e2 < 1e-5
    print("OK" if ok else "FALLO")
    sys.exit(0 if ok else 1)
