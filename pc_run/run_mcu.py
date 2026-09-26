"""
run_mcu.py
==========
Entrena la red EN EL MICROCONTROLADOR con exactamente el mismo dataset, el
mismo orden de muestras y los mismos pesos iniciales que run_pc.py.

Los floats viajan como hex IEEE-754 (8 digitos): el MCU recibe el mismo
float32 que usa la PC y devuelve sus probabilidades y perdida sin redondeo
de texto. Asi cualquier diferencia PC-MCU es de la aritmetica, no del
formato.

Al terminar:
  - evalua el conjunto de prueba en modo 'V' (forward sin actualizar),
  - descarga los pesos entrenados ('W') y genera ../firmware/pesos_entrenados.h
    (compilar con -DUSE_PRETRAINED para un binario autonomo),
  - guarda resultados_mcu.json.

Uso:
    python run_mcu.py --puerto COM6          # hardware real
    python run_mcu.py --host                 # gemelo en C (tests/host_mcu)
    python run_mcu.py --simular              # emulacion float32 en NumPy
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

import mlp_core as core
from mlp_core import (MLP, split_dataset, epoch_orders, save_json, f32_to_hex,
                      hex_to_f32, matriz_confusion, metricas_clase)
from run_pc import EPOCHS, LR, SEED_WEIGHTS, SEED_DATA, SEED_ORDER, N_POR_CLASE

PORT = "COM6"               # Linux: /dev/ttyACM0
BAUDRATE = 115200           # debe coincidir con la config de LPUART6
READ_TIMEOUT_S = 2.0
CORE_MHZ = 120.0
AQUI = os.path.dirname(os.path.abspath(__file__))
FIRMWARE_DIR = os.path.join(AQUI)
HOST_BIN = os.path.join(AQUI, "..", "tests", "host_mcu")


# =====================================================================
# Transportes: todos exponen line(cmd) -> str  y  lines_until(cmd, fin)
# =====================================================================
class SerialLink:
    def __init__(self, port, baud):
        import serial   # solo se exige pyserial con hardware real
        self.ser = serial.Serial(port, baud, timeout=READ_TIMEOUT_S)
        time.sleep(0.5)
        self.ser.reset_input_buffer()
        self.ser.write(b"X\n")                   # asegura modo IDLE
        self._readline()
        self.ser.reset_input_buffer()

    def _readline(self):
        while True:
            s = self.ser.readline().decode("ascii", errors="replace").strip()
            if not s.startswith("D,"):            # ignora telemetria residual
                return s

    def line(self, cmd):
        self.ser.write((cmd + "\n").encode("ascii"))
        r = self._readline()
        if not r:
            raise TimeoutError(f"Sin respuesta del MCU a {cmd[:20]!r}")
        return r

    def lines_until(self, cmd, fin="END"):
        self.ser.write((cmd + "\n").encode("ascii"))
        out = []
        while True:
            r = self._readline()
            if not r:
                raise TimeoutError("volcado incompleto")
            if r == fin:
                return out
            out.append(r)

    def close(self):
        self.ser.close()


class HostLink(SerialLink):
    """Mismo protocolo contra el gemelo en C compilado para PC."""

    def __init__(self, exe=HOST_BIN):
        if not os.path.exists(exe):
            raise FileNotFoundError(f"Compila primero {exe} (ver tests/README)")
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)

    def _readline(self):
        return self.p.stdout.readline().strip()

    def line(self, cmd):
        self.p.stdin.write(cmd + "\n"); self.p.stdin.flush()
        return self._readline()

    def lines_until(self, cmd, fin="END"):
        self.p.stdin.write(cmd + "\n"); self.p.stdin.flush()
        out = []
        while True:
            r = self._readline()
            if r == fin:
                return out
            out.append(r)

    def close(self):
        self.p.stdin.write("Q\n"); self.p.stdin.flush(); self.p.wait(timeout=5)


class SimLink:
    """Emulacion float32 en NumPy (no ejercita el codigo C)."""

    def __init__(self):
        self.net = MLP(seed=SEED_WEIGHTS, dtype=np.float32)

    def line(self, cmd):
        if cmd == "R":
            self.net = MLP(seed=SEED_WEIGHTS, dtype=np.float32); return "OK"
        parts = cmd.split(",")
        x = np.array([hex_to_f32(h) for h in parts[1:1 + core.N_FEAT]], dtype=np.float32)
        y = int(parts[-1])
        p, acts = self.net.forward(x)
        loss, _ = self.net.backward(p, acts, y, LR, update=(parts[0] == "T"))
        return ",".join([str(int(np.argmax(p)))] + [f32_to_hex(v) for v in p]
                        + [f32_to_hex(loss), "0", "0"])

    def lines_until(self, cmd, fin="END"):
        flat = self.net.flat_params()
        return [f"W,{i}," + ",".join(f32_to_hex(v) for v in flat[i:i + 8])
                for i in range(0, len(flat), 8)]

    def close(self):
        pass


# =====================================================================
def muestra(link, modo, x, y):
    cmd = f"{modo}," + ",".join(f32_to_hex(v) for v in x) + f",{int(y)}"
    t0 = time.perf_counter()
    r = link.line(cmd)
    rtt = time.perf_counter() - t0
    # --- ADD THIS PRINT STATEMENT FOR DEBUGGING ---
    print(f"RAW MCU RESPONSE: {r!r}")
    # ---------------------------------------------
    if r == "E":
        raise RuntimeError(f"El MCU rechazo la trama ({modo}); ¿esta en modo IDLE?")
    f = r.split(",")
    probs = [hex_to_f32(h) for h in f[1:5]]
    return int(f[0]), probs, hex_to_f32(f[5]), int(f[6]), int(f[7]), rtt


def evaluar(link, X, Y):
    P, loss, cf = [], 0.0, []
    for x, y in zip(X, Y):
        _, p, l, c_f, _, _ = muestra(link, "V", x, y)
        P.append(p); loss += l; cf.append(c_f)
    P = np.array(P)
    return loss / len(X), 100.0 * float(np.mean(P.argmax(1) == Y)), P, cf


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--simular", action="store_true", help="emulacion NumPy float32")
    g.add_argument("--host", action="store_true", help="gemelo en C (tests/host_mcu)")
    ap.add_argument("--puerto", default=PORT)
    ap.add_argument("--baud", type=int, default=BAUDRATE)
    ap.add_argument("--epocas", type=int, default=EPOCHS)
    args = ap.parse_args()

    d = split_dataset(N_POR_CLASE, SEED_DATA)
    orders = epoch_orders(len(d["X_train"]), EPOCHS, SEED_ORDER)[:args.epocas]
    Xtr = d["X_train"].astype(np.float32); Ytr = d["y_train"]

    n_tx = len("T,") + 9 * core.N_FEAT + 2
    n_rx = 2 + 5 * 9 + 12
    t_link = (n_tx + n_rx) * 10.0 / args.baud
    print(f"Dataset: {len(Xtr)} train / {len(d['X_val'])} val / {len(d['X_test'])} test")
    print(f"Enlace estimado: {1e3 * t_link:.1f} ms por muestra -> "
          f"~{t_link * args.epocas * (len(Xtr) + len(d['X_val'])) / 60:.1f} min\n")

    if args.simular:
        link, origen = SimLink(), "simulado"
    elif args.host:
        link, origen = HostLink(), "host_c"
    else:
        link, origen = SerialLink(args.puerto, args.baud), "hardware"

    hist = {"loss_train": [], "loss_val": [], "acc_val": [], "prob_log": [],
            "loss_log": [], "rtt_log": [], "cyc_fwd": [], "cyc_bwd": []}
    try:
        assert link.line("R") == "OK", "el MCU no respondio a R"
        t0 = time.perf_counter()
        for ep, order in enumerate(orders):
            acc = 0.0
            for idx in order:
                _, p, loss, cf, cb, rtt = muestra(link, "T", Xtr[idx], Ytr[idx])
                acc += loss
                hist["prob_log"].append(p); hist["loss_log"].append(loss)
                hist["rtt_log"].append(rtt); hist["cyc_fwd"].append(cf); hist["cyc_bwd"].append(cb)
            hist["loss_train"].append(acc / len(order))
            lv, av, _, _ = evaluar(link, d["X_val"].astype(np.float32), d["y_val"])
            hist["loss_val"].append(lv); hist["acc_val"].append(av)
            print(f"  epoca {ep:3d}  loss_tr={hist['loss_train'][-1]:.5f}  "
                  f"loss_val={lv:.5f}  acc_val={av:5.1f}%")
        hist["wall_time_s"] = time.perf_counter() - t0

        _, acc_te, Pte, cf_te = evaluar(link, d["X_test"].astype(np.float32), d["y_test"])
        M = matriz_confusion(d["y_test"], Pte.argmax(1))
        hist.update({"acc_test": acc_te, "cm_test": M.tolist(),
                     "metricas_test": metricas_clase(M), "prob_test": Pte.tolist()})
        print(f"\nExactitud en prueba: {acc_te:.1f} %\n{M}")

        flat = []
        for ln in link.lines_until("W"):
            flat += [hex_to_f32(h) for h in ln.split(",")[2:]]
        net = MLP(seed=SEED_WEIGHTS, dtype=np.float32)
        net.set_flat_params(np.array(flat, dtype=np.float32))
        hist["weights_final"] = net.get_weights()
        with open(os.path.join(FIRMWARE_DIR, "pesos_entrenados.h"), "w", encoding="utf-8") as f:
            f.write(net.export_c("TRAINED"))
        print("Pesos entrenados -> firmware/pesos_entrenados.h")
    finally:
        link.close()

    rtt = np.array(hist["rtt_log"])
    hist["rtt_mean_ms"] = float(rtt.mean() * 1e3)
    cf = np.array(hist["cyc_fwd"]); cb = np.array(hist["cyc_bwd"])
    if cf.any():
        hist["cyc_fwd_mean"] = float(cf.mean()); hist["cyc_bwd_mean"] = float(cb.mean())
        hist["cyc_fwd_max"] = int(cf.max()); hist["cyc_bwd_max"] = int(cb.max())
        hist["fwd_mean_us"] = hist["cyc_fwd_mean"] / CORE_MHZ
        hist["bwd_mean_us"] = hist["cyc_bwd_mean"] / CORE_MHZ
        print(f"Forward: {hist['fwd_mean_us']:.2f} us   Backward: {hist['bwd_mean_us']:.2f} us "
              f"(media, {CORE_MHZ:.0f} MHz)")
    hist["config"] = {"epochs": len(orders), "lr": LR, "baud": args.baud,
                      "origen": origen, "core_mhz": CORE_MHZ}
    save_json("resultados_mcu.json", hist)
    print(f"Ida y vuelta: {hist['rtt_mean_ms']:.2f} ms/muestra.  -> resultados_mcu.json")


if __name__ == "__main__":
    sys.exit(main())
