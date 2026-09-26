"""
telemetria.py
=============
Pruebas de control y latencia con el hardware. Registra la telemetria que el
firmware emite cada 10 ms en los modos RUN / FIXEDREF / OPENLOOP:

    D,t_ms,clsRaw,cls,ref,rpm*10,u*1000,raw1,raw2,raw3,pmax*1000,carga*1000,cicF

Subcomandos:
  identificar --duty 500 --seg 2     escalon en lazo abierto -> K y tau del motor
                                     y ganancias PI sugeridas (cancelacion de polo)
  escalon --rpm 120 --seg 2          escalon de referencia con el PI (sin red)
                                     -> sobreimpulso, t. subida, t. establecimiento
  demo --seg 60                      modo RUN completo: mueve los potes entre
                                     patrones; mide latencia deteccion -> accion,
                                     carga de CPU y ciclos del forward
  analizar archivo.csv [--tipo ...]  recalcula metricas de un registro guardado

  ganancias --kp 0.004 --ki 0.05     cambia el PI en caliente (comando P)

Cada registro se guarda como CSV (tlm_<tipo>_<fecha>.csv).
"""

import argparse
import csv
import json
import sys
import time

import numpy as np



COLS = ["t_ms", "cls_raw", "cls", "ref", "rpm", "u", "raw1", "raw2", "raw3",
        "pmax", "carga", "cic_fwd"]
ESCALA = {"rpm": 0.1, "u": 1e-3, "pmax": 1e-3, "carga": 1e-3}


def abrir(puerto, baud):
    import serial
    ser = serial.Serial(puerto, baud, timeout=0.5)
    time.sleep(0.3); ser.write(b"X\n"); time.sleep(0.1); ser.reset_input_buffer()
    return ser


def comando(ser, cmd):
    ser.write((cmd + "\n").encode("ascii"))
    t_end = time.time() + 1.0
    while time.time() < t_end:
        r = ser.readline().decode("ascii", errors="replace").strip()
        if r in ("OK", "E"):
            return r
    return ""


def registrar(ser, cmd, seg, tipo):
    assert comando(ser, cmd) == "OK", f"el MCU rechazo {cmd}"
    filas, t_end = [], time.time() + seg
    while time.time() < t_end:
        r = ser.readline().decode("ascii", errors="replace").strip()
        if r.startswith("D,"):
            v = r.split(",")[1:]
            if len(v) == len(COLS):
                filas.append([float(x) * ESCALA.get(c, 1.0) for c, x in zip(COLS, v)])
    comando(ser, "X")
    nombre = f"tlm_{tipo}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(nombre, "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(filas)
    print(f"{len(filas)} muestras -> {nombre}")
    return nombre


def leer(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        rows = [{k: float(v) for k, v in r.items()} for r in rd]
    return {c: np.array([r[c] for r in rows]) for c in COLS}


# ---------------------------------------------------------------------
def metricas_escalon(t, y, ref, banda=0.02):
    t = t - t[0]
    y0 = y[0]; dy = ref - y0
    yn = (y - y0) / dy
    mp = max(0.0, float(np.max(np.sign(dy) * (y - ref)) / abs(dy) * 100))
    tr = float(t[np.argmax(yn >= 0.9)] - t[np.argmax(yn >= 0.1)])
    fuera = np.where(np.abs(y - ref) > banda * abs(dy))[0]
    ts = float(t[fuera[-1] + 1]) if len(fuera) and fuera[-1] + 1 < len(t) else float("nan")
    return {"sobreimpulso_pct": mp, "t_subida_s": tr, "t_establecimiento_2pct_s": ts,
            "error_ss": float(np.mean(y[-30:]) - ref), "rizado_ss": float(np.std(y[-30:]))}


def analizar_escalon(d):
    t = d["t_ms"] / 1e3
    m = metricas_escalon(t, d["rpm"], d["ref"][-1])
    m["u_pico"] = float(np.max(np.abs(d["u"])))
    return m


def analizar_identificacion(d, t_cl=0.1):
    """Primer orden: K = w_ss / u,  tau = tiempo al 63.2 %."""
    t = (d["t_ms"] - d["t_ms"][0]) / 1e3
    u = float(np.median(d["u"]))
    wss = float(np.mean(d["rpm"][-50:]))
    tau = float(t[np.argmax(d["rpm"] / wss >= 0.632)])
    K = wss / u
    kp = tau / (K * t_cl); ki = kp / tau
    return {"u": u, "rpm_ss": wss, "K_rpm_por_u": K, "tau_s": tau,
            "t_cl_s": t_cl, "kp_sugerido": kp, "ki_sugerido": ki,
            "comando": f"P,{int(round(kp * 1e6))},{int(round(ki * 1e6))}"}


def analizar_demo(d, umbral_mov=60, quieto_ms=100):
    """Latencia por cambio de clase activa.

    fin de movimiento: ultimo instante antes del cambio en que algun pote
    varia mas de `umbral_mov` LSB entre muestras (10 ms). A partir de ahi:
      deteccion = t(cambio de clase activa) - fin de movimiento
      accion    = t(|rpm - ref| <= 10 % de |ref|) - t(cambio de clase)
    """
    t = d["t_ms"]; cls = d["cls"]
    raw = np.vstack([d["raw1"], d["raw2"], d["raw3"]]).T
    mov = np.any(np.abs(np.diff(raw, axis=0)) > umbral_mov, axis=1)
    mov = np.concatenate([[False], mov])
    lat = []
    for k in np.where(cls[1:] != cls[:-1])[0] + 1:
        if cls[k] == 0:
            continue                          # los default intermedios no cuentan
        prev_mov = np.where(mov[:k])[0]
        if not len(prev_mov):
            continue
        t_fin = t[prev_mov[-1]]
        ref = d["ref"][k]
        if abs(ref) > 0:
            ok = np.where((np.arange(len(t)) >= k) & (np.abs(d["rpm"] - ref) <= 0.1 * abs(ref)))[0]
        else:
            ok = np.where((np.arange(len(t)) >= k) & (np.abs(d["rpm"]) <= 10.0))[0]
        lat.append({"t_ms": float(t[k]), "clase": int(cls[k]),
                    "deteccion_ms": float(t[k] - t_fin),
                    "accion_ms": float(t[ok[0]] - t[k]) if len(ok) else float("nan")})
    dt = np.diff(t)
    res = {"latencias": lat,
           "deteccion_media_ms": float(np.mean([l["deteccion_ms"] for l in lat])) if lat else None,
           "accion_media_ms": float(np.nanmean([l["accion_ms"] for l in lat])) if lat else None,
           "carga_cpu_media_pct": float(100 * np.mean(d["carga"])),
           "carga_cpu_max_pct": float(100 * np.max(d["carga"])),
           "cic_fwd_media": float(np.mean(d["cic_fwd"][d["cic_fwd"] > 0])),
           "fwd_us": float(np.mean(d["cic_fwd"][d["cic_fwd"] > 0]) / 120.0),
           "periodo_tlm_ms_media": float(np.mean(dt)),
           "tramas_perdidas_pct": float(100 * np.mean(dt > 15))}
    return res


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["identificar", "escalon", "demo", "analizar", "ganancias"])
    ap.add_argument("archivo", nargs="?")
    ap.add_argument("--tipo", choices=["identificar", "escalon", "demo"], default="demo")
    ap.add_argument("--puerto", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seg", type=float, default=3.0)
    ap.add_argument("--duty", type=int, default=500)
    ap.add_argument("--rpm", type=int, default=120)
    ap.add_argument("--kp", type=float)
    ap.add_argument("--ki", type=float)
    ap.add_argument("--tcl", type=float, default=0.1)
    a = ap.parse_args()

    if a.accion == "analizar":
        d = leer(a.archivo)
        fn = {"identificar": lambda d: analizar_identificacion(d, a.tcl),
              "escalon": analizar_escalon, "demo": analizar_demo}[a.tipo]
        print(json.dumps(fn(d), indent=2)); return

    ser = abrir(a.puerto, a.baud)
    try:
        if a.accion == "ganancias":
            print(comando(ser, f"P,{int(round(a.kp * 1e6))},{int(round(a.ki * 1e6))}"))
            return
        if a.accion == "identificar":
            f = registrar(ser, f"O,{a.duty}", a.seg, "identificar")
            res = analizar_identificacion(leer(f), a.tcl)
        elif a.accion == "escalon":
            f = registrar(ser, f"F,{a.rpm}", a.seg, "escalon")
            res = analizar_escalon(leer(f))
        else:
            print("Modo RUN: cambia los potes entre patrones. Ctrl+C no es necesario.")
            f = registrar(ser, "I", a.seg, "demo")
            res = analizar_demo(leer(f))
        with open(f.replace(".csv", "_metricas.json"), "w") as g:
            json.dump(res, g, indent=2)
        print(json.dumps(res, indent=2))
        #d = list(csv.DictReader(open("tlm_demo_20260925_192609.csv")))
        #g = lambda k: np.array([float(r[k]) for r in d])
        #for k in ("raw1", "raw2", "raw3"):
        #    v = g(k); print(f"{k}: min {v.min():.0f}  max {v.max():.0f}")
        #cr = g("cls_raw").astype(int); pm = g("pmax")
        #print("cls_raw:", np.bincount(cr, minlength=4), " (Default, CW, CCW, Paro)")
        #print("pmax: media %.2f  max %.2f" % (pm.mean(), pm.max()))
    finally:
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
