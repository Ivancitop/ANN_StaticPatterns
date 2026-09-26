"""
simulacion_lazo.py
==================
Simulacion en PC del lazo completo tal como lo ejecuta el firmware:

  potes -> ADC (12 b, ruido) -> media movil -> ventana -> 9 features
        -> red (forward, cada 50 ms) -> umbral de confianza -> antirrebote
        -> referencia -> PI (cada 10 ms, anti-windup) -> puente H -> motor
        -> encoder x4 cuantizado -> filtro de velocidad -> PI

Sirve para dos cosas ANTES del laboratorio:
  1) Disenar las ganancias del PI sobre un modelo de primer orden del motor
     (K, tau) y ver sobreimpulso / tiempo de establecimiento esperados.
  2) Estimar la latencia deteccion -> accion que imponen la ventana, el
     periodo de clasificacion y el antirrebote.

El modelo del motor es SUPUESTO (K, tau abajo). Los valores reales se
obtienen con `python telemetria.py identificar` y se reemplazan aqui.

Genera resultados_sim.json.
"""

import numpy as np

import mlp_core as core
from mlp_core import MLP, load_json, save_json

# ---------- temporizacion del firmware (main.c) ----------
TICK = 0.005          # s
CTRL_DIV = 2          # PI cada 10 ms
CLS_DIV = 10          # red cada 50 ms
TS = TICK * CTRL_DIV
CONF_THRESHOLD = 0.70
DEBOUNCE_N = 3
REF_RPM = 120.0

# ---------- modelo supuesto del motor ----------
K_MOTOR = 200.0       # rpm por unidad de duty (u = 1 -> 200 rpm en estado estable)
TAU_MOTOR = 0.08      # s
ENC_CPR = 1320.0      # cuentas por vuelta (x4)
RPM_LPF_ALPHA = 0.5
DT_PLANT = 0.0005     # paso de integracion de la planta

# ---------- diseno del PI por cancelacion de polo ----------
# Lazo cerrado de primer orden con constante T_CL:  Ti = tau,  Kp = tau/(K T_CL)
T_CL = 0.10
KP = TAU_MOTOR / (K_MOTOR * T_CL)
KI = KP / TAU_MOTOR


class PI:
    """Copia literal de firmware/pi_ctrl.c"""

    def __init__(self, kp, ki, ts, umax=1.0):
        self.kp, self.ki, self.ts, self.umax = kp, ki, ts, umax
        self.reset()

    def reset(self):
        self.integ = 0.0

    def step(self, ref, meas):
        e = ref - meas
        u = self.kp * e + self.integ
        us = min(max(u, -self.umax), self.umax)
        if us == u or (u > self.umax and e < 0) or (u < -self.umax and e > 0):
            self.integ = min(max(self.integ + self.ki * self.ts * e, -self.umax), self.umax)
        return us


class Motor:
    def __init__(self):
        self.w = 0.0          # rpm
        self.pos = 0.0        # vueltas

    def advance(self, u, T):
        n = int(round(T / DT_PLANT))
        for _ in range(n):
            self.w += DT_PLANT / TAU_MOTOR * (K_MOTOR * u - self.w)
            self.pos += self.w / 60.0 * DT_PLANT

    def counts(self):
        return int(np.floor(self.pos * ENC_CPR))


def perfil_potes(t, eventos, rampa=0.15):
    """eventos: lista (t_inicio, niveles). Entre patrones, rampa lineal."""
    lv = np.array(eventos[0][1], dtype=float)
    for k in range(1, len(eventos)):
        t0, nuevo = eventos[k]
        prev = np.array(eventos[k - 1][1], dtype=float)
        if t >= t0 + rampa:
            lv = np.array(nuevo, dtype=float)
        elif t >= t0:
            a = (t - t0) / rampa
            lv = prev + a * (np.array(nuevo) - prev)
    return lv


def simular(net, eventos, T_total, pi_gains=(KP, KI), fixed_ref=None, seed=5):
    r = np.random.default_rng(seed)
    pi = PI(*pi_gains, TS)
    mot = Motor()
    ma, win = [], []
    cls_raw = cls_cand = cls_act = 0
    cand = 0
    enc_prev = 0
    rpm_f = 0.0
    ref = 0.0
    u = 0.0
    log = {k: [] for k in ("t", "lv1", "lv2", "lv3", "cls_raw", "cls", "ref", "rpm", "u", "pmax")}
    pmax = 0.0
    n_ticks = int(T_total / TICK)
    for k in range(1, n_ticks + 1):
        t = k * TICK
        lv = perfil_potes(t, eventos)
        raw = np.clip(np.rint(lv * core.ADC_FS + r.normal(0, 6, 3)), 0, core.ADC_FS)
        ma.append(raw); ma = ma[-core.MA_LEN:]
        if len(ma) == core.MA_LEN:
            win.append(np.sum(ma, axis=0) / (core.MA_LEN * core.ADC_FS)); win = win[-core.WIN:]

        if fixed_ref is None and k % CLS_DIV == 0 and len(win) == core.WIN:
            x = core.features_from_filtered(np.array(win)).astype(np.float32)
            p, _ = net.forward(x)
            c = int(np.argmax(p)); pmax = float(p[c])
            cls_raw = c if pmax >= CONF_THRESHOLD else 0
            if cls_raw == cls_cand:
                cand += 1
            else:
                cls_cand, cand = cls_raw, 1
            if cand >= DEBOUNCE_N and cls_act != cls_cand:
                cls_act = cls_cand
                if cls_act == 0:
                    pi.reset()

        if k % CTRL_DIV == 0:
            enc = mot.counts()
            rpm = (enc - enc_prev) * 60.0 / (ENC_CPR * TS)
            enc_prev = enc
            rpm_f += RPM_LPF_ALPHA * (rpm - rpm_f)
            if fixed_ref is not None:
                ref = fixed_ref(t)
                u = pi.step(ref, rpm_f)
            else:
                ref = {1: REF_RPM, 2: -REF_RPM, 3: 0.0}.get(cls_act, 0.0)
                if cls_act == 0:
                    pi.reset(); u = 0.0
                else:
                    u = pi.step(ref, rpm_f)
            for key, v in zip(log, (t, *lv, cls_raw, cls_act, ref, rpm_f, u, pmax)):
                log[key].append(float(v))
        mot.advance(u, TICK)
    return log


def metricas_escalon(t, y, ref, t0, t_fin=None, banda=0.02):
    """Sobreimpulso, tiempo de subida 10-90 % y de establecimiento (banda)."""
    t = np.asarray(t); y = np.asarray(y)
    m = (t >= t0) if t_fin is None else ((t >= t0) & (t < t_fin))
    tt, yy = t[m] - t0, y[m]
    y0 = yy[0]
    dy = ref - y0
    s = np.sign(dy) if dy != 0 else 1.0
    yn = (yy - y0) / dy
    mp = max(0.0, float(np.max(s * (yy - ref)) / abs(dy) * 100.0))
    try:
        tr = float(tt[np.argmax(yn >= 0.9)] - tt[np.argmax(yn >= 0.1)])
    except ValueError:
        tr = float("nan")
    fuera = np.where(np.abs(yy - ref) > banda * abs(dy))[0]
    ts = float(tt[fuera[-1] + 1]) if len(fuera) and fuera[-1] + 1 < len(tt) else float("nan")
    ess = float(np.mean(yy[-20:]) - ref)
    return {"sobreimpulso_pct": mp, "t_subida_s": tr, "t_establecimiento_s": ts,
            "error_ss_rpm": ess}


def latencias(log, eventos, rampa=0.15):
    """Para cada cambio de patron: t desde que los potes quedan quietos hasta
    que cambia la clase activa, y hasta que la velocidad entra al 10 % de la
    nueva referencia."""
    t = np.array(log["t"]); cls = np.array(log["cls"]); rpm = np.array(log["rpm"])
    ref = np.array(log["ref"])
    out = []
    for k in range(1, len(eventos)):
        t_fin_mov = eventos[k][0] + rampa
        m = t >= eventos[k][0]
        cls0 = cls[np.argmax(t >= eventos[k][0]) - 1]
        idx = np.where(m & (cls != cls0))[0]
        if not len(idx):
            continue
        # primer cambio puede ser a default (pote en movimiento); se busca la clase final
        c_fin = cls[min(np.argmax(t >= (eventos[k + 1][0] if k + 1 < len(eventos) else t[-1])) - 1, len(cls) - 1)]
        idx_fin = np.where(m & (cls == c_fin))[0]
        t_det = float(t[idx_fin[0]] - t_fin_mov)
        r_new = ref[idx_fin[0]]
        if abs(r_new) > 0:
            ok = np.where((t >= t[idx_fin[0]]) & (np.abs(rpm - r_new) <= 0.1 * abs(r_new)))[0]
            t_acc = float(t[ok[0]] - t[idx_fin[0]]) if len(ok) else float("nan")
        else:
            ok = np.where((t >= t[idx_fin[0]]) & (np.abs(rpm) <= 0.1 * REF_RPM))[0]
            t_acc = float(t[ok[0]] - t[idx_fin[0]]) if len(ok) else float("nan")
        out.append({"t_evento": eventos[k][0], "clase_final": int(c_fin),
                    "det_desde_fin_movimiento_s": t_det, "accion_s": t_acc,
                    "total_s": t_det + t_acc})
    return out


def main():
    pc = load_json("resultados_pc.json")
    net = MLP(dtype=np.float32); net.set_weights(pc["weights_final"])

    H, L = 0.85, 0.15
    eventos = [(0.0, (L, L, L)), (0.5, (H, H, L)), (2.5, (H, L, H)),
               (4.5, (L, H, H)), (6.5, (H, H, H))]
    log = simular(net, eventos, 8.0)
    lat = latencias(log, eventos)

    # Escalon puro del PI (modo FIXEDREF), sin clasificador
    esc = simular(net, [(0.0, (L, L, L))], 1.5, fixed_ref=lambda t: REF_RPM if t >= 0.2 else 0.0)
    m_esc = metricas_escalon(esc["t"], esc["rpm"], REF_RPM, 0.2)

    # Barrido de T_CL: compromiso rapidez / sobreimpulso con cuantizacion real
    barrido = []
    for tcl in (0.03, 0.05, 0.08, 0.10, 0.15, 0.25):
        kp = TAU_MOTOR / (K_MOTOR * tcl); ki = kp / TAU_MOTOR
        e = simular(net, [(0.0, (L, L, L))], 1.5, pi_gains=(kp, ki),
                    fixed_ref=lambda t: REF_RPM if t >= 0.2 else 0.0)
        mm = metricas_escalon(e["t"], e["rpm"], REF_RPM, 0.2)
        mm.update({"t_cl": tcl, "kp": kp, "ki": ki,
                   "u_pico": float(np.max(np.abs(e["u"])))})
        barrido.append(mm)

    # Barrido del antirrebote: latencia contra conmutaciones de la clase activa
    global DEBOUNCE_N
    base_db = DEBOUNCE_N
    barrido_db = []
    for db in (1, 2, 3, 5):
        DEBOUNCE_N = db
        lg = simular(net, eventos, 8.0, seed=9)
        la = latencias(lg, eventos)
        c = np.array(lg["cls"])
        barrido_db.append({"debounce": db, "conmutaciones": int(np.sum(c[1:] != c[:-1])),
                           "det_media_ms": 1e3 * float(np.mean([l["det_desde_fin_movimiento_s"]
                                                                for l in la[:3]]))})
    DEBOUNCE_N = base_db

    save_json("resultados_sim.json", {"barrido_debounce": barrido_db,
        "modelo": {"K": K_MOTOR, "tau": TAU_MOTOR, "enc_cpr": ENC_CPR, "t_cl": T_CL,
                   "kp": KP, "ki": KI},
        "escenario": log, "eventos": [[e[0], list(e[1])] for e in eventos],
        "latencias": lat, "escalon": esc, "metricas_escalon": m_esc, "barrido_tcl": barrido})

    print(f"PI: Kp={KP:.5f}  Ki={KI:.5f}  (T_cl={T_CL}s, modelo K={K_MOTOR}, tau={TAU_MOTOR})")
    print("Escalon 0->120 rpm:", {k: round(v, 4) for k, v in m_esc.items()})
    for l in lat:
        print(f"  evento t={l['t_evento']}: clase {core.CLASES[l['clase_final']]:7s} "
              f"deteccion {1e3 * l['det_desde_fin_movimiento_s']:.0f} ms, "
              f"accion {1e3 * l['accion_s']:.0f} ms")
    for b in barrido_db:
        print(f"  antirrebote={b['debounce']}  conmutaciones={b['conmutaciones']}  "
              f"deteccion media={b['det_media_ms']:.0f} ms")
    for b in barrido:
        print(f"  T_cl={b['t_cl']:.2f}  Mp={b['sobreimpulso_pct']:.1f}%  "
              f"ts={b['t_establecimiento_s']:.3f}s  u_pico={b['u_pico']:.2f}")


if __name__ == "__main__":
    main()
