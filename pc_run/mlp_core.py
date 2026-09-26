"""
mlp_core.py
===========
Modelo de referencia en PC del clasificador de patrones estaticos que corre
en el S32K312. Lo importan run_pc.py, run_mcu.py, capturar.py, evaluar.py y
graficas.py para que PC y firmware compartan EXACTAMENTE:

  - la definicion de clases y el generador del dataset sintetico,
  - el preprocesado (media movil + ventana + media/desv. estandar/maximo),
  - la normalizacion de las caracteristicas,
  - la topologia, las sigmoides parametricas y los pesos iniciales,
  - el orden de presentacion de las muestras en cada epoca.

Todo lo que aparece aqui como constante tiene un #define gemelo en
firmware/features.h o firmware/mlp.h. Si cambias uno, cambia el otro.
"""

import json
import struct

import numpy as np

# =====================================================================
# 1. Clases
# =====================================================================
# 0 = default (cualquier otra combinacion, niveles intermedios o potes en
#     movimiento) -> estado seguro: PWM apagado, integrador en cero
# 1 = pot1 H, pot2 H, pot3 L -> giro horario (CW)
# 2 = pot1 L, pot2 H, pot3 H -> giro antihorario (CCW)
# 3 = pot1 H, pot2 L, pot3 H -> sin movimiento (PI regulando a 0 rpm)
CLASES = ["Default", "CW", "CCW", "Paro"]
N_CLASES = 4
PATRONES = {1: "HHL", 2: "LHH", 3: "HLH"}
OTRAS_COMBINACIONES = ["HHH", "LLL", "HLL", "LHL", "LLH"]

# =====================================================================
# 2. Preprocesado (gemelo de firmware/features.h)
# =====================================================================
ADC_FS = 2700          # 12 bits
N_CH = 3
MA_LEN = 8             # media movil sobre la lectura cruda
WIN = 32               # ventana de caracteristicas (32 x 5 ms = 160 ms)
STD_REF = 0.02         # desv. estandar de referencia (2 % de escala)
STD_CLIP = 5.0         # tope de la desv. estandar normalizada
N_FEAT = 3 * N_CH      # [media x3, desv x3, max x3]
FEAT_NAMES = ([f"media{i+1}" for i in range(N_CH)] +
              [f"desv{i+1}" for i in range(N_CH)] +
              [f"max{i+1}" for i in range(N_CH)])

# Niveles que definen H y L (fraccion de escala completa)
NIVEL_L = (0.00, 0.30)
NIVEL_H = (0.70, 1.00)
NIVEL_MEDIO = (0.40, 0.60)   # entre 0.30-0.40 y 0.60-0.70 no se etiqueta


def moving_average(raw):
    """Media movil de MA_LEN muestras con acumulador entero, igual que el
    firmware: filt = suma / (MA_LEN * ADC_FS). raw: (T, 3) enteros."""
    raw = np.asarray(raw, dtype=np.int64)
    c = np.cumsum(raw, axis=0)
    c = np.vstack([np.zeros((1, raw.shape[1]), dtype=np.int64), c])
    s = c[MA_LEN:] - c[:-MA_LEN]
    return s / (MA_LEN * ADC_FS)


def features_from_filtered(win):
    """win: (WIN, 3) en [0, 1]. Devuelve el vector normalizado de 9."""
    win = np.asarray(win, dtype=np.float64)
    mean = win.mean(axis=0)
    # dos pasadas, igual que preproc.c (E[x^2]-media^2 pierde precision en float32)
    std = np.sqrt(((win - mean) ** 2).mean(axis=0))
    mx = win.max(axis=0)
    return np.concatenate([2.0 * mean - 1.0,
                           np.minimum(std / STD_REF, STD_CLIP),
                           2.0 * mx - 1.0])


def features_from_raw(raw):
    """Pipeline completo: crudo -> media movil -> ultima ventana -> features."""
    filt = moving_average(raw)
    assert len(filt) >= WIN, "serie demasiado corta"
    return features_from_filtered(filt[-WIN:])


# =====================================================================
# 3. Dataset sintetico
# =====================================================================
def _nivel(r, letra):
    lo, hi = NIVEL_H if letra == "H" else NIVEL_L
    return r.uniform(lo, hi)


def _serie(r, niveles_ini, niveles_fin=None):
    """Genera la lectura cruda de los 3 potenciometros durante MA_LEN+WIN-1
    muestras: nivel + ruido gaussiano + picos esporadicos (EMI del motor)."""
    T = MA_LEN + WIN - 1
    ini = np.asarray(niveles_ini, dtype=np.float64)
    fin = ini if niveles_fin is None else np.asarray(niveles_fin)
    ramp = np.linspace(0.0, 1.0, T)[:, None]
    nivel = ini + (fin - ini) * ramp
    sigma = r.uniform(2.0, 12.0)                         # LSB
    raw = nivel * ADC_FS + r.normal(0.0, sigma, size=(T, N_CH))
    picos = r.random((T, N_CH)) < 0.01
    raw += picos * r.choice([-1, 1], size=(T, N_CH)) * r.uniform(50, 250, size=(T, N_CH))
    return np.clip(np.rint(raw), 0, ADC_FS).astype(np.int64)


def build_dataset(n_por_clase=300, seed=0):
    """Devuelve X (N,9), y (N,), subtipo (N,) y niveles (N,3)."""
    r = np.random.default_rng(seed)
    X, y, sub, lv = [], [], [], []

    def add(raw, label, tipo, niveles):
        X.append(features_from_raw(raw)); y.append(label)
        sub.append(tipo); lv.append(niveles)

    for c, pat in PATRONES.items():
        for _ in range(n_por_clase):
            niv = [_nivel(r, ch) for ch in pat]
            add(_serie(r, niv), c, pat, niv)

    # Clase default: 60 % otras combinaciones H/L, 20 % algun pote en nivel
    # intermedio, 20 % algun pote girando dentro de la ventana.
    n_comb = int(0.6 * n_por_clase)
    n_mid = int(0.2 * n_por_clase)
    n_mov = n_por_clase - n_comb - n_mid
    for k in range(n_comb):
        pat = OTRAS_COMBINACIONES[k % len(OTRAS_COMBINACIONES)]
        niv = [_nivel(r, ch) for ch in pat]
        add(_serie(r, niv), 0, pat, niv)
    for _ in range(n_mid):
        pat = list(r.choice(["H", "L"], size=N_CH))
        niv = [_nivel(r, ch) for ch in pat]
        for j in r.choice(N_CH, size=r.integers(1, N_CH + 1), replace=False):
            niv[j] = r.uniform(*NIVEL_MEDIO)
        add(_serie(r, niv), 0, "medio", niv)
    for _ in range(n_mov):
        base = PATRONES[int(r.integers(1, 4))]
        ini = [_nivel(r, ch) for ch in base]
        fin = list(ini)
        j = int(r.integers(0, N_CH))
        fin[j] = _nivel(r, "L" if base[j] == "H" else "H")
        add(_serie(r, ini, fin), 0, "movimiento", ini)

    X = np.asarray(X); y = np.asarray(y, dtype=np.int64)
    idx = r.permutation(len(X))
    return X[idx], y[idx], np.asarray(sub)[idx], np.asarray(lv)[idx]


def split_dataset(n_por_clase=300, seed=0, frac=(0.70, 0.15, 0.15)):
    """Particion estratificada train / val / test."""
    X, y, sub, lv = build_dataset(n_por_clase, seed)
    r = np.random.default_rng(seed + 100)
    tr, va, te = [], [], []
    for c in range(N_CLASES):
        idx = r.permutation(np.where(y == c)[0])
        n1 = int(frac[0] * len(idx)); n2 = n1 + int(frac[1] * len(idx))
        tr += idx[:n1].tolist(); va += idx[n1:n2].tolist(); te += idx[n2:].tolist()
    tr, va, te = (np.asarray(r.permutation(s)) for s in (tr, va, te))
    return {"X_train": X[tr], "y_train": y[tr], "X_val": X[va], "y_val": y[va],
            "X_test": X[te], "y_test": y[te], "sub_test": sub[te],
            "X_all": X, "y_all": y, "sub_all": sub, "lv_all": lv}


def epoch_orders(n_train, epochs, seed=7):
    """Permutaciones fijas por epoca, compartidas PC <-> MCU (SGD por muestra:
    el orden cambia la trayectoria, asi que compartirlo es obligatorio)."""
    r = np.random.default_rng(seed)
    return [r.permutation(n_train).tolist() for _ in range(epochs)]


# =====================================================================
# 4. Sigmoide parametrica  f(z) = a / (1 + b*exp(-c*z)) + d
# =====================================================================
# Mismas tablas que ACT_H1 / ACT_H2 en firmware/mlp.c
ACT_H1 = [(1.0, 0.5, 1.0, 0.0), (1.0, 1.0, 1.0, 0.0),
          (1.0, 1.5, 1.0, 0.0), (1.0, 2.0, 1.0, 0.0),
          (1.0, 0.5, 1.5, 0.0), (1.0, 1.0, 1.5, 0.0),
          (1.0, 1.5, 1.5, 0.0), (1.0, 2.0, 1.5, 0.0)]
ACT_H2 = [(1.0, 0.5, 1.0, 0.0), (1.0, 1.0, 1.0, 0.0),
          (1.0, 1.5, 1.0, 0.0), (1.0, 0.5, 2.0, 0.0),
          (1.0, 1.0, 2.0, 0.0), (1.0, 1.5, 2.0, 0.0)]
TOPOLOGIA = (N_FEAT, 8, 6, N_CLASES)
Z_CLAMP = 80.0
LOSS_EPS = 1e-12


def act(z, a, b, c, d):
    arg = np.clip(-c * z, -Z_CLAMP, Z_CLAMP)
    return a / (1.0 + b * np.exp(arg)) + d


def act_der_a(y, a, b, c, d):
    """f'(z) = a*c*u*(1-u) con u = (y-d)/a: se calcula desde la activacion,
    sin exp(). Vale para cualquier b > 0."""
    u = (y - d) / a
    return a * c * u * (1.0 - u)


def _tabla_act(n, base):
    return [base[i % len(base)] for i in range(n)]


# =====================================================================
# 5. Red: sigmoides parametricas en capas ocultas + softmax a la salida
# =====================================================================
class MLP:
    """Perceptron multicapa generico. Con topologia=TOPOLOGIA reproduce el
    firmware; otras topologias sirven para el estudio de reduccion.

    feat_idx permite entrenar con un subconjunto de las 9 caracteristicas.
    """

    def __init__(self, topologia=TOPOLOGIA, seed=1, dtype=np.float64,
                 feat_idx=None):
        self.dtype = dtype
        self.sizes = tuple(topologia)
        self.feat_idx = None if feat_idx is None else list(feat_idx)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            W = rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_out, n_in))
            self.W.append(W.astype(dtype))
            self.b.append(np.zeros(n_out, dtype=dtype))
        bases = [ACT_H1, ACT_H2]
        self.P = []
        for k, n in enumerate(self.sizes[1:-1]):
            t = np.asarray(_tabla_act(n, bases[min(k, 1)]), dtype=dtype)
            self.P.append(t.T)               # 4 vectores: a, b, c, d

    # ---------------- utilidades ----------------
    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    def _x(self, x):
        x = np.asarray(x, dtype=self.dtype)
        return x if self.feat_idx is None else x[self.feat_idx]

    def get_weights(self):
        return {"W": [w.tolist() for w in self.W], "b": [v.tolist() for v in self.b]}

    def set_weights(self, d):
        self.W = [np.asarray(w, dtype=self.dtype) for w in d["W"]]
        self.b = [np.asarray(v, dtype=self.dtype) for v in d["b"]]

    def flat_params(self):
        """Orden de volcado del firmware (comando 'W'): W1,b1,W2,b2,W3,b3."""
        out = []
        for W, b in zip(self.W, self.b):
            out += np.ravel(W).tolist() + np.ravel(b).tolist()
        return np.asarray(out, dtype=self.dtype)

    def set_flat_params(self, flat):
        flat = np.asarray(flat, dtype=self.dtype); k = 0
        for i in range(len(self.W)):
            n = self.W[i].size
            self.W[i] = flat[k:k + n].reshape(self.W[i].shape); k += n
            n = self.b[i].size
            self.b[i] = flat[k:k + n].copy(); k += n

    # ---------------- forward ----------------
    def forward(self, x):
        dt = self.dtype
        a = self._x(x)
        acts = [a]
        for l, P in enumerate(self.P):
            z = (self.W[l] @ a + self.b[l]).astype(dt)
            a = act(z, *P).astype(dt)
            acts.append(a)
        z = (self.W[-1] @ a + self.b[-1]).astype(dt)
        e = np.exp(z - z.max()).astype(dt)
        p = (e / e.sum()).astype(dt)
        return p, acts

    # ---------------- backward ----------------
    def backward(self, p, acts, label, lr, update=True):
        """Softmax + entropia cruzada: delta_salida = p - onehot.
        Devuelve la perdida ANTES de actualizar (igual que el firmware)."""
        dt = self.dtype
        loss = float(-np.log(max(float(p[label]), LOSS_EPS)))
        delta = p.copy(); delta[label] -= dt(1.0)
        deltas = [delta]
        for l in range(len(self.W) - 1, 0, -1):
            P = self.P[l - 1]
            delta = ((self.W[l].T @ delta) * act_der_a(acts[l], *P)).astype(dt)
            deltas.insert(0, delta)
        if update:
            lr = dt(lr)
            for l in range(len(self.W)):
                self.W[l] -= (lr * np.outer(deltas[l], acts[l])).astype(dt)
                self.b[l] -= (lr * deltas[l]).astype(dt)
        return loss, deltas

    def train_sample(self, x, label, lr):
        p, acts = self.forward(x)
        loss, _ = self.backward(p, acts, int(label), lr, update=True)
        return p, loss

    def predict_proba(self, X):
        return np.array([self.forward(x)[0] for x in X])

    def evaluate(self, X, Y):
        P = self.predict_proba(X)
        loss = float(np.mean([-np.log(max(p[y], LOSS_EPS)) for p, y in zip(P, Y)]))
        acc = 100.0 * float(np.mean(P.argmax(axis=1) == Y))
        return loss, acc, P

    # ---------------- exportacion ----------------
    def export_c(self, nombre="INIT"):
        def arr(v):
            return ", ".join(f"{x:.9e}f" for x in np.ravel(v))
        s = (f"/* Pesos exportados desde mlp_core.MLP (topologia {self.sizes}).\n"
             " * Generado automaticamente: no editar a mano. */\n"
             f"#ifndef PESOS_{nombre}_H\n#define PESOS_{nombre}_H\n\n")
        for i, (W, b) in enumerate(zip(self.W, self.b), start=1):
            s += f"static const float W{i}_{nombre}[{W.size}] = {{ {arr(W)} }};\n"
            s += f"static const float B{i}_{nombre}[{b.size}] = {{ {arr(b)} }};\n"
        return s + "\n#endif\n"


# =====================================================================
# 6. Regla por umbrales (linea base sin aprendizaje)
# =====================================================================
def clasificador_umbral(X):
    """Umbral fijo en 0.5 de escala sobre la media de cada pote. Es lo que
    haria un if/else: sirve de linea base para justificar la red."""
    bits = (X[:, :N_CH] > 0.0)          # media normalizada > 0  <=> > 50 %
    out = np.zeros(len(X), dtype=np.int64)
    for c, pat in PATRONES.items():
        m = np.all(bits == np.array([ch == "H" for ch in pat]), axis=1)
        out[m] = c
    return out


# =====================================================================
# 7. Metricas
# =====================================================================
def matriz_confusion(y, yp, n=N_CLASES):
    M = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(y, yp):
        M[int(a), int(b)] += 1
    return M


def metricas_clase(M):
    tp = np.diag(M).astype(float)
    prec = tp / np.maximum(M.sum(axis=0), 1)
    rec = tp / np.maximum(M.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
            "accuracy": float(tp.sum() / max(M.sum(), 1)),
            "macro_f1": float(f1.mean())}


# =====================================================================
# 8. Cuantizacion int8 simetrica por capa (post-entrenamiento)
# =====================================================================
def cuantizar_int8(net):
    q = MLP(net.sizes, dtype=np.float64, feat_idx=net.feat_idx)
    q.P = [P.astype(np.float64) for P in net.P]
    Ws, bs = [], []
    for W, b in zip(net.W, net.b):
        s = float(np.max(np.abs(W))) / 127.0
        Ws.append(np.clip(np.rint(W / s), -127, 127) * s)
        sb = max(float(np.max(np.abs(b))), 1e-12) / 127.0
        bs.append(np.clip(np.rint(b / sb), -127, 127) * sb)
    q.W, q.b = Ws, bs
    return q


# =====================================================================
# 9. Verificacion numerica del gradiente
# =====================================================================
def gradient_check(seed=1, eps=1e-6, n_samples=10):
    d = split_dataset()
    net = MLP(seed=seed, dtype=np.float64)
    flat0 = net.flat_params()
    max_rel = 0.0
    for k in range(n_samples):
        x, y = d["X_train"][k], int(d["y_train"][k])
        net.set_flat_params(flat0)
        p, acts = net.forward(x)
        _, deltas = net.backward(p, acts, y, 0.0, update=False)
        an = []
        for l in range(len(net.W)):
            an += np.outer(deltas[l], acts[l]).ravel().tolist() + deltas[l].tolist()
        an = np.asarray(an)
        for i in range(len(flat0)):
            f = flat0.copy(); f[i] += eps; net.set_flat_params(f)
            lp = -np.log(net.forward(x)[0][y])
            f[i] -= 2 * eps; net.set_flat_params(f)
            lm = -np.log(net.forward(x)[0][y])
            g = (lp - lm) / (2 * eps)
            max_rel = max(max_rel, abs(an[i] - g) / max(1e-9, abs(an[i]) + abs(g)))
    net.set_flat_params(flat0)
    return max_rel


# =====================================================================
# 10. Utilidades de E/S y del protocolo serie (floats en hexadecimal)
# =====================================================================
def f32_to_hex(v):
    """float -> 8 digitos hex del patron IEEE-754 de 32 bits. Se transmite asi
    para que el MCU reciba exactamente el mismo float32 que usa la PC."""
    return struct.pack(">f", float(v)).hex().upper()


def hex_to_f32(h):
    return struct.unpack(">f", bytes.fromhex(h))[0]


def save_json(path, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=conv)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    z = np.linspace(-10, 10, 1001)
    print("Gradient check:", f"{gradient_check():.3e}")
    d = split_dataset()
    print({k: v.shape for k, v in d.items() if hasattr(v, "shape")})
    print("Parametros:", MLP().n_params())
