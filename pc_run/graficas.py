"""
graficas.py
===========
Figuras del informe (PDF vectorial a ancho de columna IEEE). Usa los JSON que
existan; las figuras que dependen de hardware se generan solo si ya estan
los resultados correspondientes.

  resultados_pc.json, resultados_pc_f32.json, resultados_reduccion.json  (run_pc.py)
  resultados_host.json   gemelo C en PC        (run_mcu.py --host, renombrado)
  resultados_mcu.json    entrenamiento en MCU  (run_mcu.py)
  resultados_real.json   dataset real          (evaluar.py)
  resultados_sim.json    lazo simulado         (simulacion_lazo.py)
  tlm_escalon_*.csv      escalon real          (telemetria.py escalon)

Uso:  python graficas.py  [--salida ../informe/figuras]
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mlp_core as core
from mlp_core import load_json

COL = 3.4
OUT = "figuras"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "lines.linewidth": 1.1, "legend.framealpha": 0.9,
})
COLORES = ["0.45", "C0", "C3", "C2"]     # default, CW, CCW, paro


def cargar(p):
    return load_json(p) if os.path.exists(p) else None


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=200)
    plt.close(fig); print(f"  {path}.pdf")


# ---------------------------------------------------------------------
def fig_dataset(pc):
    s = pc["split"]
    X = np.array(s["X_train"] + s["X_val"] + s["X_test"])
    y = np.array(s["y_train"] + s["y_val"] + s["y_test"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.75))
    for c in range(4):
        m = y == c
        a1.scatter((X[m, 0] + 1) / 2, (X[m, 2] + 1) / 2, s=2, c=COLORES[c],
                   label=core.CLASES[c], alpha=0.6, lw=0)
        a2.scatter((X[m, 1] + 1) / 2, X[m, 3:6].max(1), s=2, c=COLORES[c], alpha=0.6, lw=0)
    a1.set_xlabel("media pot 1"); a1.set_ylabel("media pot 3")
    a2.set_xlabel("media pot 2"); a2.set_ylabel(r"máx. $\sigma/\sigma_{ref}$")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=4,
               markerscale=4, bbox_to_anchor=(0.5, 1.08), handletextpad=0.1)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_dataset")


def fig_convergencia(pc, pc32, host, mcu):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * COL, 1.9),
                                 gridspec_kw={"width_ratios": [1, 1.3]})
    ep = np.arange(1, len(pc["loss_train"]) + 1)
    a1.semilogy(ep, pc["loss_train"], "-", c="C0", label="PC float64 · entren.")
    a1.semilogy(ep, pc["loss_val"], "--", c="C0", label="PC float64 · valid.")
    if host:
        a1.semilogy(ep, host["loss_train"], "o", c="C1", ms=2.5, mfc="none",
                    label="C (gemelo en PC) · entren.")
    if mcu:
        em = np.arange(1, len(mcu["loss_train"]) + 1)
        a1.semilogy(em, mcu["loss_train"], "s", c="C3", ms=2.5, mfc="none",
                    label="S32K312 · entren.")
    a1.set_xlabel("Época"); a1.set_ylabel("Entropía cruzada media")
    a1.legend(loc="upper right")

    p64 = np.array(pc["prob_log"]); p32 = np.array(pc32["prob_log"])
    k = np.arange(len(p64))
    a2.semilogy(k, np.maximum(np.abs(p64 - p32).max(1), 1e-12), ".", ms=0.6, c="C0",
                alpha=0.4, label=r"|PC$_{64}$ − PC$_{32}$|")
    if host:
        ph = np.array(host["prob_log"])
        a2.semilogy(k, np.maximum(np.abs(p32 - ph).max(1), 1e-12), ".", ms=0.6, c="C1",
                    alpha=0.4, label=r"|PC$_{32}$ − C|")
    if mcu:
        pm = np.array(mcu["prob_log"]); n = min(len(pm), len(p32))
        a2.semilogy(k[:n], np.maximum(np.abs(p32[:n] - pm[:n]).max(1), 1e-12), ".",
                    ms=0.6, c="C3", alpha=0.4, label=r"|PC$_{32}$ − S32K312|")
    a2.axhline(np.finfo(np.float32).eps, c="k", ls=":", lw=0.8)
    a2.text(len(k) * 0.99, np.finfo(np.float32).eps * 1.6, r"$\epsilon_{32}$",
            ha="right", fontsize=6.5)
    a2.set_xlabel("Paso de SGD (acumulado)")
    a2.set_ylabel(r"máx$_k$ |Δ$p_k$|")
    a2.legend(loc="lower right", markerscale=12)
    fig.tight_layout()
    save(fig, "fig_convergencia")


def _cm(ax, M, titulo):
    M = np.array(M); Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=6.5,
                    color="w" if Mn[i, j] > 0.6 else "k")
    et = ["Def", "CW", "CCW", "Paro"]
    ax.set_xticks(range(4)); ax.set_xticklabels(et, fontsize=6)
    ax.set_yticks(range(4)); ax.set_yticklabels(et, fontsize=6)
    ax.set_title(titulo, fontsize=7); ax.grid(False)


def fig_confusion(pc, red, host, mcu, real):
    paneles = [(red["umbral"]["cm_test"], "Regla de umbral"), (pc["cm_test"], "Red · PC float64")]
    if mcu:
        paneles.append((mcu["cm_test"], "Red · S32K312"))
    elif host:
        paneles.append((host["cm_test"], "Red · C (gemelo)"))
    if real and "cm_mcu" in real:
        paneles.append((real["cm_mcu"], "S32K312 · datos reales"))
    fig, axs = plt.subplots(1, len(paneles), figsize=(COL * len(paneles) / 2.0, 1.75))
    for ax, (M, t) in zip(np.atleast_1d(axs), paneles):
        _cm(ax, M, t)
    np.atleast_1d(axs)[0].set_ylabel("Real")
    for ax in np.atleast_1d(axs):
        ax.set_xlabel("Predicha", fontsize=7)
    fig.tight_layout(w_pad=0.3)
    save(fig, "fig_confusion")


def fig_frontera(pc):
    fr = pc["frontera"]
    x = np.array(fr["niveles"]); P = np.array(fr["probs"])
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    for c in range(4):
        ax.plot(x, P[:, c], c=COLORES[c], label=core.CLASES[c])
    ax.axhline(0.70, c="k", ls=":", lw=0.8)
    ax.axvspan(0.30, 0.40, color="0.9", zorder=0); ax.axvspan(0.60, 0.70, color="0.9", zorder=0)
    ax.set_xlabel("Nivel del pot 3 (pot 1 = pot 2 = 1.0)")
    ax.set_ylabel("Probabilidad")
    ax.legend(loc="center right", ncol=2)
    save(fig, "fig_frontera")


def fig_lazo(sim):
    s = sim["escenario"]; t = np.array(s["t"])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(2 * COL, 2.9), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 0.7, 1.3]})
    for i, c in enumerate(("C1", "C4", "C5")):
        a1.plot(t, s[f"lv{i + 1}"], c=c, label=f"pot {i + 1}")
    a1.set_ylabel("Nivel"); a1.legend(loc="center left", ncol=3, bbox_to_anchor=(0.30, 0.5))
    a2.step(t, s["cls_raw"], where="post", c="0.6", lw=0.8, label="salida de la red")
    a2.step(t, s["cls"], where="post", c="k", label="clase activa")
    a2.set_yticks(range(4)); a2.set_yticklabels(core.CLASES, fontsize=6)
    a2.legend(loc="upper right", ncol=2)
    a3.plot(t, s["ref"], "--", c="k", lw=0.9, label="referencia")
    a3.plot(t, s["rpm"], c="C0", label="velocidad medida")
    a3.set_ylabel("rpm"); a3.set_xlabel("Tiempo [s]")
    a3.legend(loc="lower left", ncol=2)
    fig.tight_layout(h_pad=0.2)
    save(fig, "fig_lazo")


def fig_control(sim, tlm_escalon=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * COL, 1.8))
    e = sim["escalon"]; t = np.array(e["t"])
    a1.plot(t - 0.2, e["ref"], "--", c="k", lw=0.9, label="referencia")
    a1.plot(t - 0.2, e["rpm"], c="C0", label="simulado")
    if tlm_escalon is not None:
        import csv
        with open(tlm_escalon) as f:
            rows = list(csv.DictReader(f))
        tr = np.array([float(r["t_ms"]) for r in rows]) / 1e3
        a1.plot(tr - tr[0], [float(r["rpm"]) for r in rows], c="C3", label="S32K312")
    a1.set_xlim(-0.1, 1.0); a1.set_xlabel("Tiempo desde el escalón [s]"); a1.set_ylabel("rpm")
    a1.legend(loc="lower right")

    b = sim["barrido_tcl"]
    tcl = [x["t_cl"] for x in b]
    a2.plot(tcl, [x["t_establecimiento_s"] for x in b], "o-", c="C0", label=r"$t_s$ (2 %)")
    a2.plot(tcl, [x["t_subida_s"] for x in b], "s-", c="C2", label=r"$t_r$ (10–90 %)")
    a2.set_xlabel(r"$T_{cl}$ de diseño [s]"); a2.set_ylabel("Tiempo [s]")
    ax2 = a2.twinx()
    ax2.plot(tcl, [x["u_pico"] for x in b], "^--", c="C3", label=r"$|u|_{máx}$")
    ax2.set_ylabel(r"$|u|_{máx}$", color="C3"); ax2.set_ylim(0, 1.1); ax2.grid(False)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="upper center")
    fig.tight_layout()
    save(fig, "fig_control")


def fig_reduccion(red):
    t = red["topologias"]
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    x = np.arange(len(t))
    ax.bar(x, [r["bytes_f32"] for r in t], color="C0", width=0.6)
    ax.set_ylabel("Parámetros [B, float32]")
    ax.set_xticks(x); ax.set_xticklabels([r["nombre"].split(" ")[0] for r in t], fontsize=6.5)
    ax2 = ax.twinx()
    ax2.plot(x, [r["acc_default"] * 100 for r in t], "s", c="C3", label="exactitud Default")
    ax2.plot(x, [r["acc_test"] for r in t], "o", c="k", label="exactitud total")
    ax2.set_ylim(80, 101); ax2.set_ylabel("Exactitud prueba [%]"); ax2.grid(False)
    ax2.legend(loc="lower center", ncol=2)
    ax.set_xlabel("Topología (entradas–ocultas–salidas)")
    save(fig, "fig_reduccion")


# ---------------------------------------------------------------------
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="figuras")
    OUT = ap.parse_args().salida
    os.makedirs(OUT, exist_ok=True)

    pc = load_json("resultados_pc.json"); pc32 = load_json("resultados_pc_f32.json")
    red = load_json("resultados_reduccion.json")
    host, mcu = cargar("resultados_host.json"), cargar("resultados_mcu.json")
    real, sim = cargar("resultados_real.json"), cargar("resultados_sim.json")
    tlm = sorted(glob.glob("tlm_escalon_*.csv"))

    print(f"Generando figuras en {OUT}/  (MCU: {'sí' if mcu else 'no'}, "
          f"real: {'sí' if real else 'no'})")
    fig_dataset(pc)
    fig_convergencia(pc, pc32, host, mcu)
    fig_confusion(pc, red, host, mcu, real)
    fig_frontera(pc)
    fig_reduccion(red)
    if sim:
        fig_lazo(sim)
        fig_control(sim, tlm[-1] if tlm else None)


if __name__ == "__main__":
    main()
