"""Regenerate param_convergence.png for variant D (combo) from saved
param_history.json + FD .npz, with two bug fixes relative to the original
plotting code in run_pinn_inverse_qnm.py:

  (1) The bottom-right ringdown-template panel x-axis was incorrectly
      relabelled "Training step (logged)" by a sweep-all-bottom-row loop.
      Correct label is "t / M".
  (2) The analytic template was plotted only over [t_ring_min, tmax] M
      (i.e. starting at t = 18 M for variant D). It is now plotted across
      the full simulation domain [0, tmax] M, with the training window
      shaded so the reader can still see where the loss was applied.

This script does NOT retrain. It only re-renders the figure from data
already on disk.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = os.path.join(
    os.path.dirname(__file__),
    "..", "outputs", "pinn", "zerilli_l2_inverse_qnm_combo",
)
HIST = os.path.join(OUTDIR, "param_history.json")
FD = os.path.join(OUTDIR, "zerilli_l2_inverse_qnm_combo_fd.npz")
OUTPNG = os.path.join(OUTDIR, "param_convergence.png")

# Variant-D config constants (from configs/zerilli_l2_inverse_qnm_combo.yaml)
T_RING_MIN = 18.0
T_MAX = 50.0
XQ = 10.0
NOISE = 0.01

with open(HIST) as f:
    H = json.load(f)

M_hist = np.asarray(H["M_history"])
omega_hist = np.asarray(H["omega_history"])
tau_hist = np.asarray(H["tau_history"])
A_hist = np.asarray(H["A_ring_history"])
phi_hist = np.asarray(H["phi0_history"])

M_true = H["M_true"]; M_init = H["M_init"]
omega_true = H["omega_true"]; omega_init = H["omega_init"]
tau_true = H["tau_true"]; tau_init = H["tau_init"]
A_init = 0.5
phi_init = 0.0

M_final = float(M_hist[-1])
omega_final = float(omega_hist[-1])
tau_final = float(tau_hist[-1])
A_final = float(A_hist[-1])
phi_final = float(phi_hist[-1])

M_err = abs(M_final - M_true) / M_true * 100.0
omega_err = abs(omega_final - omega_true) / omega_true * 100.0
tau_err = abs(tau_final - tau_true) / tau_true * 100.0

fd = np.load(FD)
t_fd = fd["t"]
x_fd = fd["x"]
phi_fd = fd["phi"]
ix = int(np.argmin(np.abs(x_fd - XQ)))

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# M
ax = axes[0, 0]
ax.plot(M_hist, "b-", linewidth=0.8, label="Learned M")
ax.axhline(M_true, color="r", linestyle="--", linewidth=1.5, label=f"True M = {M_true}")
ax.axhline(M_init, color="gray", linestyle=":", linewidth=1.0, label=f"Init M = {M_init}")
ax.set_ylabel("M"); ax.set_title(f"Mass convergence (err = {M_err:.3f}%)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# omega
ax = axes[0, 1]
ax.plot(omega_hist, "b-", linewidth=0.8, label="Learned ω")
ax.axhline(omega_true, color="r", linestyle="--", linewidth=1.5, label=f"True ω = {omega_true}")
ax.axhline(omega_init, color="gray", linestyle=":", linewidth=1.0, label=f"Init ω = {omega_init}")
ax.set_ylabel("ω"); ax.set_title(f"Frequency convergence (err = {omega_err:.3f}%)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# tau
ax = axes[1, 0]
ax.plot(tau_hist, "b-", linewidth=0.8, label="Learned τ")
ax.axhline(tau_true, color="r", linestyle="--", linewidth=1.5, label=f"True τ = {tau_true}")
ax.axhline(tau_init, color="gray", linestyle=":", linewidth=1.0, label=f"Init τ = {tau_init}")
ax.set_ylabel("τ"); ax.set_title(f"Damping time convergence (err = {tau_err:.3f}%)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# A
ax = axes[1, 1]
ax.plot(A_hist, "b-", linewidth=0.8, label="Learned A")
ax.axhline(A_init, color="gray", linestyle=":", linewidth=1.0, label=f"Init A = {A_init}")
ax.set_ylabel("A"); ax.set_title("Ringdown amplitude convergence")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# phi
ax = axes[2, 0]
ax.plot(phi_hist, "b-", linewidth=0.8, label="Learned φ₀")
ax.axhline(phi_init, color="gray", linestyle=":", linewidth=1.0, label=f"Init φ₀ = {phi_init}")
ax.set_ylabel("φ₀"); ax.set_title("Phase convergence")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Template overlay (FIXES BOTH BUGS)
ax = axes[2, 1]
t_plot = np.linspace(0.0, T_MAX, 1000)
template_plot = (A_final * np.exp(-t_plot / tau_final)
                 * np.cos(omega_final * t_plot + phi_final))
ax.plot(t_fd, phi_fd[:, ix], "k-", linewidth=0.8, label="FD (truth)")
ax.plot(t_plot, template_plot, "r--", linewidth=1.2,
        label="Template: A·e^(-t/τ)·cos(ωt+φ₀)")
ax.axvspan(T_RING_MIN, T_MAX, color="orange", alpha=0.10,
           label=f"training window [{T_RING_MIN:.0f}, {T_MAX:.0f}] M")
ax.set_xlabel("t / M")
ax.set_ylabel("φ")
ax.set_title(f"Ringdown template fit at x*={XQ}")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)

# X-labels only on the parameter-history panels.
for a in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]):
    a.set_xlabel("Training step (logged)")

fig.suptitle(f"Inverse+QNM Parameter Convergence (noise={NOISE:.1%})",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUTPNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUTPNG}")
