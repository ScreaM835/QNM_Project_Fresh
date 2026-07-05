"""Generate background-section figures for the paper.

Produces:
  outputs/background/zerilli_potential.png  - V_l^Z(x) for l=2,3,4
  outputs/background/tortoise_map.png       - r(x) vs x

All quantities in geometric units M = 1.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent.parent / "outputs" / "background"
OUT.mkdir(parents=True, exist_ok=True)

M = 1.0


def f_metric(r: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * M / r


def tortoise(r: np.ndarray) -> np.ndarray:
    return r + 2.0 * M * np.log(r / (2.0 * M) - 1.0)


def r_of_x(x: float, r_lo: float = 2.0 + 1e-10, r_hi: float = 1e6) -> float:
    return brentq(lambda r: tortoise(np.array([r]))[0] - x, r_lo, r_hi)


def zerilli_V(r: np.ndarray, ell: int) -> np.ndarray:
    n = 0.5 * (ell - 1) * (ell + 2)
    num = n * n * (n + 1) * r**3 + 3.0 * n * n * M * r**2 \
        + 9.0 * n * M**2 * r + 9.0 * M**3
    den = (n * r + 3.0 * M) ** 2
    return 2.0 * f_metric(r) / r**3 * num / den


def plot_potential() -> None:
    x_grid = np.linspace(-30.0, 60.0, 1200)
    r_grid = np.array([r_of_x(xi) for xi in x_grid])

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    styles = [("-", 2.0), ("--", 1.5), (":", 1.5)]
    names = {2: "quadrupole", 3: "octupole", 4: "hexadecapole"}
    for ell, (ls, lw) in zip([2, 3, 4], styles):
        V = zerilli_V(r_grid, ell)
        ax.plot(x_grid, V, ls, lw=lw,
                label=rf"{names[ell]} ($\ell={ell}$)")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0.0, color="k", lw=0.5, alpha=0.2)
    ax.set_xlabel(r"tortoise coordinate $x/M$")
    ax.set_ylabel(r"$V^{\mathrm{Z}}_\ell(r(x))\,M^2$")
    ax.set_title(r"Zerilli potential, Schwarzschild, $M=1$")
    ax.set_xlim(-30, 60)
    ax.set_ylim(-0.02, 0.45)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "zerilli_potential.png", dpi=200)
    fig.savefig(OUT / "zerilli_potential.pdf")
    plt.close(fig)


def plot_tortoise_map() -> None:
    r_grid = np.linspace(2.0 + 1e-3, 60.0, 1500)
    x_grid = tortoise(r_grid)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(x_grid, r_grid, "-", lw=2.0, color="C0")
    ax.axhline(2.0, color="C3", lw=1.2, ls="--", label=r"horizon $r=2M$")
    ax.axvline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.plot(x_grid, x_grid, "k:", lw=1.0, alpha=0.5,
            label=r"$r=x$ (asymptote at large $x$)")
    ax.set_xlabel(r"tortoise coordinate $x/M$")
    ax.set_ylabel(r"areal radius $r/M$")
    ax.set_title(r"Tortoise map $r(x)$, Schwarzschild")
    ax.set_xlim(-30, 60)
    ax.set_ylim(1.0, 60.0)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "tortoise_map.png", dpi=200)
    fig.savefig(OUT / "tortoise_map.pdf")
    plt.close(fig)


def rw_V(r: np.ndarray, ell: int) -> np.ndarray:
    return f_metric(r) * (ell * (ell + 1) / r**2 - 6.0 * M / r**3)


def plot_rw_vs_zerilli_l2() -> None:
    """Side-by-side Regge-Wheeler vs Zerilli potential at l=2."""
    x_grid = np.linspace(-30.0, 60.0, 1200)
    r_grid = np.array([r_of_x(xi) for xi in x_grid])

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(x_grid, rw_V(r_grid, 2), "-", lw=2.0, color="C3",
            label=r"Regge--Wheeler $V^{\mathrm{RW}}_{\ell=2}$ (axial)")
    ax.plot(x_grid, zerilli_V(r_grid, 2), "--", lw=2.0, color="C0",
            label=r"Zerilli $V^{\mathrm{Z}}_{\ell=2}$ (polar)")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0.0, color="k", lw=0.5, alpha=0.2)
    ax.set_xlabel(r"tortoise coordinate $x/M$")
    ax.set_ylabel(r"$V_\ell(r(x))\,M^2$")
    ax.set_title(r"Two parity sectors share the same QNM spectrum, $\ell=2,\ M=1$")
    ax.set_xlim(-30, 60)
    ax.set_ylim(-0.02, 0.22)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "rw_vs_zerilli_l2.png", dpi=200)
    fig.savefig(OUT / "rw_vs_zerilli_l2.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Kerr background figures
# ----------------------------------------------------------------------
def plot_kerr_horizons() -> None:
    """Kerr horizon radii r_pm/M and the compactification sigma = r_+/r."""
    a = np.linspace(0.0, 1.0, 400)
    disc = np.sqrt(np.clip(1.0 - a**2, 0.0, None))
    r_plus = M + M * disc
    r_minus = M - M * disc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3))

    # (a) horizon structure
    ax1.plot(a, r_plus, "-", lw=2.0, color="C0", label=r"outer horizon $r_+$")
    ax1.plot(a, r_minus, "--", lw=2.0, color="C3", label=r"inner horizon $r_-$")
    ax1.fill_between(a, r_minus, r_plus, color="C0", alpha=0.10)
    ax1.axvline(0.95, color="k", lw=0.8, ls=":", alpha=0.6)
    ax1.text(0.95, 0.10, r"$a/M=0.95$", rotation=90, va="bottom", ha="right",
             fontsize=8, color="k", alpha=0.7)
    ax1.set_xlabel(r"spin $a/M$")
    ax1.set_ylabel(r"horizon radius $r/M$")
    ax1.set_title(r"(a) Kerr horizons $r_\pm = M \pm \sqrt{M^2-a^2}$")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 2.05)
    ax1.legend(frameon=False, loc="center left")
    ax1.grid(True, alpha=0.25)

    # (b) compactification sigma = r_+/r for three spins
    for a0, color, ls in [(0.0, "C0", "-"), (0.5, "C1", "--"), (0.9, "C3", ":")]:
        rp = M + M * np.sqrt(1.0 - a0**2)
        r = np.linspace(rp, 12.0 * M, 500)
        sig = rp / r
        ax2.plot(r, sig, ls, lw=1.9, color=color, label=rf"$a/M={a0:.1f}$")
    ax2.axhline(1.0, color="C3", lw=0.8, ls="--", alpha=0.5)
    ax2.text(11.5, 1.02, r"horizon $\sigma=1$", ha="right", fontsize=8, color="C3")
    ax2.text(11.5, 0.03, r"$\mathcal{I}^+\ (\sigma=0)$", ha="right", fontsize=8, color="k")
    ax2.set_xlabel(r"areal radius $r/M$")
    ax2.set_ylabel(r"compactified coordinate $\sigma = r_+/r$")
    ax2.set_title(r"(b) Radial compactification")
    ax2.set_xlim(1.0, 12.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "kerr_horizons.png", dpi=200)
    fig.savefig(OUT / "kerr_horizons.pdf")
    plt.close(fig)


def plot_kerr_spectrum() -> None:
    """Gravitational (s=-2) QNM spectrum M*omega_R and tau/M vs spin a/M
    for the fundamental n=0 modes (2,2), (3,2), (4,2), via the qnm package
    (Leaver / Cook-Zalutskiy continued fraction)."""
    import qnm
    try:
        qnm.download_data()
    except Exception:
        pass

    spins = np.linspace(0.0, 0.99, 120)
    modes = [(2, 2, 0, "C0", "-"), (3, 2, 0, "C1", "--"), (4, 2, 0, "C3", ":")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for (ell, m, n, color, ls) in modes:
        seq = qnm.modes_cache(s=-2, l=ell, m=m, n=n)
        wr = np.empty_like(spins)
        tau = np.empty_like(spins)
        for i, a in enumerate(spins):
            omega, _, _ = seq(a=a)
            wr[i] = omega.real
            tau[i] = -1.0 / omega.imag
        lab = rf"$(\ell,m,n)=({ell},{m},{n})$"
        ax1.plot(spins, wr, ls, lw=1.9, color=color, label=lab)
        ax2.plot(spins, tau, ls, lw=1.9, color=color, label=lab)

    for ax in (ax1, ax2):
        ax.axvspan(0.0, 0.95, color="green", alpha=0.06)
        ax.axvline(0.95, color="k", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel(r"spin $a/M$")
        ax.set_xlim(0.0, 0.99)
        ax.grid(True, alpha=0.25)
    ax1.set_ylabel(r"oscillation frequency $M\omega_R$")
    ax1.set_title(r"(a) Real frequency")
    ax1.legend(frameon=False, loc="upper left")
    ax2.set_ylabel(r"damping time $\tau/M$")
    ax2.set_title(r"(b) Damping time")
    ax2.text(0.475, 0.02, r"study range $a/M\in[0,0.95]$",
             transform=ax2.get_xaxis_transform(), ha="center",
             fontsize=8, color="green")

    fig.tight_layout()
    fig.savefig(OUT / "kerr_spectrum.png", dpi=200)
    fig.savefig(OUT / "kerr_spectrum.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_potential()
    plot_tortoise_map()
    plot_rw_vs_zerilli_l2()
    plot_kerr_horizons()
    plot_kerr_spectrum()
    print(f"wrote figures to {OUT}")
