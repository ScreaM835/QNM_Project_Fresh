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


if __name__ == "__main__":
    plot_potential()
    plot_tortoise_map()
    plot_rw_vs_zerilli_l2()
    print(f"wrote figures to {OUT}")
