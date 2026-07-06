from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import ensure_dir


# --- Standardised axis scales for cross-model comparison -------------------
# Reference: the Regge--Wheeler reproduction (report Figs. 13-14). Fixing these
# so every Schwarzschild model shares one scale lets the reader compare the
# absolute-difference, snapshot and pointwise-error plots directly, instead of
# each panel auto-scaling to its own data (which made the scales drift between
# models). Callers may override per plot, but the defaults are the house scale.
ABS_DIFF_YMAX = 0.02           # abs-difference snapshot y-axis top (RW max 1.84e-2)
SNAPSHOT_YLIM = (-0.85, 0.6)   # field-snapshot y-range
HEATMAP_VMIN = 1.0e-6          # pointwise-error log colour floor
HEATMAP_VMAX = 2.1e-2          # pointwise-error log colour ceiling (RW max 2.09e-2)


def plot_snapshots(
    x: np.ndarray,
    t: np.ndarray,
    phi_fd: np.ndarray,
    phi_pinn: np.ndarray,
    times: List[float],
    outpath: str,
    title: str,
    model_label: str = "PINN",
    ylim: Optional[Tuple[float, float]] = SNAPSHOT_YLIM,
) -> None:
    ensure_dir(os.path.dirname(outpath))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, tt in zip(axes, times):
        idx = int(np.argmin(np.abs(t - tt)))
        ax.plot(x, phi_fd[idx], label="FD")
        ax.plot(x, phi_pinn[idx], label=model_label)
        ax.set_title(f"t/M = {t[idx]:.0f}")
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_abs_diff_snapshots(
    x: np.ndarray,
    t: np.ndarray,
    phi_fd: np.ndarray,
    phi_pinn: np.ndarray,
    times: List[float],
    outpath: str,
    title: str,
    ymax: Optional[float] = ABS_DIFF_YMAX,
) -> None:
    ensure_dir(os.path.dirname(outpath))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, tt in zip(axes, times):
        idx = int(np.argmin(np.abs(t - tt)))
        ax.plot(x, np.abs(phi_fd[idx] - phi_pinn[idx]))
        ax.set_title(f"t/M = {t[idx]:.0f}")
        ax.grid(True, alpha=0.3)

    if ymax is not None:
        axes[0].set_ylim(0.0, ymax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_loss(history: Dict[str, List[float]], outpath: str, title: str) -> None:
    ensure_dir(os.path.dirname(outpath))
    # Guard against a duplicated history: a resumed two-phase (Adam + L-BFGS)
    # run can log the whole cycle twice, so the second half exactly repeats the
    # first. Keep only the first, valid cycle.
    Lt = history.get("L_total", [])
    n = len(Lt)
    if n >= 4 and n % 2 == 0 and list(Lt[: n // 2]) == list(Lt[n // 2:]):
        history = {
            k: (v[: n // 2] if isinstance(v, list) and len(v) == n else v)
            for k, v in history.items()
        }
    fig = plt.figure(figsize=(9, 5))

    # Use actual iteration numbers if available, otherwise fallback to indices
    if "steps" in history and len(history["steps"]) == len(history["L_total"]):
        it = np.array(history["steps"])
    else:
        it = np.arange(len(history["L_total"]))

    ax = plt.gca()
    ax.semilogy(it, history["L_total"], label="L_total", linewidth=1.5)
    for k in ["Lr", "Lrx", "Lrt", "Lic", "Liv", "Lbl", "Lbr"]:
        ax.semilogy(it, history[k], label=k, alpha=0.7)

    # Mark Adam → L-BFGS transition
    if "phase" in history:
        phases = history["phase"]
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                ax.axvline(it[i], color="grey", linestyle="--", alpha=0.7, linewidth=1.2)
                ax.text(
                    it[i], ax.get_ylim()[1], " L-BFGS →",
                    fontsize=8, va="top", ha="left", color="grey",
                )
                break

    # If causal training is active, show the minimum causal weight on a secondary axis
    if "w_min" in history and any(v < 1.0 for v in history["w_min"]):
        ax2 = ax.twinx()
        ax2.plot(it, history["w_min"], color="black", linestyle="--", alpha=0.5, label="w_min (causal)")
        ax2.set_ylabel("Min causal weight", fontsize=8)
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc="center right", fontsize=8)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (unweighted MSE)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_loss_fno(history: List[Dict], outpath: str, title: str) -> None:
    """FNO/operator training history in the same visual style as plot_loss.

    ``history`` is a list of per-epoch dicts as written by train_fno.py
    (keys: 'epoch', 'loss', and any subset of the weighted components
    'field', 'h1', 'ring', 'obs', 'tw', 'pde').
    """
    ensure_dir(os.path.dirname(outpath))
    if not history:
        return
    epochs = np.array([rec["epoch"] for rec in history])
    components = ["field", "h1", "ring", "obs", "tw", "pde"]

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    if "loss" in history[0]:
        L_tot = np.array([rec["loss"] for rec in history])
        ax.semilogy(epochs, L_tot, label="L_total", linewidth=1.5)
    for k in components:
        if k in history[0]:
            vals = np.array([rec.get(k, np.nan) for rec in history], dtype=float)
            ax.semilogy(epochs, vals, label=f"L_{k}", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (weighted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_ringdown(t: np.ndarray, y: np.ndarray, outpath: str, title: str) -> None:
    ensure_dir(os.path.dirname(outpath))
    fig = plt.figure(figsize=(8, 4))
    plt.plot(t, y)
    plt.xlabel("t/M")
    plt.ylabel("Phi(xq, t)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_ringdown_overlay(
    t: np.ndarray,
    y_fd: np.ndarray,
    y_pinn: np.ndarray,
    outpath: str,
    title: str = "Ringdown comparison",
    xq: float = 10.0,
    model_label: str = "PINN",
) -> None:
    """
    Two-panel ringdown comparison (similar to paper Fig. 5):
      Top:    linear scale, FD vs model overlaid
      Bottom: semi-log |phi| to compare exponential decay / damping rates
    """
    ensure_dir(os.path.dirname(outpath))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Top panel: linear
    ax1.plot(t, y_fd, label="FD", linewidth=1.2)
    ax1.plot(t, y_pinn, label=model_label, linewidth=1.2, linestyle="--")
    ax1.set_ylabel(r"$\Phi(x_q, t)$")
    ax1.set_title(f"{title}  ($x_q = {xq}$)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom panel: semi-log |phi|
    ax2.semilogy(t, np.abs(y_fd) + 1e-30, label="FD", linewidth=1.2)
    ax2.semilogy(t, np.abs(y_pinn) + 1e-30, label=model_label, linewidth=1.2, linestyle="--")
    ax2.set_xlabel("t / M")
    ax2.set_ylabel(r"$|\Phi(x_q, t)|$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_error_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    phi_fd: np.ndarray,
    phi_pinn: np.ndarray,
    outpath: str,
    title: str = "Pointwise error",
    signed: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    model_label: str = "PINN",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    2D colormap of |phi_FD - phi_model| (or signed difference) over the full
    space-time domain.  Similar to paper Fig. 7.
    """
    ensure_dir(os.path.dirname(outpath))

    diff = phi_fd - phi_pinn
    if not signed:
        diff = np.abs(diff)

    fig, ax = plt.subplots(figsize=(10, 5))
    if signed:
        vmax = np.max(np.abs(diff))
        im = ax.pcolormesh(
            x, t, diff, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(rf"$\Phi_{{\mathrm{{FD}}}} - \Phi_{{\mathrm{{{model_label}}}}}$")
    else:
        from matplotlib.colors import LogNorm
        vmin = HEATMAP_VMIN if vmin is None else vmin
        vmax = HEATMAP_VMAX if vmax is None else vmax
        im = ax.pcolormesh(
            x, t, np.clip(diff, vmin, None), shading="auto", cmap="magma_r",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(rf"$|\Phi_{{\mathrm{{FD}}}} - \Phi_{{\mathrm{{{model_label}}}}}|$ (log scale)")

    ax.set_xlabel(r"$x_* / M$")
    ax.set_ylabel("t / M")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_snapshots_zoomed(
    x: np.ndarray,
    t: np.ndarray,
    phi_fd: np.ndarray,
    phi_pinn: np.ndarray,
    times: List[float],
    outpath: str,
    title: str,
    xlim: Tuple[float, float] = (-20.0, 60.0),
) -> None:
    """
    Same as plot_snapshots but with a zoomed x-axis (default [-20, 60])
    to match the paper's presentation.
    """
    ensure_dir(os.path.dirname(outpath))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, tt in zip(axes, times):
        idx = int(np.argmin(np.abs(t - tt)))
        ax.plot(x, phi_fd[idx], label="FD", linewidth=1.2)
        ax.plot(x, phi_pinn[idx], label="PINN", linewidth=1.2, linestyle="--")
        ax.set_title(f"t/M = {t[idx]:.0f}")
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    axes[2].set_xlabel(r"$x_* / M$")
    axes[3].set_xlabel(r"$x_* / M$")
    axes[0].set_ylabel(r"$\Phi$")
    axes[2].set_ylabel(r"$\Phi$")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
