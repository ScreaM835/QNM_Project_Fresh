"""Parse a hybrid-training .out file into history.json and a loss plot.

Usage:
    python scripts/plot_hybrid_history.py qnm_hybrid_29842387.out outputs/hybrid/fno_sw_k2
    python scripts/plot_hybrid_history.py qnm_hybrid_29842388.out outputs/hybrid/fno_sw_k4
"""
from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# [ADAM] ep   10/100  train MSE 2.994e-06 (L2 1.217e-02) | val MSE 3.015e-06 (L2 1.227e-02) | 65.5s
RE_ADAM = re.compile(
    r"\[ADAM\]\s+ep\s+(\d+)/(\d+)\s+"
    r"train MSE\s+([0-9.eE+-]+)\s+\(L2\s+([0-9.eE+-]+)\)\s+\|\s+"
    r"val MSE\s+([0-9.eE+-]+)\s+\(L2\s+([0-9.eE+-]+)\)\s+\|\s+"
    r"([0-9.]+)s"
)
# [LBFGS] ep   3/30  val MSE 1.234e-08 (L2 1.10e-04) | 12.0s
RE_LBFGS = re.compile(
    r"\[LBFGS\]\s+ep\s+(\d+)/(\d+)\s+"
    r"val MSE\s+([0-9.eE+-]+)\s+\(L2\s+([0-9.eE+-]+)\)\s+\|\s+"
    r"([0-9.]+)s"
)


def parse(path: str) -> list:
    history = []
    with open(path) as f:
        for line in f:
            m = RE_ADAM.search(line)
            if m:
                ep, _tot, tr, trl2, va, val2, w = m.groups()
                history.append({
                    "phase": "adam", "epoch": int(ep),
                    "train_mse": float(tr), "train_l2_ratio": float(trl2),
                    "val_mse":   float(va), "val_l2_ratio":   float(val2),
                    "wall_s":    float(w),
                })
                continue
            m = RE_LBFGS.search(line)
            if m:
                ep, _tot, va, val2, w = m.groups()
                history.append({
                    "phase": "lbfgs", "epoch": int(ep),
                    "val_mse": float(va), "val_l2_ratio": float(val2),
                    "wall_s":  float(w),
                })
    return history


def plot(history: list, out_dir: str, title: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    adam = [h for h in history if h["phase"] == "adam"]
    lbfgs = [h for h in history if h["phase"] == "lbfgs"]

    fig, (ax_mse, ax_l2) = plt.subplots(1, 2, figsize=(11, 4))

    if adam:
        ep_a = [h["epoch"] for h in adam]
        ax_mse.semilogy(ep_a, [h["train_mse"] for h in adam], "o-", label="Adam train")
        ax_mse.semilogy(ep_a, [h["val_mse"]   for h in adam], "s-", label="Adam val")
        ax_l2.semilogy(ep_a, [h["train_l2_ratio"] for h in adam], "o-", label="Adam train")
        ax_l2.semilogy(ep_a, [h["val_l2_ratio"]   for h in adam], "s-", label="Adam val")
    if lbfgs:
        ep_l = [adam[-1]["epoch"] + h["epoch"] for h in lbfgs]  # continuous x axis
        ax_mse.semilogy(ep_l, [h["val_mse"] for h in lbfgs], "^-", label="L-BFGS val")
        ax_l2.semilogy(ep_l, [h["val_l2_ratio"] for h in lbfgs], "^-", label="L-BFGS val")

    for ax, ylab in [(ax_mse, "MSE on residual $\\delta\\Phi$"),
                     (ax_l2, "$\\|\\delta\\Phi_{err}\\|_2 / \\|\\Phi\\|_2$")]:
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[PLOT] wrote {path}")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    out_log, out_dir = sys.argv[1], sys.argv[2]
    history = parse(out_log)
    if not history:
        print(f"[PLOT] no [ADAM]/[LBFGS] lines parsed from {out_log}")
        sys.exit(1)
    hist_path = os.path.join(out_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump({"history": history, "source_log": os.path.abspath(out_log)},
                  f, indent=2)
    print(f"[PLOT] wrote {hist_path} ({len(history)} epochs parsed)")
    plot(history, out_dir, title=os.path.basename(out_dir))


if __name__ == "__main__":
    main()
