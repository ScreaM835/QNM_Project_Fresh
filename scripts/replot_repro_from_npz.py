"""Re-plot reproduction field figures from saved FD/PINN field npz.

Regenerates the snapshot overlay, the absolute-difference snapshots and the
global pointwise-error heatmap in the current house convention (snapshot times
{10,20,30,40} M, magma_r log heatmap, neutral titles) directly from the stored
field arrays, so a reproduction run can be brought to the shared style without
retraining. Requires ``outputs/pinn/<run>/<run>_fd.npz`` and ``<run>_pinn.npz``.

Usage:
    python scripts/replot_repro_from_npz.py --run regge_wheeler_l2_paper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                       # noqa: E402
from src.plotting import (                               # noqa: E402
    plot_snapshots, plot_abs_diff_snapshots, plot_error_heatmap,
)

TIMES = [10.0, 20.0, 30.0, 40.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="regge_wheeler_l2_paper")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    run = args.run
    cfg = load_config(args.config or f"configs/{run}.yaml")
    pot = str(cfg["physics"]["potential"]).title()
    ell = cfg["physics"]["l"]

    outdir = ROOT / "outputs" / "pinn" / run
    fd = np.load(outdir / f"{run}_fd.npz")
    pinn = np.load(outdir / f"{run}_pinn.npz")
    x, t = fd["x"], fd["t"]
    phi_fd, phi_pinn = fd["phi"], pinn["phi"]

    plot_snapshots(
        x, t, phi_fd, phi_pinn, TIMES, str(outdir / "snapshots.png"),
        title=f"Snapshots \u2014 {pot} potential (l={ell})",
    )
    plot_abs_diff_snapshots(
        x, t, phi_fd, phi_pinn, TIMES, str(outdir / "abs_diff_snapshots.png"),
        title=f"Absolute difference \u2014 {pot} potential (l={ell})",
    )
    plot_error_heatmap(
        x, t, phi_fd, phi_pinn, str(outdir / "error_heatmap.png"),
        title=f"Pointwise error \u2014 {pot} (l={ell})",
    )
    print(f"[repro] regenerated snapshots + abs_diff + error_heatmap for {run}")


if __name__ == "__main__":
    main()
