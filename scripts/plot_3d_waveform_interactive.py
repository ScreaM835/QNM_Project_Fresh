"""Interactive 3D plot of the Schwarzschild (SW) ringdown waveform (FD oracle).

Schwarzschild analogue of ``kerr/scripts/plot_3d_waveform_kerr.py``.  Unlike
Kerr, the Schwarzschild master field ``Phi(x*, t)`` is purely REAL, so a single
rotatable surface shows the whole solution.  Output is a self-contained,
rotatable/zoomable Plotly HTML (open it in any browser; works offline via the
plotly CDN).

Examples
--------
    python scripts/plot_3d_waveform_interactive.py
    python scripts/plot_3d_waveform_interactive.py --data outputs/fd/regge_wheeler_l2_paper_fd.npz
    python scripts/plot_3d_waveform_interactive.py --zoom-x -20 60 --out outputs/qnm/sw_waveform_3d.html
"""
import argparse
import glob
import os

import numpy as np
import plotly.graph_objects as go


def _find_default_data() -> str:
    patterns = [
        "outputs/fd/*zerilli*_fd.npz",
        "outputs/pinn/zerilli_l2_greedy_f03_lbfgs30k/*_fd.npz",
        "outputs/pinn/zerilli_l2_paper/*_fd.npz",
        "outputs/fd/*_fd.npz",
        "outputs/**/*_fd.npz",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    raise SystemExit(
        "No *_fd.npz found under outputs/.  Pass one explicitly with --data."
    )


def main():
    ap = argparse.ArgumentParser(description="Interactive 3D Schwarzschild waveform plot")
    ap.add_argument("--data", type=str, default=None,
                    help="FD npz with x, t, phi (auto-detected if omitted)")
    ap.add_argument("--zoom-x", nargs=2, type=float, default=None,
                    metavar=("XMIN", "XMAX"),
                    help="Zoom into a spatial sub-range, e.g. --zoom-x -20 60")
    ap.add_argument("--max-points", type=int, default=300,
                    help="max grid points per axis (downsample for the browser)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    data_path = args.data or _find_default_data()
    print(f"Loading {data_path}", flush=True)
    d = np.load(data_path)
    x = d["x"].astype(np.float64)          # (Nx,)
    t = d["t"].astype(np.float64)          # (Nt,)
    phi = d["phi"].astype(np.float64)      # (Nt, Nx)
    print(f"  x:{x.shape}  t:{t.shape}  phi:{phi.shape}", flush=True)

    if args.zoom_x:
        mask = (x >= args.zoom_x[0]) & (x <= args.zoom_x[1])
        x = x[mask]
        phi = phi[:, mask]
        print(f"  zoomed x to [{args.zoom_x[0]}, {args.zoom_x[1]}] -> {len(x)} pts",
              flush=True)

    # downsample for a smooth but light browser surface
    st = max(1, len(t) // args.max_points)
    sx = max(1, len(x) // args.max_points)
    xg = x[::sx]
    tg = t[::st]
    Z = phi[::st, ::sx]
    print(f"  grid {phi.shape} -> plotted {Z.shape} (stride t={st}, x={sx})", flush=True)

    vmax = float(np.max(np.abs(Z))) or 1.0

    fig = go.Figure(data=[
        go.Surface(
            x=xg, y=tg, z=Z,
            colorscale="RdBu",
            reversescale=True,
            cmin=-vmax, cmax=vmax,
            showscale=True,
            colorbar=dict(title="Phi", len=0.6),
        )
    ])
    fig.update_layout(
        title=(f"Schwarzschild ringdown (FD reference): {os.path.basename(data_path)}<br>"
               f"<sub>x* = tortoise coordinate (horizon at x*&#8594;-&#8734;), "
               f"real master field &#934;(x*, t)</sub>"),
        scene=dict(
            xaxis_title="Tortoise coordinate x*",
            yaxis_title="Time t / M",
            zaxis_title="Phi(x*, t)",
            camera=dict(eye=dict(x=1.6, y=-1.7, z=1.0)),
        ),
        width=1100, height=820, margin=dict(l=0, r=0, t=80, b=0),
    )

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(data_path))[0]
        out = f"outputs/qnm/{base}_waveform_3d.html"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {out}", flush=True)
    print("open in a browser (drag to rotate, scroll to zoom)", flush=True)


if __name__ == "__main__":
    main()
