"""Interactive 3D plot of the Kerr ringdown waveform (FD oracle).

Analogue of the Schwarzschild ``plot_3d_waveform.py``, but for Kerr the master
(Teukolsky) field ``Psi`` is **complex** -- frame-dragging gives it a genuine
imaginary part that grows with spin -- so a single real surface (as for SW)
cannot show the whole solution. This script renders three surfaces over the
hyperboloidal slice ``(sigma, tau)`` -- ``Re(Psi)``, ``Im(Psi)`` and the
envelope ``|Psi|`` -- and lets you switch between them with buttons in a
self-contained, rotatable/zoomable Plotly HTML (open it in any browser; works
over SSH, no display needed).

Coordinates:
  * ``sigma`` in [0, 1] is the compactified hyperboloidal radius: ``sigma = 0``
    is future null infinity (scri, where the gravitational wave is read off)
    and ``sigma = 1`` is the horizon.
  * ``tau`` is hyperboloidal time, in units of ``M``.

Data source: the validated Phase B/C FD corpus (``dataset_*.npz``), which stores
the field as ``*_psi_fine_re`` / ``*_psi_fine_im`` of shape
``(n_config, n_tau, n_sigma)`` with per-config parameters ``*_P`` = (a/M, r0, w)
and reference QNMs ``*_qnm`` = (M*omega_R, M*omega_I, tau/M).

Usage (from the improved-repo ROOT):
  venv_csd3/bin/python kerr/scripts/plot_3d_waveform_kerr.py --spin 0.7
  venv_csd3/bin/python kerr/scripts/plot_3d_waveform_kerr.py --spin 0.945 \
      --out kerr/outputs/phase_c/kerr_waveform_3d_a094.html
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _split(prefix: str, files) -> str:
    """Return the dataset key prefix present in the npz (test/val/train)."""
    for cand in (prefix, "test", "val", "train"):
        if cand and f"{cand}_psi_fine_re" in files:
            return cand
    raise KeyError("No *_psi_fine_re array found in the dataset.")


def main():
    ap = argparse.ArgumentParser(description="Interactive 3D Kerr waveform plot")
    ap.add_argument("--data", type=str,
                    default="kerr/outputs/phase_c/dataset_test.npz",
                    help="corpus npz with *_psi_fine_re/_im, *_P, *_qnm")
    ap.add_argument("--split", type=str, default="test",
                    choices=["test", "val", "train"],
                    help="which split prefix to read from the npz")
    ap.add_argument("--spin", type=float, default=0.7,
                    help="target a/M; the nearest available config is plotted")
    ap.add_argument("--component", type=str, default="re",
                    choices=["re", "im", "abs"],
                    help="surface shown first (all three are toggle-able)")
    ap.add_argument("--max-points", type=int, default=220,
                    help="max grid points per axis (downsample for the browser)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    pre = _split(args.split, d.files)

    sigma = d["sigma_fine"]                       # (n_sigma,)
    tau = d["tau"]                                # (n_tau,)
    P = d[f"{pre}_P"]                             # (n_config, 3) = a/M, r0, w
    Q = d[f"{pre}_qnm"]                           # (n_config, 3)
    re_all = d[f"{pre}_psi_fine_re"]             # (n_config, n_tau, n_sigma)
    im_all = d[f"{pre}_psi_fine_im"]

    # pick the config whose spin is closest to the requested one
    i = int(np.argmin(np.abs(P[:, 0] - args.spin)))
    a_over_M, r0, w = float(P[i, 0]), float(P[i, 1]), float(P[i, 2])
    mw_r, mw_i, tau_qnm = float(Q[i, 0]), float(Q[i, 1]), float(Q[i, 2])

    re = re_all[i].astype(np.float64)            # (n_tau, n_sigma)
    im = im_all[i].astype(np.float64)
    amp = np.hypot(re, im)

    # downsample for a smooth but light browser surface
    st = max(1, len(tau) // args.max_points)
    ss = max(1, len(sigma) // args.max_points)
    sg = sigma[::ss]
    ta = tau[::st]
    ReZ = re[::st, ::ss]
    ImZ = im[::st, ::ss]
    AbsZ = amp[::st, ::ss]

    print(f"config idx {i}: a/M={a_over_M:.4f}  r0={r0:.3f}M  w={w:.3f}M", flush=True)
    print(f"  QNM ref: M*omega_R={mw_r:.4f}  M*omega_I={mw_i:.4f}  tau/M={tau_qnm:.3f}",
          flush=True)
    print(f"  grid {re.shape} -> plotted {ReZ.shape} (stride tau={st}, sigma={ss})",
          flush=True)
    print(f"  |Psi|max={amp.max():.3e}  max|Im|/max|Re|={np.max(np.abs(im))/max(np.max(np.abs(re)),1e-30):.3f}",
          flush=True)

    cmap = "Plasma"
    surf_kw = dict(x=sg, y=ta, colorscale=cmap, showscale=True,
                   colorbar=dict(title="amp", len=0.6))
    order = {"re": 0, "im": 1, "abs": 2}[args.component]
    vis = [False, False, False]
    vis[order] = True

    fig = go.Figure(data=[
        go.Surface(z=ReZ, name="Re(Psi)", visible=vis[0], **surf_kw),
        go.Surface(z=ImZ, name="Im(Psi)", visible=vis[1], **surf_kw),
        go.Surface(z=AbsZ, name="|Psi|", visible=vis[2], **surf_kw),
    ])

    def _btn(label, k, ztitle):
        v = [False, False, False]
        v[k] = True
        return dict(label=label, method="update",
                    args=[{"visible": v},
                          {"scene.zaxis.title": ztitle}])

    fig.update_layout(
        title=(f"Kerr ringdown (FD): a/M = {a_over_M:.3f}, "
               f"r0 = {r0:.2f}M, w = {w:.2f}M<br>"
               f"<sub>M&#969;<sub>R</sub> = {mw_r:.4f}, "
               f"&#964;/M = {tau_qnm:.2f} &nbsp;|&nbsp; "
               f"&#963;=0 scri (GW), &#963;=1 horizon</sub>"),
        scene=dict(
            xaxis_title="sigma  (0 = scri, 1 = horizon)",
            yaxis_title="tau / M",
            zaxis_title={"re": "Re(Psi)", "im": "Im(Psi)",
                         "abs": "|Psi|"}[args.component],
            camera=dict(eye=dict(x=1.6, y=-1.7, z=1.0)),
        ),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.0, y=1.08,
            xanchor="left", yanchor="top", showactive=True,
            buttons=[_btn("Re(\u03a8)", 0, "Re(Psi)"),
                     _btn("Im(\u03a8)", 1, "Im(Psi)"),
                     _btn("|\u03a8|", 2, "|Psi|")],
        )],
        width=1100, height=820, margin=dict(l=0, r=0, t=80, b=0),
    )

    out = args.out
    if out is None:
        tag = f"{a_over_M:.3f}".replace(".", "")
        out = f"kerr/outputs/phase_c/kerr_waveform_3d_a{tag}.html"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {out}", flush=True)
    print(f"open in a browser (rotate/zoom; buttons toggle Re / Im / |Psi|)",
          flush=True)


if __name__ == "__main__":
    main()
