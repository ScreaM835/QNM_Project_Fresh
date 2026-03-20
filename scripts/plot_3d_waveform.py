"""
3D waveform surface plot — continuous smooth rendering.

Usage:
    python scripts/plot_3d_waveform.py [--data PATH] [--out PATH]
                                       [--zoom-x XMIN XMAX]
                                       [--elev E] [--azim A]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import uniform_filter


def smooth_downsample(phi, x, t, nx_out=300, nt_out=300):
    """Anti-aliased downsampling: low-pass filter then sub-sample."""
    sx = max(1, len(x) // nx_out)
    st = max(1, len(t) // nt_out)
    # light Gaussian-ish smoothing to avoid aliasing
    phi_smooth = uniform_filter(phi, size=(st, sx))
    return phi_smooth[::st, ::sx], x[::sx], t[::st]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None,
                        help="Path to *_fd.npz (auto-detected if omitted)")
    parser.add_argument("--out", default="outputs/pinn/waveform_3d.png")
    parser.add_argument("--zoom-x", nargs=2, type=float, default=None,
                        metavar=("XMIN", "XMAX"),
                        help="Zoom into a spatial sub-range, e.g. --zoom-x -20 60")
    parser.add_argument("--elev", type=float, default=30)
    parser.add_argument("--azim", type=float, default=-60)
    args = parser.parse_args()

    # ---- Locate data ----
    if args.data:
        data_path = args.data
    else:
        import glob
        candidates = sorted(glob.glob("outputs/pinn/zerilli_l2_greedy_f03_lbfgs30k/*_fd.npz"))
        if not candidates:
            candidates = sorted(glob.glob("outputs/pinn/zerilli_l2_paper/*_fd.npz"))
        if not candidates:
            candidates = sorted(glob.glob("outputs/pinn/*/*.npz"))
        data_path = candidates[0]
    print(f"Loading {data_path}")
    data = np.load(data_path)
    x, t, phi = data["x"], data["t"], data["phi"]
    print(f"  x: {x.shape}  t: {t.shape}  phi: {phi.shape}")

    # ---- Optional spatial zoom ----
    if args.zoom_x:
        mask = (x >= args.zoom_x[0]) & (x <= args.zoom_x[1])
        x = x[mask]
        phi = phi[:, mask]
        print(f"  Zoomed x to [{args.zoom_x[0]}, {args.zoom_x[1]}], {len(x)} pts")

    # ---- Smooth downsample for rendering ----
    phi_ds, x_ds, t_ds = smooth_downsample(phi, x, t, nx_out=350, nt_out=350)
    X, T = np.meshgrid(x_ds, t_ds)
    print(f"  Plot grid: {X.shape}")

    # ---- Normalise colour to amplitude ----
    vmax = max(abs(phi_ds.min()), abs(phi_ds.max()))

    # ---- Plot ----
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, T, phi_ds,
        rcount=300, ccount=300,     # high polygon count → smooth
        cmap="RdBu_r",             # diverging: red for +, blue for −
        vmin=-vmax, vmax=vmax,
        edgecolor="none",
        antialiased=True,
        alpha=0.95,
    )

    ax.set_xlabel(r"Tortoise coordinate $x_*$", fontsize=13, labelpad=12)
    ax.set_ylabel(r"Time $t/M$", fontsize=13, labelpad=12)
    ax.set_zlabel(r"$\Phi(x_*,\,t)$", fontsize=13, labelpad=10)
    ax.set_title("Zerilli Waveform — FD Reference Solution",
                 fontsize=15, pad=15)
    ax.view_init(elev=args.elev, azim=args.azim)

    # Subtle grid style
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("grey")
    ax.yaxis.pane.set_edgecolor("grey")
    ax.zaxis.pane.set_edgecolor("grey")

    fig.colorbar(surf, shrink=0.55, aspect=12, pad=0.08,
                 label=r"Amplitude $\Phi$")

    plt.savefig(args.out, dpi=250, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
