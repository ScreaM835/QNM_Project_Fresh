"""Isolation test (a): how much error does the coarse->fine UPSAMPLING add,
and can a band-limited resampler beat the production cubic spline?

Context
-------
The hybrid pipeline (src/hybrid_data_pipe.py: upsample_to_fine) lifts the coarse
field onto the fine grid with a CUBIC SPLINE in (t, x) before the FNO sees it.
That is a SECOND, independent error source on top of the coarse-solve
dispersion tested in coarse_stencil_isolation.py: a spline rings/overshoots on
steep gradients, re-roughening even a perfect coarse field.

To isolate the upsampler ALONE we feed it an EXACT coarse field (the converged
6th-order truth, subsampled onto the coarse grid -- so it carries zero solver
error) and compare the upsampled result against the truth on the fine grid. Any
difference is pure interpolation error.

The coarse->fine map is exactly factor-2 in each axis on this config
(dx 0.4->0.2, dt 0.2->0.1; Nx 501->1001=2*501-1, Nt 251->501=2*251-1), which is
the natural setting for band-limited (sinc) resampling.

Resamplers compared on the same field:
    linear     RegularGridInterpolator(method="linear")
    cubic      RegularGridInterpolator(method="cubic")   = production
    quintic    RegularGridInterpolator(method="quintic")
    spectral   even-reflection FFT zero-pad (band-limited / DCT-I upsample)

Production code untouched. Login-node safe (seconds, tiny memory).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _REPO)

from coarse_stencil_isolation import (  # noqa: E402  (sibling script, no side effects on import)
    solve, d2_order6,
    DX_COARSE, DT_COARSE, DX_FINE, DT_FINE, DX_TRUTH, DT_TRUTH,
)

OUTDIR = os.path.abspath(os.path.join(_REPO, "outputs", "exploration", "coarse_stencil"))


# ---------------------------------------------------------------------------
# Band-limited (sinc) upsampling by factor 2 via even reflection + FFT zero-pad.
# Even reflection makes the data periodic without an endpoint jump (DCT-I), so
# the resample is band-limited with no periodic-wrap discontinuity.
# ---------------------------------------------------------------------------

def _spectral_upsample2_1d(f: np.ndarray) -> np.ndarray:
    """N samples on [0, L] -> 2N-1 samples on the same interval, band-limited."""
    n = f.shape[-1]
    # even (mirror) extension, period L = 2N-2 :  f0 f1 ... f_{N-1} f_{N-2} ... f1
    g = np.concatenate([f, f[..., -2:0:-1]], axis=-1)        # length 2N-2
    big = 2 * g.shape[-1]                                     # length 4N-4
    g_up = np.fft.irfft(np.fft.rfft(g, axis=-1), n=big, axis=-1) * 2.0
    return g_up[..., : 2 * n - 1]                            # first 2N-1 points


def spectral_upsample2_2d(phi_c: np.ndarray) -> np.ndarray:
    """Upsample (Nt_c, Nx_c) -> (2Nt_c-1, 2Nx_c-1) separably (x then t)."""
    tmp = _spectral_upsample2_1d(phi_c)                       # along x (last axis)
    return _spectral_upsample2_1d(tmp.T).T                    # along t


def rgi_upsample(method: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def up(phi_c, x_c, t_c, x_f, t_f):
        interp = RegularGridInterpolator(
            (t_c, x_c), phi_c, method=method, bounds_error=False, fill_value=0.0,
        )
        T, X = np.meshgrid(t_f, x_f, indexing="ij")
        return interp(np.stack([T.ravel(), X.ravel()], axis=1)).reshape(t_f.size, x_f.size)
    return up


def _self_test() -> None:
    """Sanity: band-limited upsampler must reproduce a smooth low-k signal exactly."""
    n = 51
    s = np.linspace(0.0, 1.0, n)
    f = np.cos(2.0 * np.pi * 1.5 * s) + 0.3 * np.cos(2.0 * np.pi * 3.0 * s)
    up = _spectral_upsample2_1d(f)
    sf = np.linspace(0.0, 1.0, 2 * n - 1)
    ref = np.cos(2.0 * np.pi * 1.5 * sf) + 0.3 * np.cos(2.0 * np.pi * 3.0 * sf)
    err = float(np.max(np.abs(up - ref)))
    assert err < 1e-9, f"spectral upsampler self-test FAILED, Linf={err:.2e}"
    print(f"  [self-test] band-limited factor-2 upsample Linf on smooth signal = {err:.2e}  OK", flush=True)


# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    print("=== Upsampling isolation test (exact coarse field, no solver error) ===", flush=True)
    _self_test()

    # converged truth on the dx=0.1 grid
    t0 = time.time()
    xt, tt, phi_truth = solve(d2_order6, DX_TRUTH, DT_TRUTH)
    print(f"  truth solve (6th, dx={DX_TRUTH}): {time.time()-t0:.2f}s, grid {phi_truth.shape}", flush=True)

    sx_c, st_c = int(round(DX_COARSE / DX_TRUTH)), int(round(DT_COARSE / DT_TRUTH))  # 4, 4
    sx_f, st_f = int(round(DX_FINE / DX_TRUTH)), int(round(DT_FINE / DT_TRUTH))      # 2, 2

    x_c, t_c = xt[::sx_c], tt[::st_c]
    x_f, t_f = xt[::sx_f], tt[::st_f]
    phi_coarse_exact = phi_truth[::st_c, ::sx_c]   # exact field on the coarse grid
    phi_fine_truth = phi_truth[::st_f, ::sx_f]     # target on the fine grid

    assert x_f.size == 2 * x_c.size - 1 and t_f.size == 2 * t_c.size - 1, "non factor-2 grid"

    wave_mask = np.abs(x_f) > 10.0

    upsamplers: Dict[str, Callable] = {
        "linear":   rgi_upsample("linear"),
        "cubic":    rgi_upsample("cubic"),     # production
        "quintic":  rgi_upsample("quintic"),
    }

    results: Dict[str, Dict] = {}
    err_fields: Dict[str, np.ndarray] = {}

    for name, up in upsamplers.items():
        t0 = time.time()
        phi_up = up(phi_coarse_exact, x_c, t_c, x_f, t_f)
        wall = time.time() - t0
        err = phi_up - phi_fine_truth
        err_fields[name] = err
        results[name] = {
            "l2": float(np.sqrt(np.mean(err**2))),
            "linf": float(np.max(np.abs(err))),
            "l2_wave": float(np.sqrt(np.mean(err[:, wave_mask]**2))),
            "wall_s": wall,
        }

    # spectral (separable factor-2)
    t0 = time.time()
    phi_sp = spectral_upsample2_2d(phi_coarse_exact)
    wall = time.time() - t0
    err = phi_sp - phi_fine_truth
    err_fields["spectral"] = err
    results["spectral"] = {
        "l2": float(np.sqrt(np.mean(err**2))),
        "linf": float(np.max(np.abs(err))),
        "l2_wave": float(np.sqrt(np.mean(err[:, wave_mask]**2))),
        "wall_s": wall,
    }

    print("\n=== Pure upsampling error (exact coarse -> fine vs truth) ===", flush=True)
    print(f"{'method':>10} {'L2':>11} {'Linf':>11} {'L2(wave)':>11} {'wall_s':>8} {'L2 vs cubic':>12}", flush=True)
    base = results["cubic"]["l2"]
    for name in ["linear", "cubic", "quintic", "spectral"]:
        r = results[name]
        tag = "  (production)" if name == "cubic" else ""
        print(f"{name:>10} {r['l2']:>11.3e} {r['linf']:>11.3e} {r['l2_wave']:>11.3e} "
              f"{r['wall_s']:>8.3f} {base/r['l2']:>11.1f}x{tag}", flush=True)

    print("\n  Context (from coarse_stencil_isolation.py, same config):", flush=True)
    print("    coarse 2nd-order SOLVE error  L2 ~ 1.85e-3", flush=True)
    print("    coarse 4th-order SOLVE error  L2 ~ 1.69e-5", flush=True)
    print("    coarse 6th-order SOLVE error  L2 ~ 4.88e-6", flush=True)
    print("  -> compare the upsampling L2 above against these to see which source dominates.", flush=True)

    # heatmaps
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        order = ["linear", "cubic", "quintic", "spectral"]
        allerr = np.concatenate([np.abs(err_fields[k]).ravel() for k in order])
        vmax = float(np.percentile(allerr, 99.9))
        vmin = max(vmax * 1e-4, 1e-9)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        for ax, name in zip(axes, order):
            err = err_fields[name]
            im = ax.pcolormesh(t_f, x_f, np.abs(err).T,
                               norm=LogNorm(vmin=vmin, vmax=vmax),
                               cmap="magma_r", shading="auto")
            tag = " (prod)" if name == "cubic" else ""
            ax.set_title(f"{name}{tag}  (L2={results[name]['l2']:.2e})")
            ax.set_xlabel("t / M")
            ax.set_ylabel("x* / M")
        fig.colorbar(im, ax=axes, location="right", shrink=0.8,
                     label="|upsample(Phi_coarse_exact) - Phi_truth_fine|")
        fig.suptitle("Pure upsampling error vs resampler (exact coarse field, factor-2 to fine grid)",
                     fontsize=14)
        figpath = os.path.join(OUTDIR, "upsample_error_heatmaps.png")
        fig.savefig(figpath, dpi=120)
        print(f"\nSaved heatmaps: {figpath}", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"\n[plot skipped: {e}]", flush=True)

    np.savez(os.path.join(OUTDIR, "upsample_errors.npz"),
             x_fine=x_f, t_fine=t_f,
             **{f"err_{k}": v for k, v in err_fields.items()},
             **{f"l2_{k}": results[k]["l2"] for k in results})
    print(f"Saved arrays: {os.path.join(OUTDIR, 'upsample_errors.npz')}", flush=True)


if __name__ == "__main__":
    main()
