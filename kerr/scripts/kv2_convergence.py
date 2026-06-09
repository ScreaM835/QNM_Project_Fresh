"""KV.2 self-convergence gate for the complex Kerr Teukolsky operator.

Kerr analogue of the Phase A V.2 self-convergence gate
(scripts/v2_self_convergence.py). With the FULL Teukolsky operator (s=-2,
l=m=2, the complex minimal-gauge characteristic form) evolve IDENTICAL initial
data on three NESTED grids and verify the spatial discretisation converges at
the design order (2nd).

Method (identical to the validated Phase A V.2, generalised to complex Kerr):

  * Nested grids  sigma = linspace(eps, 1-eps, N),  N in {401, 801, 1601}.
    Since 400*2 = 800 and 800*2 = 1600 and eps is the SAME on every grid,
    coarse point i coincides EXACTLY with mid point 2i and fine point 4i.
    No interpolation; we difference coincident points (psi2[::2], psi3[::4]).

  * A SINGLE shared dt (set by the COARSEST-grid CFL, safety=0.4) is used for
    all three grids, so the RK4 time error O(dt^4) is IDENTICAL on every grid
    and cancels exactly in the self-convergence differences -- what remains is
    the spatial error. Two doublings at safety 0.4 keep the finest-grid CFL
    number at 0.4*4 = 1.6, comfortably inside the RK4 stability bound (~2.8),
    so the shared-dt evolution is stable on all three grids.

  * Self-convergence at fixed final tau:
        e12 = psi_401        - psi_801[::2]
        e23 = psi_801[::2]   - psi_1601[::4]
        Q   = ||e12|| / ||e23||,    p = log2(Q).
    A 2nd-order-accurate spatial scheme gives Q -> 4, p -> 2. Reported in both
    discrete L2 and max norm, for the final FIELD and for two observer TIME
    SERIES (the scri+ waveform at index 0, coincident on every grid; and a
    finite-radius r=10M observer at coincident indices i,2i,4i).

  * What is GATED vs REPORTED. The gate requires clean 2nd order in BOTH the
    final FIELD over the whole slice and the scri+ waveform time series (the
    physical gravitational-wave output, the quantity B.8 extracts QNMs from).
    The exterior-bulk field (sigma <= BULK_SIGMA_MAX) and the r=10M observer
    series are also computed and REPORTED: the bulk-vs-full field comparison is
    a deliberate cross-check of the KV.1 near-horizon rescaling finding and
    confirms the convergence ORDER is uniform across the slice (the horizon
    layer inflates the field magnitude but not its truncation order -- bulk and
    full p agree to ~0.02), so no region is excluded from the gate.

  * Pre-asymptotic note (honest). The Gaussian launch pulse (width 1M at r=10M)
    maps to a sigma-width ~0.014; the design 2nd order needs ~5+ points across
    it, i.e. N >= 401. On coarser grids p is pre-asymptotic (p_field rises 1.50
    -> 1.86 -> ~2 along 101/201/401 -> 201/401/801 -> 401/801/1601). The gate
    therefore runs at 401/801/1601.

Kreiss-Oliger note (honest). The production operator carries KO dissipation at
SIGMA_KO=0.2 (needed for high-spin stability, established in KV.1/B.6); this
gate tests THAT operator, not a KO-free one. KO is a 4th-difference,
-(sigma/16)*delta^4 u = -(sigma/16)(d sigma)^4 d^4u/dsigma^4 + ..., i.e. an
O(d sigma^4) modification of the RHS. It is therefore subdominant to the
O(d sigma^2) central-difference truncation by a factor ~ (C_KO/C2)(d sigma)^2
~ 1e-5 at N=401, so it does NOT degrade the measured 2nd order; raising it from
the Phase A 0.02 to 0.2 changes the differences negligibly. (The same KO is
applied identically on all three grids.)

a/M=0.9 is the gate spin (the plan's KV.2 target -- the hardest case, where the
frame-dragging coupling and the strongest near-horizon rescaling oscillation
live). a=0 is also run as a reported cross-check that the Bardeen-Press
reduction is likewise 2nd-order (relevant to the B.8 a=0 gate).

Output: kerr/outputs/phase_b/kv2_conv_<JOBID>.npz
Runs on SLURM (CPU). A cheap login-node smoke test:
    venv_csd3/bin/python -u kerr/scripts/kv2_convergence.py --smoke
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

from kerr.src.teukolsky_minimal_gauge import (
    build_teukolsky_op,
    rhs_teuk,
    state_from_psi,
    cfl_dt,
    observer_index,
    scri_index,
)
from kerr.src.fd_stencils import d1_central
from kerr.src.mol_rk4 import integrate_state
from kerr.src.dissipation import ko_dissipation
from kerr.src.qnm_kerr_reference import kerr_qnm


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
M = 1.0
ELL = 2
MM = 2                       # azimuthal m
TAU_FINAL = 60.0             # well past the launch pulse, field still >> floor
SAFETY = 0.4                 # coarse CFL safety; 2 doublings -> finest CFL 1.6
SIGMA_KO = 0.2               # production value (KV.1); O(dsigma^4), see docstring
RESOLUTIONS = [401, 801, 1601]   # nested: 400, 800, 1600 doublings
SPINS = [0.0, 0.9]           # gate spin is GATE_SPIN; the other is reported
GATE_SPIN = 0.9
ID_R0 = 10.0
ID_WIDTH = 1.0
ID_AMP = 1.0
OBS_RM = 10.0                # coincident interior observer radius r/M
# Exterior bulk excludes the near-horizon layer sigma > BULK_SIGMA_MAX. This is
# a transparency CROSS-CHECK against the KV.1/B.6 finding that the regular-field
# rescaling Psi = psi*sigma^3*(1-sigma)^{-2-i beta} amplifies the near-horizon
# field: it confirms the spatial CONVERGENCE ORDER is uniform across the slice
# (bulk and full field converge at the same p -- the horizon rescaling inflates
# the field's magnitude but NOT its truncation order). Both are reported; the
# gate uses the FULL field plus the scri+ waveform.
BULK_SIGMA_MAX = 0.9
# Acceptance band for the convergence order p = log2(Q). The Phase A V.2 gate
# treats 1.7-2.3 as a clean 2nd-order pass and <1.5 or >2.7 as a problem; we
# adopt the same band for the gate.
P_LO = 1.7
P_HI = 2.3
OUTDIR = os.path.abspath(os.path.join(_THIS, "..", "outputs", "phase_b"))


if "--smoke" in sys.argv:
    # Cheap login-node check: nested 100/200/400, short tau. Verifies the script
    # runs, grids nest, and p is in the right ballpark -- NOT an authoritative
    # gate (coarse grids under-resolve the launch pulse, so p is pre-asymptotic;
    # ~5+ points across the sigma-width of the Gaussian, reached at N>=401, are
    # needed for the design 2nd order -- verified: p_field 1.50 at 101/201/401 ->
    # 1.86 at 201/401/801 -> ~2 at the gate grid 401/801/1601).
    RESOLUTIONS = [101, 201, 401]
    TAU_FINAL = 20.0


def make_initial_pulse(amp, r0, width):
    def psi0(r):
        return amp * np.exp(-((r - r0) ** 2) / (2.0 * width ** 2))
    return psi0


def evolve(a_over_M, N, dt, n_steps, record_every, i_c_coarse):
    """Evolve the full Teukolsky operator on grid N with the SHARED dt.

    Returns (op, psi_final, taus, series, elapsed). `series` holds the scri+
    (index 0) and r=10M (coincident index factor*i_c_coarse) waveform time
    series; both are sampled at the same tau on every grid (shared dt/n_steps).
    """
    ref = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    omega_ref = complex(ref.M_omega_R, ref.M_omega_I)
    op = build_teukolsky_op(
        N=N, a_over_M=a_over_M, M=M, ell=ELL, m=MM,
        omega_ref=omega_ref, include_potential=True,
    )
    state0 = state_from_psi(make_initial_pulse(ID_AMP, ID_R0, ID_WIDTH), op, d1_central)

    def rhs_fn(s):
        dPsi, dU, dW = rhs_teuk(s, op, d1_central)
        dU = dU + ko_dissipation(s[1], SIGMA_KO)
        dW = dW + ko_dissipation(s[2], SIGMA_KO)
        return dPsi, dU, dW

    factor = (N - 1) // (RESOLUTIONS[0] - 1)   # 1, 2, 4 for 401/801/1601
    observers = {"scri": scri_index(op), "r10M": factor * i_c_coarse}

    t0 = time.time()
    state, taus, series = integrate_state(
        state0, dt, n_steps, rhs_fn,
        observer_field=0, record_every=record_every, observers=observers,
    )
    elapsed = time.time() - t0
    return op, state[0], taus, series, elapsed


def conv_from_triplet(a, b, c):
    """Self-convergence metrics from three ALIGNED (coincident) arrays.

    a, b, c are the coarse/mid/fine samples on coincident points/times.
    Uses |.| so it is correct for the complex Teukolsky field.
    """
    e12 = a - b
    e23 = b - c
    l2_e12 = float(np.sqrt(np.mean(np.abs(e12) ** 2)))
    l2_e23 = float(np.sqrt(np.mean(np.abs(e23) ** 2)))
    max_e12 = float(np.max(np.abs(e12)))
    max_e23 = float(np.max(np.abs(e23)))
    Q_l2 = l2_e12 / l2_e23 if l2_e23 > 0 else float("inf")
    Q_max = max_e12 / max_e23 if max_e23 > 0 else float("inf")
    p_l2 = float(np.log2(Q_l2)) if np.isfinite(Q_l2) and Q_l2 > 0 else float("nan")
    p_max = float(np.log2(Q_max)) if np.isfinite(Q_max) and Q_max > 0 else float("nan")
    return dict(
        l2_e12=l2_e12, l2_e23=l2_e23, max_e12=max_e12, max_e23=max_e23,
        Q_l2=Q_l2, Q_max=Q_max, p_l2=p_l2, p_max=p_max,
    )


def run_spin(a_over_M):
    """Run the three nested grids at one spin and compute self-convergence."""
    N0, N1, N2 = RESOLUTIONS
    f1, f2 = (N1 - 1) // (N0 - 1), (N2 - 1) // (N0 - 1)

    # Shared dt from the COARSEST-grid CFL (so the RK4 time error cancels).
    ref = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    omega_ref = complex(ref.M_omega_R, ref.M_omega_I)
    op_coarse = build_teukolsky_op(
        N=N0, a_over_M=a_over_M, M=M, ell=ELL, m=MM,
        omega_ref=omega_ref, include_potential=True,
    )
    dt = cfl_dt(op_coarse, safety=SAFETY)
    n_steps = int(np.ceil(TAU_FINAL / dt))
    dt = TAU_FINAL / n_steps
    record_every = max(1, n_steps // 2000)
    i_c = observer_index(op_coarse, OBS_RM)   # coarse index of the r=10M observer

    print(f"\n=== a/M = {a_over_M}  (qnm: M_omega_R={ref.M_omega_R:.5f}, "
          f"M_omega_I={ref.M_omega_I:.5f}, tau/M={ref.tau_over_M:.3f}) ===", flush=True)
    print(f"  shared dt = {dt:.4e}  n_steps = {n_steps}  record_every = {record_every}  "
          f"sigma_KO = {SIGMA_KO}", flush=True)
    print(f"  r=10M observer coarse index i_c = {i_c} "
          f"(sigma = {op_coarse.sigma[i_c]:.6f})", flush=True)

    ops, psis, series_all = {}, {}, {}
    for N in RESOLUTIONS:
        op, psi_final, taus, series, elapsed = evolve(
            a_over_M, N, dt, n_steps, record_every, i_c)
        ops[N], psis[N], series_all[N] = op, psi_final, series
        finite = bool(np.all(np.isfinite(psi_final)))
        print(f"  N={N:5d}: {elapsed:6.1f}s  |Psi|_max(tau_f) = "
              f"{np.max(np.abs(psi_final)):.4e}  finite={finite}", flush=True)

    # Verify the grids actually nest (coincident sigma) before differencing.
    assert np.allclose(ops[N0].sigma, ops[N1].sigma[::f1], atol=1e-12), "grid 801 not nested"
    assert np.allclose(ops[N0].sigma, ops[N2].sigma[::f2], atol=1e-12), "grid 1601 not nested"

    finite_all = all(bool(np.all(np.isfinite(psis[N]))) for N in RESOLUTIONS)

    # Final FIELD self-convergence on coincident points (the N=401 grid).
    # Full grid (horizon-rescaling-dominated, reported) and exterior bulk
    # (sigma <= BULK_SIGMA_MAX, the physical content -- gated), see KV.1.
    a_f, b_f, c_f = psis[N0], psis[N1][::f1], psis[N2][::f2]
    field_full = conv_from_triplet(a_f, b_f, c_f)
    bulk = ops[N0].sigma <= BULK_SIGMA_MAX
    field_bulk = conv_from_triplet(a_f[bulk], b_f[bulk], c_f[bulk])

    # Observer TIME-SERIES self-convergence (already time-aligned: shared dt).
    obs_conv = {}
    for key in ("scri", "r10M"):
        obs_conv[key] = conv_from_triplet(
            series_all[N0][key], series_all[N1][key], series_all[N2][key])

    return dict(
        a_over_M=a_over_M, dt=dt, n_steps=n_steps, finite_all=finite_all,
        field_full=field_full, field_bulk=field_bulk, obs=obs_conv,
        psis={N: psis[N] for N in RESOLUTIONS},
        series={N: series_all[N] for N in RESOLUTIONS},
        taus=taus,
    )


def _fmt(c):
    return (f"L2: ||e12||={c['l2_e12']:.3e} ||e23||={c['l2_e23']:.3e} "
            f"Q={c['Q_l2']:.3f} p={c['p_l2']:.3f}  |  "
            f"max: Q={c['Q_max']:.3f} p={c['p_max']:.3f}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    smoke = "--smoke" in sys.argv
    print(f"=== KV.2 Kerr self-convergence gate (job {job}{' SMOKE' if smoke else ''}) ===", flush=True)
    print(f"  s=-2, l=m={MM}, full Teukolsky operator, complex characteristic form", flush=True)
    print(f"  Gaussian(amp={ID_AMP}, r0={ID_R0}M, sigma={ID_WIDTH}M)  tau_final={TAU_FINAL}", flush=True)
    print(f"  nested resolutions = {RESOLUTIONS}  (shared dt per spin)", flush=True)
    print(f"  spins = {SPINS}  gate spin = {GATE_SPIN}  accept p in [{P_LO}, {P_HI}]", flush=True)

    results = {}
    for a in SPINS:
        results[a] = run_spin(a)

    print("\n=== Self-convergence summary (p ~ 2 is clean 2nd-order) ===", flush=True)
    gate_ok = True
    for a in SPINS:
        r = results[a]
        tag = " [GATE]" if a == GATE_SPIN else " [report]"
        print(f"\n a/M = {a}{tag}   finite={r['finite_all']}", flush=True)
        print(f"   field full   {_fmt(r['field_full'])}", flush=True)
        print(f"   field bulk   {_fmt(r['field_bulk'])}   (sigma<={BULK_SIGMA_MAX}, horizon cross-check)", flush=True)
        print(f"   scri+ series {_fmt(r['obs']['scri'])}", flush=True)
        print(f"   r10M  series {_fmt(r['obs']['r10M'])}", flush=True)
        if a == GATE_SPIN:
            p_field = r['field_full']['p_l2']
            p_scri = r['obs']['scri']['p_l2']
            ok = (r['finite_all']
                  and P_LO <= p_field <= P_HI
                  and P_LO <= p_scri <= P_HI)
            gate_ok = gate_ok and ok
            print(f"   -> gate: full-field p_l2={p_field:.3f} in band={P_LO <= p_field <= P_HI}, "
                  f"scri p_l2={p_scri:.3f} in band={P_LO <= p_scri <= P_HI}", flush=True)

    # Save everything for transparency / later plotting.
    out_path = os.path.join(OUTDIR, f"kv2_conv_{job}.npz")
    save = dict(
        resolutions=np.array(RESOLUTIONS), spins=np.array(SPINS),
        tau_final=np.array(TAU_FINAL), sigma_ko=np.array(SIGMA_KO),
        gate_spin=np.array(GATE_SPIN),
    )
    for a in SPINS:
        r = results[a]
        ta = f"a{a}".replace(".", "p")
        save[f"{ta}_dt"] = np.array(r["dt"])
        save[f"{ta}_n_steps"] = np.array(r["n_steps"])
        for fk in ("p_l2", "p_max", "l2_e12", "l2_e23", "max_e12", "max_e23"):
            save[f"{ta}_fieldbulk_{fk}"] = np.array(r["field_bulk"][fk])
            save[f"{ta}_fieldfull_{fk}"] = np.array(r["field_full"][fk])
            save[f"{ta}_scri_{fk}"] = np.array(r["obs"]["scri"][fk])
            save[f"{ta}_r10M_{fk}"] = np.array(r["obs"]["r10M"][fk])
        for N in RESOLUTIONS:
            save[f"{ta}_psi_{N}"] = r["psis"][N]
            save[f"{ta}_scri_series_{N}"] = r["series"][N]["scri"]
            save[f"{ta}_r10M_series_{N}"] = r["series"][N]["r10M"]
        save[f"{ta}_taus"] = r["taus"]
    np.savez(out_path, **save)
    print(f"\nSaved: {out_path}", flush=True)

    if smoke:
        print("\n(smoke run -- not an authoritative gate)", flush=True)
        return 0

    print(f"\n{'PASS' if gate_ok else 'FAIL'}: "
          f"KV.2 self-convergence at a/M={GATE_SPIN}", flush=True)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
