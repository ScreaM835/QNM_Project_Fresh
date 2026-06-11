"""C.3 single-config Kerr PINN proof.

Trains the time-domain Teukolsky PINN at ONE fixed ``(a/M, r0, w)`` with a
physics-only loss (PDE residual + hard-IC ansatz; no FD field in the loss), then
grades it against the validated Phase B oracle:

  * generate the clean FD reference at the SAME config via the B.9 evolver
    (``kerr_dataset.evolve_full_field`` — identical physics to the corpus);
  * predict the PINN scri waveform on the FD tau-axis and compute the field
    relative-L2 error vs the FD oracle;
  * extract the QNM from the PINN waveform with the Phase B extractor
    (``kv3_qnm.extract_fundamental``) and compare M*omega / tau to the `qnm`
    package (Leaver).

ACCEPTANCE (C.3, accuracy go/no-go — NOT a speed claim): scri rel-L2 <= 5%
AND M*omega error <= 1%. Writes the metrics + PASS/FAIL to JSON.

Run from the improved-repo ROOT, e.g.
  venv_csd3/bin/python -u kerr/scripts/train_kerr_pinn.py --spin 0.0 --smoke
  venv_csd3/bin/python -u kerr/scripts/train_kerr_pinn.py --spin 0.0  --out kerr/outputs/phase_c/pinn_single_a0.json
  venv_csd3/bin/python -u kerr/scripts/train_kerr_pinn.py --spin 0.7  --out kerr/outputs/phase_c/pinn_single_a07.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # improved repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # kerr/

from kerr.src.kerr_pinn import (
    KerrPINNConfig, build_model, train, scri_waveform, predict_field, rel_l2,
)
from kerr.src.kerr_dataset import evolve_full_field
from kerr.src.qnm_kerr_reference import kerr_qnm
from kerr.scripts.kv3_qnm import extract_fundamental, M, ELL, MM

REL_L2_GATE = 0.05      # 5% scri waveform field error vs FD oracle
MW_GATE = 0.01          # 1% M*omega error vs Leaver
TAU_REPORT = 0.05       # 5% tau (reported; the locked gate is rel-L2 + Mw)


def main():
    ap = argparse.ArgumentParser(description="C.3 single-config Kerr PINN proof")
    ap.add_argument("--spin", type=float, required=True, help="a/M")
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=None,
                    help="time-domain end (default ~14.5*tau_ref to fit the "
                         "extraction window)")
    ap.add_argument("--adam", type=int, default=10000)
    ap.add_argument("--lbfgs", type=int, default=30000)
    ap.add_argument("--num-domain", type=int, default=20000)
    ap.add_argument("--n-fourier", type=int, default=64)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128, 128])
    ap.add_argument("--out-scale", type=float, default=1.0,
                    help="amplitude preconditioner on the network correction "
                         "(baseline 1.0; raise only if amplitude growth stalls)")
    ap.add_argument("--fourier", action="store_true",
                    help="opt-in Fourier-feature embedding (baseline = plain FNN)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run to validate the pipeline (won't pass)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    spin = float(args.spin)
    is_real = (spin == 0.0)
    ref = kerr_qnm(a_over_M=spin, ell=ELL, m=MM, n=0)
    omega_ref = complex(ref.M_omega_R, ref.M_omega_I)
    tau_ref = float(ref.tau_over_M)
    root = float(np.sqrt(1.0 - spin * spin))
    r_plus = M * (1.0 + root)

    # Time domain: cover the extraction window te in [10,14] tau_ref unless the
    # user overrides. ~14.5 tau_ref leaves the extractor a valid te interval.
    T = float(args.T) if args.T is not None else float(np.ceil(14.5 * tau_ref))

    if args.smoke:
        T = float(np.ceil(4.0 * tau_ref))
        args.adam, args.lbfgs = 400, 0
        args.num_domain, args.n_fourier = 2000, 16
        args.hidden = [32, 32]

    print(f"=== C.3 single-config PINN  a/M={spin:.3f}  r0={args.r0}  w={args.w} ===",
          flush=True)
    print(f"  qnm ref: M*omega_R={ref.M_omega_R:.6f} M*omega_I={ref.M_omega_I:.6f} "
          f"tau/M={tau_ref:.4f}  r_plus={r_plus:.4f}", flush=True)
    print(f"  domain: sigma in [1e-8, 1-1e-8], tau in [0, {T:.1f}]  "
          f"(~{T/tau_ref:.1f} tau_ref)  is_real={is_real}", flush=True)

    # --- FD oracle at the SAME config (validated B.9 physics) ----------------
    t0 = time.time()
    tau_fd, psi_fd, op, info = evolve_full_field(
        spin, 801, args.r0, args.w, t_store=T, dt_store=0.25)
    scri_idx = int(info["scri_idx"])
    scri_fd = psi_fd[:, scri_idx]
    print(f"  FD oracle: N=801 dt={info['dt']:.2e} finite={info['finite']} "
          f"({time.time()-t0:.1f}s); scri |psi|max={np.max(np.abs(scri_fd)):.3e}",
          flush=True)

    # --- build + train PINN --------------------------------------------------
    cfg = KerrPINNConfig(
        a_over_M=spin, omega_ref=omega_ref, r_plus=r_plus,
        A=1.0, r0=args.r0, w=args.w, T=T,
        hidden=tuple(args.hidden), use_fourier=args.fourier,
        n_fourier=args.n_fourier,
        num_domain=args.num_domain, out_scale=args.out_scale,
        seed=args.seed,
    )
    model, coeffs = build_model(cfg)
    t0 = time.time()
    train(model, adam_iters=args.adam, lbfgs_iters=args.lbfgs,
          resample_period=200, display_every=max(100, args.adam // 20 or 100))
    train_s = time.time() - t0
    print(f"  trained in {train_s:.1f}s", flush=True)

    # --- compare scri waveform ----------------------------------------------
    sigma_scri = float(op.sigma[scri_idx])
    scri_pinn = scri_waveform(model, tau_fd, sigma_scri)
    scri_relL2 = rel_l2(scri_pinn, scri_fd)

    # full-field rel-L2 (diagnostic; gate is on the scri waveform)
    psi_pinn = predict_field(model, op.sigma, tau_fd)
    field_relL2 = rel_l2(psi_pinn, psi_fd)
    imag_frac = float(np.max(np.abs(scri_pinn.imag)) /
                      max(np.max(np.abs(scri_pinn.real)), 1e-300))

    # --- extract QNM from the PINN waveform (Phase B extractor) --------------
    sm = extract_fundamental(tau_fd, scri_pinn, is_real=is_real,
                             tau_ref=tau_ref, tau_final=T)
    mw_pinn = float(sm.get("omega", np.nan))
    tau_pinn = float(sm.get("tau", np.nan))
    e_omega = abs(mw_pinn - ref.M_omega_R) / ref.M_omega_R if np.isfinite(mw_pinn) else np.inf
    e_tau = abs(tau_pinn - tau_ref) / tau_ref if np.isfinite(tau_pinn) else np.inf

    passed = (scri_relL2 <= REL_L2_GATE) and (e_omega <= MW_GATE)

    print(f"\n  --- C.3 metrics (a/M={spin:.3f}) ---", flush=True)
    print(f"  scri rel-L2      = {scri_relL2:.4f}   (gate <= {REL_L2_GATE})", flush=True)
    print(f"  full-field rel-L2= {field_relL2:.4f}   (diagnostic)", flush=True)
    print(f"  M*omega (PINN)   = {mw_pinn:.6f}  vs Leaver {ref.M_omega_R:.6f} "
          f"-> err {e_omega:.2e}  (gate <= {MW_GATE})", flush=True)
    print(f"  tau/M   (PINN)   = {tau_pinn:.4f}  vs {tau_ref:.4f} "
          f"-> err {e_tau:.2e}  (report <= {TAU_REPORT})", flush=True)
    print(f"  imag/real (scri) = {imag_frac:.2e}  "
          f"({'should be ~0' if is_real else 'genuinely complex'})", flush=True)
    print(f"  ==> C.3 {'PASS' if passed else 'FAIL'} "
          f"(scri rel-L2 AND M*omega)\n", flush=True)

    result = {
        "task": "C.3_single_config_pinn",
        "spin": spin, "r0": args.r0, "w": args.w, "T": T, "is_real": is_real,
        "smoke": bool(args.smoke),
        "qnm_ref": {"M_omega_R": ref.M_omega_R, "M_omega_I": ref.M_omega_I,
                    "tau_over_M": tau_ref},
        "pinn": {"M_omega_R": mw_pinn, "tau_over_M": tau_pinn,
                 "imag_real_frac": imag_frac},
        "metrics": {"scri_rel_l2": scri_relL2, "field_rel_l2": field_relL2,
                    "e_omega": e_omega, "e_tau": e_tau},
        "gate": {"rel_l2_gate": REL_L2_GATE, "mw_gate": MW_GATE},
        "train": {"adam": args.adam, "lbfgs": args.lbfgs,
                  "num_domain": args.num_domain,
                  "use_fourier": bool(args.fourier),
                  "n_fourier": args.n_fourier if args.fourier else 0,
                  "hidden": list(args.hidden), "causal": False,
                  "train_seconds": train_s},
        "passed": bool(passed),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {args.out}", flush=True)

    sys.exit(0 if (passed or args.smoke) else 1)


if __name__ == "__main__":
    main()
