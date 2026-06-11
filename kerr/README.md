our# kerr/ — Kerr Teukolsky $(\ell=m=2)$ extension

Subfolder of `project32_qnm_pinn_improved`. **Self-contained: does not
import from the parent `src/`.** Reusable parent modules (M4 extractor,
plotting, the Schwarzschild forward-PINN core) are *copied* into `kerr/src/`
rather than imported, so the parent paper's pipeline stays frozen.

## Scope

- **Phase A.** Time-domain Teukolsky $(s=-2,\ell=m=2)$ FD core on hyperboloidal
  slicing, **hand-coded** (RK4 method-of-lines, ~500 LoC). No wrapping of
  external time-domain Teukolsky libraries; the integrator is the research
  artefact and must live here. Validated at $a/M = 0$ against the parent
  Schwarzschild Zerilli result.
- **Phase B.** Forward FD waveforms across the full sweep
  $a/M \in [0, 0.95]$, with M4 plateau extraction validated against the
  `qnm` (PyPI) continued-fraction package. **Reported in paper body at
  three canonical spins $\{0, 0.5, 0.9\}$;** the full sweep
  $\omega_{220}(a)$ curve and table go in an appendix.
- **Phase C.** Kerr-native neural waveform solver: a **parametric
  physics-informed neural network (PINN)** that solves the time-domain
  Teukolsky equation directly (PDE residual + analytic initial data;
  **no FD field in the loss**), conditioned on $(a/M, r_0, w)$. QNMs are
  then extracted from the PINN waveform by the validated Phase B M4
  extractor. Pivoted here from the originally-planned coarse-FD +
  FNO-residual hybrid (see `kerr/notes/phase_c_plan.md` §C.0′ for the
  rationale and the derived PDE). Framing in paper: *"inspired by our
  Schwarzschild forward PINN, adapted and improved for Kerr"* — and the
  first rung of the higher-dimensional PDE-solver programme.
- **Phase D.** Paper integration (new Kerr methods + results subsections in
  `../paper/`).

Out of scope: inverse PINN on Kerr (parameter recovery from data), Bayesian
posteriors, detector data, multi-mode joint extraction. Listed as future
work. **Forward PINNs on Kerr are now in scope** as the Phase C method (see
above); this supersedes the earlier listing of "PINNs on Kerr" as future
work, per the Phase C pivot (`kerr/notes/phase_c_plan.md` §C.0′).

## Layout

```
kerr/
  src/
    teukolsky_fd.py        Phase A — to be written
    qnm_kerr_reference.py  Phase A — wraps the `qnm` PyPI package
    extractor_m4.py        copied from parent src/qnm.py (renamed to
                           kill the third-party `qnm` package shadow)
    plotting.py            copied from parent src/plotting.py
    pinn.py                Phase C — to copy from parent src/pinn.py (SW
                           forward-PINN core; basis for the Kerr PINN)
    teukolsky_residual.py  Phase C — torch Teukolsky PDE residual (to write)
    kerr_pinn.py           Phase C — parametric PINN + train loop (to write)
    fno_model.py           hybrid-era copy; superseded by PINN pivot, kept
                           for provenance
    hybrid_fno.py          hybrid-era copy; superseded by PINN pivot, kept
                           for provenance
  scripts/
    validate_a0.py         Phase A acceptance: Teukolsky a=0 vs Zerilli
    train_kerr_pinn.py     Phase C — PINN training (to write)
    eval_kerr_pinn.py      Phase C — PINN vs FD vs Leaver (to write)
  configs/
    teukolsky_a0.yaml      Phase A canonical run
    kerr_pinn.yaml         Phase C — PINN config (to write)
  outputs/                  gitignored; large artefacts
```

Run scripts from inside `kerr/`:
```
cd kerr
python scripts/validate_a0.py
```

## Acceptance gates (verbatim, do not soften)

- **Phase A:** $M\omega_{220}$ at $a=0$ within $0.1\%$ of Leaver and within
  $0.05\%$ of the parent paper's Zerilli M4 result; integrator stable to
  $t=200M$; no boundary growth.
- **Phase B:** $M\omega_{220}$ error against `qnm` package $\le 0.1\%$ at
  all three reported spins $\{0, 0.5, 0.9\}$; first overtone $n=1$
  resolved at $a/M = 0.9$; population-mean error across the full
  $a/M \in [0, 0.95]$ sweep $\le 0.2\%$.
- **Phase C:** on **held-out spins** (test split, unseen in training), the
  PINN waveform matches the fine FD field to relative $L^2 \le 5\%$, **and**
  the QNM extracted from the PINN waveform matches **Leaver** ($M\omega_{220}$
  within $1\%$, damping time $\tau$ within $5\%$). **Both** must hold.
  *(Replaces the hybrid-era "$\ge 10\times$ coarse-up baseline" gate, which
  became void once the coarse-FD baseline was dropped in the PINN pivot —
  see `kerr/notes/phase_c_plan.md` §C.0′. This is a re-anchoring to the
  oracles we already trust (fine FD + Leaver), not a softening.)*

## Phase C abort policy

If Phase C fails its acceptance gate after one architecture has been
trained and evaluated: **ship Phase A+B only.** Kerr section becomes
"forward Teukolsky generalisation, validated against `qnm` across
$a/M \in [0, 0.95]$." No second-architecture rescue attempt within this
paper; deferred-architectures go to future work.
