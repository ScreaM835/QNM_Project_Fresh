# kerr/ — Kerr Teukolsky $(\ell=m=2)$ extension

Subfolder of `project32_qnm_pinn_improved`. **Self-contained: does not
import from the parent `src/`.** Reusable parent modules (M4 extractor,
plotting, FNO architectures) are *copied* into `kerr/src/` rather than
imported, so the parent paper's pipeline stays frozen.

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
- **Phase C.** Kerr-native neural surrogate. **Not** a straight port of the
  Schwarzschild hybrid; allowed to be a fundamentally different
  architecture using cutting-edge methodology for Kerr. Framing in paper:
  *"inspired by our Schwarzschild hybrid result, adapted and improved for
  Kerr."* Specific architecture chosen at start of Phase C using results
  from Phase B (e.g. which observable is hardest, where coarse-FD aliases,
  whether overtone separation needs explicit head, etc.).
- **Phase D.** Paper integration (new Kerr methods + results subsections in
  `../paper/`).

Out of scope: PINNs on Kerr, inverse PINN on Kerr, Bayesian posteriors,
detector data, multi-mode joint extraction. Listed as future work.

## Layout

```
kerr/
  src/
    teukolsky_fd.py        Phase A — to be written
    qnm_kerr_reference.py  Phase A — wraps the `qnm` PyPI package
    extractor_m4.py        copied from parent src/qnm.py (renamed to
                           kill the third-party `qnm` package shadow)
    plotting.py            copied from parent src/plotting.py
    fno_model.py           copied from parent src/fno_model.py
    hybrid_fno.py          copied from parent src/hybrid_fno.py
  scripts/
    validate_a0.py         Phase A acceptance: Teukolsky a=0 vs Zerilli
  configs/
    teukolsky_a0.yaml      Phase A canonical run
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
- **Phase C:** Kerr surrogate beats coarse-up baseline by $\ge 10\times$
  in field RMSD on the canonical Kerr evaluation set, OR suppresses
  worst-case $|\Delta\tau/\tau|$ on the population tail by at least the
  same factor the Schwarzschild hybrid did ($30\% \to 5\%$). One of the
  two must hold.

## Phase C abort policy

If Phase C fails its acceptance gate after one architecture has been
trained and evaluated: **ship Phase A+B only.** Kerr section becomes
"forward Teukolsky generalisation, validated against `qnm` across
$a/M \in [0, 0.95]$." No second-architecture rescue attempt within this
paper; deferred-architectures go to future work.
