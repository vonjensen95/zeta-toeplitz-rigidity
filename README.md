# A Toeplitz positivity test on the angular distribution of zeta zeros

Code and paper for the note *"A Toeplitz positivity test on the angular
distribution of zeta zeros, with an independent reproduction of the
Connes-Consani-Moscovici spectral realization."*

## What is here

- **`paper/`** — the note itself (`.tex` source and compiled `.pdf`).
- **`lambda_n/`** — two independent computations of the Keiper-Li
  coefficients λₙ (Table 1 of the note), plus the shared geometry module
  and its identity checks (Propositions 2.1-2.2).
- **`detector/`** — the Toeplitz positivity detector, its controls, the
  empirical eigenvalue law, and the resolution table (Section 4 / Table 2).
- **`ccm_reproduction/`** — a full independent reproduction, from scratch,
  of the central numerical result of Connes, Consani and Moscovici,
  *"Zeta Spectral Triples"* (arXiv:2511.22755), at (λ, N) = (3, 120)
  (Section 5 / Table 3), including numerical verification of the two
  standing hypotheses of their Theorem 1.1 at that parameter point.

## Quick start

```bash
pip install mpmath numpy scipy
cd lambda_n   && python3 keiper_geometry.py && python3 method_a_cauchy.py && python3 method_b_zerosum.py
cd ../detector && python3 toeplitz_detector.py
cd ../ccm_reproduction && python3 build_matrix.py && python3 extract_nearkernel.py && python3 extract_zeros.py
```

Each script is self-contained, prints its own verification checks, and
takes under three minutes. `ccm_reproduction/extract_zeros.py` is the
slowest step (a few minutes) since it bisects twenty roots to ~200-digit
precision.

## Design principle: verify before trusting

Every closed-form special-function identity used anywhere in this
repository was checked against direct numerical quadrature or an
independent method before being relied on, and those checks are left in
the scripts (marked `[CHECK]`) rather than stripped out. This caught two
real bugs during development, both documented in `ccm_reproduction/`:

1. A diagonal-entry shift of exactly `2*c(L)` from a grouping ambiguity in
   the source paper's equation (4.4), which buries the near-kernel
   eigenvalue under a spurious constant floor.
2. A missing `1/sqrt(2)` basis-normalization factor when converting the
   orthonormal parity-basis eigenvector back to the paper's coordinates,
   which left every extracted root *exactly real* (the self-adjointness
   guarantee is structural and survives the bug) while destroying all
   numerical accuracy (errors of order 1 instead of order 1e-34). This
   split is recorded as Remark 5.1 of the note: reality of the spectrum is
   structural, accuracy of the approximation lives in the fine structure of
   the eigenvector.

If you adapt this code to a different (λ, N) or a different symbol, keep
the `[CHECK]` blocks and do not trust a formula that "looks right" without
running them.

## Scope

The Toeplitz detector is a numerical resolution study, not a method for
verifying the Riemann Hypothesis in any range; direct zero-location methods
are far more efficient for that. Its results are reported as measurements
with explicit controls, not as claims about RH itself. See the note's
Section 6 for what is and is not established, and for the planned
Davenport-Heilbronn validation (not yet run at the time of this release).

## Citation

If you use this code, please cite the note (see `paper/`) and, for the
reproduced material, Connes, Consani and Moscovici, *Zeta Spectral
Triples*, arXiv:2511.22755.

```
Von Jensen. "A Toeplitz positivity test on the angular distribution of
zeta zeros, with an independent reproduction of the Connes-Consani-
Moscovici spectral realization." 2026.
https://github.com/vonjensen95/zeta-toeplitz-rigidity
```

## Contact

Von Jensen — vonjensen95@gmail.com

## License

MIT. See `LICENSE`.
