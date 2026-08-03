# CCM reproduction

Independent, from-scratch reproduction of the central numerical computation
in Connes, Consani and Moscovici, *"Zeta Spectral Triples"* (arXiv:2511.22755),
at (λ, N) = (3, 120).

## Pipeline

```
build_matrix.py  -->  ab.pkl  -->  extract_nearkernel.py  -->  xi.pkl  -->  extract_zeros.py
```

1. **`build_matrix.py`** assembles the Weil-quadratic-form matrix in the
   Fourier basis, using the Loewner structure of their Lemma 5.1
   (τ_{n,m} = (b_n - b_m)/(n-m) for n≠m) so that only two length-121 real
   sequences a_n, b_n need to be computed rather than the full matrix. Every
   closed-form piece (Lemma 4.1, Proposition 4.2, Proposition 4.3) is
   checked against direct quadrature at the end of the script.

2. **`extract_nearkernel.py`** reduces the problem by parity into even and
   odd blocks, extracts the smallest even eigenpair by 200-digit LU-based
   inverse iteration, and verifies the two hypotheses of their Theorem 1.1
   (simplicity, evenness) as numerical facts at this parameter point.

3. **`extract_zeros.py`** builds the rational function S(z) proportional to
   the Fourier transform ξ̂(z) of the near-kernel vector, finds its real
   zeros by bracketing each pole gap and bisecting to full precision, and
   compares them against both the true Riemann zero ordinates
   (`mpmath.zetazero`) and the values published in Figure 1 of the paper.

## Results

All twenty published eigenvalue-zero discrepancies were reproduced to every
stated figure; see `results.md` for the full table and the verified
simplicity/evenness data (ε_N = 2.954058093×10⁻³⁸, gap ratio 1.4×10⁻⁷,
odd-sector floor four orders above ε_N).

## Two bugs, kept as documentation

Both are described in detail in the module docstrings and in the top-level
README's "verify before trusting" section. In short: a diagonal shift bug
(traceable to a grouping ambiguity in the source paper's eq. 4.4) produces a
spectrum with a spurious degenerate floor at exactly `2*c(L)`; a basis-weight
bug (a missing 1/√2 in the parity-basis conversion) produces roots that are
still exactly real but wrong by order 1. If you modify this code and results
degrade in either of these specific ways, these are the first two places to
check.

## A note on the source PDF

The closed-form expression for `c(L)` printed in our reading of the source
PDF does not evaluate to the same value as the integral it is meant to
represent; this is most likely a PDF-extraction artifact rather than an
error in the original paper. `build_matrix.py` uses the quadrature value of
`c(L)` throughout, which is what produces the matching results.
