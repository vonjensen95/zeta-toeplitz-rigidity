# Keiper-Li coefficients: two independent methods

Both scripts compute λₙ for small n and are compared in Table 1 of the note.

- **`keiper_geometry.py`** — the shared coordinate geometry (Keiper's
  conformal map z_ρ = 1 - 1/ρ, the angle formula, the rapidity identity).
  Run directly to reproduce the identity checks of Section 2 of the note.

- **`method_a_cauchy.py`** — extracts λₙ as Taylor coefficients of
  d/dz log ξ(1/(1-z)) by discretized Cauchy integration. Uses **no zeros of
  zeta anywhere**; it is a completely independent route to the same numbers.

- **`method_b_zerosum.py`** — sums the angular contribution of the first
  300 known zeros (via Proposition 2.1) plus a smoothed-density tail
  correction for the remainder.

The two methods agree to 3-4 significant figures at small n, with a
systematic, monotonically growing difference (Table 1) attributable to the
tail approximation in Method B. Method A is the more accurate of the two at
this n-range but, like every derivative/coefficient-extraction method for
this sequence, is subject to the well-known cancellation wall of the
subject (~n bits of precision needed per coefficient at large n); see
Johansson (2015) for the rigorous large-n approach this does not attempt to
replace.

```bash
python3 keiper_geometry.py
python3 method_a_cauchy.py
python3 method_b_zerosum.py   # loads 300 zeros, takes ~2 min
```
