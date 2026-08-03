# The Toeplitz detector

`toeplitz_detector.py` reproduces Section 4 of the note in full: the
baseline near-singularity of the genuine zero-angle Toeplitz matrix, the
on-circle control (confirms the detector responds to off-circle
displacement specifically, not to injected mass generally), the empirical
eigenvalue law of Conjecture 4.1, the tolerance-dependence check, and the
resolution table (Table 2) comparing the matrix detection order N* against
Voros's scalar visibility threshold n ~ γ²/δ.

```bash
python3 toeplitz_detector.py   # ~2-3 minutes total
```

Runtime note: `detect()` uses exponential search (doubling) to bracket the
detection order before bisecting, rather than always starting from a fixed
large matrix size. This is purely a performance optimization and does not
change any result; a naive fixed-upper-bound bisection at N_MAX=6000
reproduces identical N* values but costs an O(N_MAX) eigendecomposition on
every single query, which is prohibitively slow when N* is small (as it is
for most of the table).

Historical note on the tolerance-sweep numbers: an early exploratory
version of this experiment (not shipped in this repo) used a detection
threshold of the raw form `-tol` rather than `-tol*c0` in one throwaway
control script, while the main resolution table always used the `-tol*c0`
scaled form shown here. Because the two scripts used a different threshold
convention, that exploratory script's `[2c]`-equivalent output does not
match this one (e.g. it gives N*=115 at tol=1e-3 where this script gives
N*=208): the two are answering "unscaled `|lambda_min| > tol`" versus
"`|lambda_min| > tol * c_0`" respectively, not the same question. This was
caught during an independent re-verification pass and traced by
reproducing the old script's exact (unscaled) comparison line by line; the
paper originally quoted the old, unscaled numbers by mistake and has been
corrected to match the numbers this script actually produces (208 to 82,
exponent 0.081 per decade). The scaled convention used here and throughout
the paper is the right one to use, since it makes the detection threshold
comparable across configurations with different baseline scale c_0; this
note exists only so that anyone who finds an old copy of the unscaled
script elsewhere isn't confused by the mismatch.
