"""
test_claims.py

Fast regression test: checks a handful of the note's central numeric
claims against fresh computation, without re-running the full (several
minutes) pipelines. Intended as a quick sanity check after any edit to the
shared modules, not as a substitute for running the full scripts.

Usage:
    python3 tests/test_claims.py
Exits nonzero if any check fails.
"""
import sys
import os
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda_n"))
from keiper_geometry import verify_identities  # noqa: E402

mp.mp.dps = 30

FAILURES = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition:
        FAILURES.append(name)


def main():
    print("Checking Section 2 identities (50 zeros, reduced precision for speed)...")
    res = verify_identities(k=50, dps=25, verbose=False)
    check("Mobius identity z_rho = m(2rho-1) exact",
          res["mobius_identity_max_err"] < mp.mpf("1e-20"))
    check("unit circle |z_rho|=1 to < 1e-20",
          res["unit_circle_max_err"] < mp.mpf("1e-20"))
    check("angle formula matches arg(z_rho) to < 1e-20",
          res["angle_formula_max_err"] < mp.mpf("1e-20"))
    check("rapidity Im(w_rho) = pi/4 exact",
          res["rapidity_pi4_max_err"] < mp.mpf("1e-20"))

    print("\nChecking lambda_1 closed form (Method A calibration)...")
    from method_a_cauchy import compute_lambda  # noqa: E402
    lam = compute_lambda(n_max=1)
    closed = 1 + mp.euler / 2 - mp.log(4 * mp.pi) / 2
    check("lambda_1 matches 1 + gamma_E/2 - log(4pi)/2 to 1e-25",
          abs(lam[1] - closed) < mp.mpf("1e-25"))

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} CHECK(S) FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
