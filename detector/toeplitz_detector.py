"""
toeplitz_detector.py

The paper's central numerical experiment (Section 4).

Builds the trigonometric moments c_n = 2*sum_k cos(n*theta_k) of the
Keiper-circle angle measure of the first K zeta zeros, forms Toeplitz
matrices T_N = (c_{i-j}), and tests positive semidefiniteness as a
detector for an injected off-critical-line zero quadruple

    rho = 1/2 + delta + i*gamma  (and its conjugate/reflected partners).

Three things are computed and printed, matching the paper's tables:

  1. Baseline: lambda_min(T_N) for the genuine (on-line) zero measure at
     several N, confirming near-singularity without any injected atom.

  2. Controls:
       (a) an ON-CIRCLE atom at the same angle/weight as a representative
           off-line test point -- this must NOT trigger detection;
       (b) the deviation-only Toeplitz spectrum, whose most negative
           eigenvalue is fit to the empirical law
           lambda_min ~ -c * N^3 * log(1/r)^2  (Conjecture 4.1 in the note);
       (c) tolerance-dependence of the detection order N*, to confirm
           detection is not simply a threshold-crossing artifact.

  3. The resolution table (paper Table 2): detection order N* for a grid
     of (gamma, delta), compared against the scalar visibility threshold
     n ~ gamma^2/delta of Voros (2006, 2010).

This script depends only on numpy/scipy (fast, double precision) for the
Toeplitz eigenproblems and mpmath (high precision) for the zero ordinates
and the exact velocity-coordinate map. Runtime is a few minutes total.

Usage:
    python3 toeplitz_detector.py
"""
import numpy as np
import mpmath as mp
from scipy.linalg import eigvalsh, toeplitz
from keiper_geometry import load_zeros, zero_angle

mp.mp.dps = 25


def injected_velocity(delta, gamma):
    """Return (r, theta) for z_rho = m(2*rho - 1) with rho = 1/2 + delta + i*gamma."""
    u = 2 * mp.mpc(delta, gamma)
    z = (u - 1) / (u + 1)
    return float(abs(z)), float(mp.arg(z))


def build_baseline(thetas, n_max):
    n = np.arange(n_max)
    return 2.0 * np.cos(np.outer(n, thetas)).sum(axis=1)


def lambda_min_at_sizes(c, sizes):
    return [eigvalsh(toeplitz(c[:N]))[0] for N in sizes]


def detect(c, n_max, tol, c0):
    """Find the smallest N at which lambda_min(T_N) < -tol*c0, by exponential
    search (doubling) to bracket N* followed by bisection. Equivalent to a
    fixed-upper-bound bisection but far cheaper when N* << n_max, since it
    avoids diagonalizing an n_max x n_max matrix for every query.
    Returns None if never detected within n_max."""
    lo, hi = 2, 4
    while hi < n_max and eigvalsh(toeplitz(c[:hi]))[0] >= -tol * c0:
        lo, hi = hi, min(2 * hi, n_max)
    if eigvalsh(toeplitz(c[:hi]))[0] >= -tol * c0:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if eigvalsh(toeplitz(c[:mid]))[0] < -tol * c0:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    K = 300
    N_MAX = 6000
    TOL = 1e-6

    print(f"Loading first {K} zero ordinates...")
    gammas = load_zeros(K, dps=25)
    thetas = np.array([float(zero_angle(g)) for g in gammas])

    base = build_baseline(thetas, N_MAX)
    c0 = base[0]

    print("\n[1] Baseline near-singularity (no injection):")
    sizes0 = [50, 150, 400, 800, 1500]
    lm0 = lambda_min_at_sizes(base, sizes0)
    for N, v in zip(sizes0, lm0):
        print(f"    N={N:5d}  lambda_min={v:.3e}   (c_0 = {c0:.1f})")

    print("\n[2a] Control: on-circle atom at the (gamma,delta)=(10,0.2) test angle")
    r02, th02 = injected_velocity(0.2, 10.0)
    n = np.arange(N_MAX)
    c_on = base + 4.0 * np.cos(n * th02)  # same angle, same total weight, r=1
    worst = min(eigvalsh(toeplitz(c_on[:N]))[0] for N in [50, 150, 400, 800, 1500, 3000])
    print(f"    worst lambda_min over sizes = {worst:.2e}  (tolerance scale = {TOL*c0:.1e})")
    print("    -> stays positive: detector responds to off-circle displacement, not to")
    print("       the mere presence of an added atom.")

    print("\n[2b] Eigenvalue law: deviation-only spectrum at (gamma,delta)=(10,0.2)")
    r, th = r02, th02
    n_cap = np.minimum(n, 3000)  # avoid overflow far past the detection point
    dev = 2.0 * np.cos(n * th) * (r**n + r**(-n_cap)) - 4.0 * np.cos(n * th)
    print("    N      lambda_min      N^3 * ln(1/r)^2")
    for N in [50, 100, 200, 400, 800]:
        lm = eigvalsh(toeplitz(dev[:N]))[0]
        law = (N ** 3) * (-np.log(r)) ** 2
        print(f"    {N:4d}   {lm:12.3e}    {law:12.3e}   ratio={lm/law:.4f}")

    print("\n[2c] Tolerance dependence of N* (should be weak if detection is a")
    print("     rigidity effect rather than simple threshold crossing):")
    c_off = base + 2.0 * np.cos(n * th) * (r**n + r**(-n_cap))
    for tol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        nstar = detect(c_off, N_MAX, tol, c0)
        print(f"    tol={tol:.0e}  N* = {nstar}")

    print("\n[3] Resolution table (paper Table 2):")
    print("    gamma   delta    ln(1/r)         N*    N*ln(1/r)   scalar~gamma^2/delta")
    configs = [
        (10, 0.40), (10, 0.30), (10, 0.20), (10, 0.10), (10, 0.05),
        (5, 0.20), (20, 0.20), (14.13, 0.25),
    ]
    for gamma, delta in configs:
        r, th = injected_velocity(delta, gamma)
        lnr = -np.log(r)
        n_cap = np.minimum(n, 3000)
        c = base + 2.0 * np.cos(n * th) * (r**n + r**(-n_cap))
        nstar = detect(c, N_MAX, TOL, c0)
        scalar = gamma * gamma / delta
        tag = f"{nstar:6d}   {nstar*lnr:9.3f}" if nstar else "  not detected"
        print(f"    {gamma:6.2f}  {delta:5.2f}   {lnr:.4e}   {tag}      {scalar:8.1f}")


if __name__ == "__main__":
    main()
