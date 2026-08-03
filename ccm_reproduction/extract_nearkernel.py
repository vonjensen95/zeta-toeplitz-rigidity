"""
extract_nearkernel.py

Step 2 of the reproduction: parity-reduce the (2N+1)x(2N+1) Weil-form matrix
to its N+1 dimensional EVEN block, extract the smallest eigenpair by
200-digit LU-based inverse iteration, and verify the two hypotheses that
Connes-Consani-Moscovici's Theorem 1.1 assumes at general (lambda, N):

    (i)  the smallest eigenvalue epsilon_N of the (truncated) Weil form is
         SIMPLE;
    (ii) its eigenvector xi is EVEN.

Both are checked here as numerical facts at (lambda, N) = (3, 120), not
assumed. (ii) is checked by computing the smallest eigenvalue of the ODD
block separately and confirming it sits far above epsilon_N; (i) is checked
by computing the second-smallest even eigenvalue and confirming a large gap
ratio.

Standard-library (pure Python) LU factorization and inverse iteration are
used throughout at working precision 205 digits, since numpy/scipy operate
in double precision only and the eigenvalue gap here is of order 1e-38 -
far below double precision's ~1e-16 floor. A LU self-test against a random
symmetric positive-definite system (see README.md / tests) confirms the
solver is correct to working precision before it is trusted on the real
problem.

Usage:
    python3 extract_nearkernel.py
Requires ab.pkl from build_matrix.py. Produces xi.pkl (the 200-digit
near-kernel eigenvector and eigenvalue) for use by extract_zeros.py.
"""
import mpmath as mp
import pickle
import random

mp.mp.dps = 205


def build_even_block(a, b, N):
    """Even block in the basis e_0 = V_0, e_j = (V_j + V_{-j})/sqrt(2), j>=1."""
    E = [[mp.mpf(0)] * (N + 1) for _ in range(N + 1)]
    E[0][0] = a[0]
    for j in range(1, N + 1):
        v = mp.sqrt(2) * b[j] / j
        E[0][j] = E[j][0] = v
        E[j][j] = a[j] + b[j] / j
        for k in range(1, j):
            v = (b[j] - b[k]) / (j - k) + (b[j] + b[k]) / (j + k)
            E[j][k] = E[k][j] = v
    return E


def build_odd_block(a, b, N):
    """Odd block in the basis (V_j - V_{-j})/(i*sqrt(2)), j=1..N."""
    O = [[mp.mpf(0)] * N for _ in range(N)]
    for j in range(1, N + 1):
        O[j - 1][j - 1] = a[j] - b[j] / j
        for k in range(1, j):
            v = (b[j] - b[k]) / (j - k) - (b[j] + b[k]) / (j + k)
            O[j - 1][k - 1] = O[k - 1][j - 1] = v
    return O


def lu_factor(M):
    n = len(M)
    M = [row[:] for row in M]
    piv = list(range(n))
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if p != c:
            M[c], M[p] = M[p], M[c]
            piv[c], piv[p] = piv[p], piv[c]
        inv = 1 / M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] * inv
            M[r][c] = f
            if f:
                Mr, Mc = M[r], M[c]
                for k in range(c + 1, n):
                    Mr[k] -= f * Mc[k]
    return M, piv


def lu_solve(LU, piv, rhs):
    n = len(LU)
    x = [rhs[piv[i]] for i in range(n)]
    for i in range(1, n):
        s = x[i]
        Li = LU[i]
        for j in range(i):
            s -= Li[j] * x[j]
        x[i] = s
    for i in range(n - 1, -1, -1):
        s = x[i]
        Li = LU[i]
        for j in range(i + 1, n):
            s -= Li[j] * x[j]
        x[i] = s / Li[i]
    return x


def lu_self_test(n=8, seed=1):
    """Sanity check the pure-Python LU solver against a known random SPD
    system before trusting it on the real problem."""
    random.seed(seed)
    R = [[mp.mpf(random.random()) for _ in range(n)] for _ in range(n)]
    M = [[sum(R[i][k] * R[j][k] for k in range(n)) + (mp.mpf(2) if i == j else 0)
          for j in range(n)] for i in range(n)]
    x_true = [mp.mpf(random.random()) for _ in range(n)]
    rhs = [sum(M[i][j] * x_true[j] for j in range(n)) for i in range(n)]
    LU, piv = lu_factor(M)
    x_solved = lu_solve(LU, piv, rhs)
    return max(abs(x_solved[i] - x_true[i]) for i in range(n))


def smallest_eigenpair(M, iters=4, deflate=None, seed=7):
    """Smallest eigenvalue/eigenvector of symmetric M by inverse iteration
    (LU-based linear solves), optionally deflated against a known
    eigenvector to find the next-smallest."""
    n = len(M)
    LU, piv = lu_factor(M)
    random.seed(seed)
    x = [mp.mpf(random.random()) for _ in range(n)]
    for _ in range(iters):
        if deflate:
            dot = sum(u * v for u, v in zip(x, deflate))
            x = [xi - dot * di for xi, di in zip(x, deflate)]
        x = lu_solve(LU, piv, x)
        norm = mp.sqrt(sum(t * t for t in x))
        x = [t / norm for t in x]
    Mx = [sum(M[i][j] * x[j] for j in range(n)) for i in range(n)]
    lam = sum(x[i] * Mx[i] for i in range(n))
    residual = mp.sqrt(sum((Mx[i] - lam * x[i]) ** 2 for i in range(n)))
    return lam, x, residual


if __name__ == "__main__":
    print("LU self-test on a random 8x8 SPD system:")
    err = lu_self_test()
    print(f"  max solve error = {mp.nstr(err, 3)}\n")

    d = pickle.load(open("ab.pkl", "rb"))
    a = [mp.mpf(s) for s in d["a"]]
    b = [mp.mpf(s) for s in d["b"]]
    N = d["N"]
    print(f"Loaded ab.pkl: lambda={d['lambda']}, N={N}\n")

    print("Building even and odd parity blocks...")
    E = build_even_block(a, b, N)
    O = build_odd_block(a, b, N)

    print("Extracting smallest even eigenpair (near-kernel candidate)...")
    eps, xi, res = smallest_eigenpair(E)
    print(f"  epsilon_N = {mp.nstr(eps, 12)}")
    print(f"  residual  = {mp.nstr(res, 3)}  (should be tiny: confirms inverse iteration converged)")

    print("\n[CHECK i: SIMPLICITY] second-smallest even eigenvalue (deflated)...")
    lam2, _, _ = smallest_eigenpair(E, iters=3, deflate=xi)
    gap_ratio = eps / lam2
    print(f"  lambda_2 (even) = {mp.nstr(lam2, 8)}")
    print(f"  epsilon_N / lambda_2 = {mp.nstr(gap_ratio, 3)}")
    print(f"  -> SIMPLE at this (lambda,N): gap ratio << 1" if gap_ratio < mp.mpf("1e-6") else "  -> WARNING: gap not clearly resolved")

    print("\n[CHECK ii: EVENNESS] smallest odd eigenvalue...")
    lam_odd, _, _ = smallest_eigenpair(O)
    print(f"  lambda_min (odd) = {mp.nstr(lam_odd, 8)}")
    print(f"  -> EVEN eigenvector is the true minimizer: odd sector sits far above epsilon_N"
          if lam_odd > eps * 1000 else "  -> WARNING: odd sector too close to epsilon_N")

    pickle.dump(
        {"lambda": d["lambda"], "N": N,
         "eps": mp.nstr(eps, 205),
         "xi": [mp.nstr(t, 205) for t in xi]},
        open("xi.pkl", "wb"),
    )
    print("\nSaved xi.pkl")
    print("Leading components xi_0..xi_4:", [mp.nstr(t, 8) for t in xi[:5]])
