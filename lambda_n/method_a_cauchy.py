"""
method_a_cauchy.py

Keiper-Li coefficients lambda_n via zero-free Cauchy extraction.

Computes lambda_n as the n-th Taylor coefficient of
    f(z) = d/dz log xi(1/(1-z))
by discretized contour integration on |z| = r. No zeros of zeta are used
anywhere in this computation; it is an independent check on any zero-sum
method (see method_b_zerosum.py) and on published high-n computations
(e.g. Johansson 2015).

    xi'/xi(s) = 1/s + 1/(s-1) - (1/2) log(pi) + (1/2) psi(s/2) + zeta'(s)/zeta(s)

Radius: r = 0.75. At r = 0.9 an aliasing floor of order 1e-9 was observed
(from lambda_{n+256} r^256 folding into the extracted coefficients); at
r = 0.75 the extracted lambda_1 matches the closed form
    lambda_1 = 1 + gamma_E/2 - (1/2) log(4 pi)
to all 15 printed digits. This is the calibration check run at the bottom
of this file.

Precision caution: at FIXED radius r, coefficient extraction amplifies
error by r^{-n}. This method is accurate for the range tested (n <= 48)
at 40-digit working precision; it is not a substitute for the rigorous
n-bit-precision methods needed at large n (Johansson 2015, using Arb).

Usage:
    python3 method_a_cauchy.py
"""
import mpmath as mp

mp.mp.dps = 40


def lnxi_prime(s):
    return (
        1 / s
        + 1 / (s - 1)
        - mp.log(mp.pi) / 2
        + mp.digamma(s / 2) / 2
        + mp.zeta(s, derivative=1) / mp.zeta(s)
    )


def compute_lambda(n_max=48, n_nodes=256, radius="0.75"):
    """Return {n: lambda_n} for n = 1..n_max via Cauchy extraction."""
    r = mp.mpf(radius)
    vals = []
    for k in range(n_nodes):
        z = r * mp.expjpi(2 * mp.mpf(k) / n_nodes)
        s = 1 / (1 - z)
        vals.append(lnxi_prime(s) / (1 - z) ** 2)

    lam = {}
    for n in range(n_max):
        acc = mp.mpc(0)
        for k in range(n_nodes):
            acc += vals[k] * mp.expjpi(-2 * mp.mpf(k) * n / n_nodes)
        lam[n + 1] = mp.re(acc / (n_nodes * r**n))
    return lam


if __name__ == "__main__":
    lam = compute_lambda()

    closed_form_lambda_1 = 1 + mp.euler / 2 - mp.log(4 * mp.pi) / 2
    print("Calibration: lambda_1 vs closed form 1 + gamma_E/2 - log(4pi)/2")
    print("  computed:", mp.nstr(lam[1], 15))
    print("  closed  :", mp.nstr(closed_form_lambda_1, 15))
    print("  |diff|  :", mp.nstr(abs(lam[1] - closed_form_lambda_1), 3))
    print()
    print("lambda_n for selected n:")
    for n in [1, 2, 3, 4, 5, 8, 13, 21, 34, 48]:
        print(f"  lambda_{n:2d} = {mp.nstr(lam[n], 15)}")
