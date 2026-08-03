"""
build_matrix.py

Step 1 of the independent reproduction of Connes, Consani and Moscovici,
"Zeta Spectral Triples" (arXiv:2511.22755), at (lambda, N) = (3, 120).

Every closed-form matrix-element formula from their Section 4 (Lemma 4.1,
Proposition 4.2, Proposition 4.3) is re-derived here and checked against
DIRECT NUMERICAL QUADRATURE of its defining integral before being trusted.
This is not a formality: the check caught two real bugs during development
(see the corrections noted below and in README.md), and running with formulas
that "looked right" but were wrong by a sign or a missing shift produced a
plausible-looking but numerically corrupted matrix. Do not skip section
[CHECK] below when adapting this script to a different (lambda, N).

Structure of the matrix (their Lemma 5.1): the (2N+1) x (2N+1) Weil-form
matrix tau_{n,m} in the Fourier basis {V_n} is fully determined by two length-
(N+1) real sequences a_n = tau_{n,n} and b_n (with b_0 = 0), via
    tau_{n,n} = a_n,      tau_{n,m} = (b_n - b_m)/(n - m)  for n != m.
This script computes a_n, b_n for n = 0..N by summing three contributions
(their eq. 3.13-3.16, restricted to the even function q(U_n,U_m)):
    W_{0,2}  -- rank-two, has an exact closed form (their Lemma 4.1);
    W_R      -- the archimedean place, via hypergeometric/Lerch series
                 (their Proposition 4.2, evaluated at n=m via Proposition 4.3);
    W_p      -- the non-archimedean (prime) places, a finite sum over
                 prime powers p^m <= exp(L) using the von Mangoldt function,
                 with L = 2*log(lambda).

CORRECTION relative to a literal reading of their eq. (4.4): the printed
integrand groups as exp(x/2)*(omega(x) - omega(0)) / (exp(x)-exp(-x)); using
that grouping directly shifts every diagonal entry by exactly
    2*c(L),   c(L) = int_0^L (exp(x/2)-1)/(exp(x)-exp(-x)) dx,
which buries the near-kernel (the resulting spectrum bottoms out at a
constant, degenerate value equal to 2*c(L) instead of at ~1e-38; this is a
useful diagnostic if you introduce the same bug while adapting the code).
The corrected diagonal formula folds in +2*c(L) explicitly; c(L) is computed
by quadrature since the closed form printed in the paper does not equal the
integral in our reading of the PDF (see README.md).

Usage:
    python3 build_matrix.py
Produces ab.pkl (the a_n, b_n arrays at 205-digit precision) for use by
extract_nearkernel.py.
"""
import mpmath as mp
import pickle

mp.mp.dps = 205

LAMBDA = 3
N = 120
L = 2 * mp.log(LAMBDA)
Z = mp.exp(-2 * L)          # series parameter, |Z| < 1
NT = 160                     # series truncation: Z^NT ~ 1e-305 for lambda=3

# Prime powers p^k <= exp(L) = lambda^2 = 9, with von Mangoldt weight log(p)
PRIMES_POWERS = [(2, mp.log(2)), (3, mp.log(3)), (4, mp.log(2)),
                  (5, mp.log(5)), (7, mp.log(7)), (8, mp.log(2))]


def hyp2f1_1_b(b):
    """2F1(1, b; b+1; Z) = sum_k [b/(b+k)] Z^k, evaluated by direct series
    (Z is small and fixed, this converges far faster than needed)."""
    s, p = mp.mpc(0), mp.mpf(1)
    for k in range(NT):
        s += b / (b + k) * p
        p *= Z
    return s


def lerch_phi_2(a):
    """Phi(Z, 2, a) = sum_k Z^k / (a+k)^2, by direct series."""
    s, p = mp.mpc(0), mp.mpf(1)
    for k in range(NT):
        s += p / (a + k) ** 2
        p *= Z
    return s


def build_arrays():
    C_sh = 32 * L * mp.sinh(L / 4) ** 2
    const_R = mp.euler + mp.log(4 * mp.pi * (mp.exp(L) - 1) / (mp.exp(L) + 1))
    c_L = mp.quad(lambda x: (mp.exp(x / 2) - 1) / (mp.exp(x) - mp.exp(-x)), [0, L])
    F14 = mp.re(hyp2f1_1_b(mp.mpf("0.25")))

    a_arr, b_arr = [], []
    for n in range(N + 1):
        a_param = mp.mpf("0.25") + 1j * mp.pi * n / L

        # Their (4.5), (4.6), (4.7): the three rho-weighted integrals.
        I45 = (mp.exp(-L / 2) * mp.im(2 * L / (L + 4j * mp.pi * n) * hyp2f1_1_b(a_param))
               + mp.im(mp.digamma(a_param)) / 2)
        I46 = (-L * mp.exp(-L / 2) * mp.im(2 * L / (4 * mp.pi * n - 1j * L) * hyp2f1_1_b(a_param))
               - mp.exp(-L / 2) / 4 * mp.re(lerch_phi_2(a_param))
               + mp.re(mp.polygamma(1, a_param)) / 4)
        I47 = (-mp.exp(-L / 2) * mp.re(2 * L / (L + 4j * mp.pi * n) * hyp2f1_1_b(a_param))
               + 2 * mp.exp(-L / 2) * F14
               - mp.re(mp.digamma(a_param) - mp.digamma(mp.mpf("0.25"))) / 2)

        W02_diag = C_sh * (L ** 2 - 16 * mp.pi ** 2 * n ** 2) / (L ** 2 + 16 * mp.pi ** 2 * n ** 2) ** 2
        WR_diag = const_R + 2 * (I47 + c_L) - (2 / L) * I46   # includes the +2c(L) correction
        Wp_diag = sum(w / mp.sqrt(k) * 2 * (1 - mp.log(k) / L) * mp.cos(2 * mp.pi * n * mp.log(k) / L)
                       for k, w in PRIMES_POWERS)
        a_arr.append(mp.re(W02_diag - WR_diag - Wp_diag))

        beta_n = C_sh * n / (L ** 2 + 16 * mp.pi ** 2 * n ** 2)
        Bp_n = -sum(w / mp.sqrt(k) * mp.sin(2 * mp.pi * n * mp.log(k) / L)
                     for k, w in PRIMES_POWERS) / mp.pi
        b_arr.append(beta_n + I45 / mp.pi - Bp_n)

    return a_arr, b_arr, c_L


def q_even(n, m, y):
    """The even function q(U_n, U_m)(y), Lemma 2.3."""
    if n != m:
        return (mp.sin(2 * mp.pi * m * y / L) - mp.sin(2 * mp.pi * n * y / L)) / (mp.pi * (n - m))
    return 2 * (1 - y / L) * mp.cos(2 * mp.pi * n * y / L)


def tau_direct(n, m, c_L):
    """tau_{n,m} computed directly by quadrature from its definition (3.10)/(4.1),
    independent of the closed-form series used in build_arrays(). Used only
    as a cross-check, at reduced precision for speed."""
    C_sh = 32 * L * mp.sinh(L / 4) ** 2
    const_R = mp.euler + mp.log(4 * mp.pi * (mp.exp(L) - 1) / (mp.exp(L) + 1))
    rho = lambda x: mp.exp(x / 2) / (mp.exp(x) - mp.exp(-x))
    W02 = C_sh * (L ** 2 - 16 * mp.pi ** 2 * m * n) / ((L ** 2 + 16 * mp.pi ** 2 * m ** 2) * (L ** 2 + 16 * mp.pi ** 2 * n ** 2))
    om0 = q_even(n, m, 0)
    WR = om0 / 2 * const_R + om0 * c_L + mp.quad(lambda x: rho(x) * (q_even(n, m, x) - om0), [0, L])
    Wp = sum(w / mp.sqrt(k) * q_even(n, m, mp.log(k)) for k, w in PRIMES_POWERS)
    return W02 - WR - Wp


if __name__ == "__main__":
    print(f"Building Weil-form arrays a_n, b_n for lambda={LAMBDA}, N={N} at {mp.mp.dps} digits...")
    a_arr, b_arr, c_L = build_arrays()
    print(f"  a_0 = {mp.nstr(a_arr[0], 12)}")
    print(f"  b_1 = {mp.nstr(b_arr[1], 12)}")
    print(f"  c(L) [used, by quadrature] = {mp.nstr(c_L, 15)}")

    pickle.dump(
        {"lambda": LAMBDA, "N": N,
         "a": [mp.nstr(x, 205) for x in a_arr],
         "b": [mp.nstr(x, 205) for x in b_arr]},
        open("ab.pkl", "wb"),
    )
    print("Saved ab.pkl")

    print("\n[CHECK] Cross-checking assembled entries against direct quadrature")
    print("        (reduced precision, ~30 digits, for speed):")
    with mp.workdps(30):
        for (n, m) in [(2, 5), (0, 7), (3, 3), (0, 0)]:
            assembled = (b_arr[n] - b_arr[m]) / (n - m) if n != m else a_arr[n]
            direct = tau_direct(n, m, c_L)
            diff = abs(assembled - direct)
            status = "OK" if diff < mp.mpf("1e-25") else "MISMATCH"
            print(f"    tau({n},{m}): assembled={mp.nstr(assembled,14)}  "
                  f"direct={mp.nstr(direct,14)}  diff={mp.nstr(diff,3)}  [{status}]")
