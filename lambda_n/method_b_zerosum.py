"""
method_b_zerosum.py

Keiper-Li coefficients lambda_n by direct zero summation with a smoothed
tail correction, using Proposition 2.1 of the note:

    lambda_n = sum_{0 < gamma <= Gamma} 4 sin^2(n*theta_gamma / 2) + tail(n)

where theta_gamma = pi - 2*arctan(2*gamma) is the angle of the zero's image
under the Keiper map z_rho = 1 - 1/rho (see keiper_geometry.py).

The tail beyond the K-th zero is estimated by integrating the pair
contribution 2*Re[1 - (1-1/rho)^n] against the smoothed zero density
(1/(2*pi)) * log(gamma / (2*pi)).

This is independent of method_a_cauchy.py (which uses no zeros at all) and
serves as a cross-check. See Table 1 of the note for the comparison; the
difference between the two methods carries a single sign and grows smoothly
with n, consistent with it being the deterministic bias of the tail
approximation beyond the K-th zero rather than numerical error in either
method.

Usage:
    python3 method_b_zerosum.py
"""
import mpmath as mp
from keiper_geometry import load_zeros, zero_angle

mp.mp.dps = 25


def compute_lambda(n, gammas, T_max):
    thetas = [zero_angle(g) for g in gammas]
    main = 4 * mp.fsum(mp.sin(n * th / 2) ** 2 for th in thetas)

    def pair_term(gamma):
        rho = mp.mpf("0.5") + 1j * gamma
        return 2 * mp.re(1 - (1 - 1 / rho) ** n)

    density = lambda g: mp.log(g / (2 * mp.pi)) / (2 * mp.pi)
    tail = mp.quad(lambda g: pair_term(g) * density(g),
                    [T_max, 10 * T_max, 1000 * T_max, mp.inf])
    return main + tail


if __name__ == "__main__":
    K = 300
    print(f"Loading first {K} zero ordinates (this takes a moment)...")
    gammas = load_zeros(K, dps=25)
    T_max = gammas[-1]

    print(f"\nlambda_n via zero sum + tail correction (K={K}, T_max={mp.nstr(T_max, 6)}):")
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 48]:
        val = compute_lambda(n, gammas, T_max)
        print(f"  lambda_{n:2d} = {mp.nstr(val, 15)}")
