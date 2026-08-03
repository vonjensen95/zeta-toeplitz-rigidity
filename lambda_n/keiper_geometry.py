"""
keiper_geometry.py

Shared geometry for the Keiper disk coordinate z_rho = 1 - 1/rho.

Provides:
    load_zeros(k)      -- first k nontrivial zeta zeros via mpmath.zetazero
    zero_angle(gamma)   -- theta = pi - 2*arctan(2*gamma) for an on-line zero
    mobius(u)            -- m(u) = (u-1)/(u+1)
    verify_identities(k) -- numerical check of Propositions 2.1-2.2 of the note

These are the elementary restatements of Keiper's (1992) conformal map,
recorded here once and imported by both the lambda_n scripts and the
detector, rather than re-derived in each.
"""
import mpmath as mp


def mobius(u):
    return (u - 1) / (u + 1)


def zero_angle(gamma):
    """theta_gamma = pi - 2*arctan(2*gamma), the angle of z_rho on the unit
    circle for an on-line zero rho = 1/2 + i*gamma."""
    return mp.pi - 2 * mp.atan(2 * gamma)


def load_zeros(k, dps=30):
    """First k positive zero ordinates gamma_1 < gamma_2 < ... via mpmath."""
    with mp.workdps(dps):
        return [mp.im(mp.zetazero(n)) for n in range(1, k + 1)]


def verify_identities(k=50, dps=25, verbose=True):
    """Numerically check, on the first k zeros:
        (i)   z_rho = m(2*rho - 1) == 1 - 1/rho   [Prop 2.1 mechanism]
        (ii)  |z_rho| == 1                          [on the critical line]
        (iii) arg(z_rho) == zero_angle(gamma)        [Prop 2.1]
        (iv)  Im( (1/2) Log(2*rho-1) ) == pi/4        [Prop 2.2, rapidity]
    Returns a dict of max absolute errors.
    """
    with mp.workdps(dps):
        zeros = [mp.mpc(mp.mpf("0.5"), g) for g in load_zeros(k, dps=dps)]
        e_ident = max(abs(mobius(2 * z - 1) - (1 - 1 / z)) for z in zeros)
        e_circ = max(abs(abs(mobius(2 * z - 1)) - 1) for z in zeros)
        e_theta = max(
            abs(mp.arg(mobius(2 * z - 1)) - zero_angle(mp.im(z))) for z in zeros
        )
        e_rapidity = max(
            abs(mp.im(mp.log(2 * z - 1) / 2) - mp.pi / 4) for z in zeros
        )
    result = {
        "mobius_identity_max_err": e_ident,
        "unit_circle_max_err": e_circ,
        "angle_formula_max_err": e_theta,
        "rapidity_pi4_max_err": e_rapidity,
    }
    if verbose:
        print(f"Identity checks on the first {k} zeros (dps={dps}):")
        for name, val in result.items():
            print(f"  {name:28s} = {mp.nstr(val, 3)}")
    return result


if __name__ == "__main__":
    verify_identities()
