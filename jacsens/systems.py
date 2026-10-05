"""Chaotic flows, RK4 integration and reference largest Lyapunov exponents.

Integration uses scalar-float loops (no per-step numpy overhead), which is
~10x faster than a vectorised RK4 for a single trajectory.

To add a system: write f(x, y, z) and its tangent action jvp(x, y, z, u, v, w)
(= Jacobian(x,y,z) @ (u,v,w)) and register it in SYSTEMS.
"""
import math

import numpy as np


def _lorenz_f(x, y, z):
    return 10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z


def _lorenz_jvp(x, y, z, u, v, w):
    return (-10.0 * u + 10.0 * v,
            (28.0 - z) * u - v - x * w,
            y * u + x * v - (8.0 / 3.0) * w)


def _rossler_f(x, y, z):
    return -y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)


def _rossler_jvp(x, y, z, u, v, w):
    return (-v - w, u + 0.2 * v, z * u + (x - 5.7) * w)


SYSTEMS = {
    "lorenz": dict(f=_lorenz_f, jvp=_lorenz_jvp, ic=(1.0, 1.0, 20.0), ic_scale=1.0,
                   dt=0.02, lle_literature=0.9056),
    "rossler": dict(f=_rossler_f, jvp=_rossler_jvp, ic=(1.0, 1.0, 0.5), ic_scale=0.5,
                    dt=0.02, lle_literature=0.0714),
}


def initial_condition(name, rng):
    s = SYSTEMS[name]
    return np.array(s["ic"]) + s["ic_scale"] * rng.normal(size=3)


def integrate(name, s0, n_samples, dt_sample=0.02, h=0.001, transient=20.0):
    """RK4 trajectory sampled every dt_sample after discarding `transient` time units.

    Returns an array of shape (n_samples, 3) of *consecutive* samples.
    """
    f = SYSTEMS[name]["f"]
    nsub = int(round(dt_sample / h))
    n_tr = int(round(transient / dt_sample))
    x, y, z = (float(c) for c in s0)
    h2, h6 = 0.5 * h, h / 6.0
    out = np.empty((n_samples, 3))
    for i in range(n_tr + n_samples):
        for _ in range(nsub):
            k1 = f(x, y, z)
            k2 = f(x + h2 * k1[0], y + h2 * k1[1], z + h2 * k1[2])
            k3 = f(x + h2 * k2[0], y + h2 * k2[1], z + h2 * k2[2])
            k4 = f(x + h * k3[0], y + h * k3[1], z + h * k3[2])
            x += h6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y += h6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            z += h6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if i >= n_tr:
            out[i - n_tr] = (x, y, z)
    return out


def reference_lle(name, T=1000.0, h=0.005, renorm_every=4, transient=50.0, seed=0):
    """Benettin two-trajectory-free (tangent-vector) estimate of the largest LE, per unit time."""
    sp = SYSTEMS[name]
    f, jvp = sp["f"], sp["jvp"]
    rng = np.random.default_rng(seed)
    x, y, z = integrate(name, initial_condition(name, rng), 1, h=0.001, transient=transient)[0]
    u, v, w = 1.0, 0.0, 0.0
    h2, h6 = 0.5 * h, h / 6.0

    def faug(x, y, z, u, v, w):
        a = f(x, y, z)
        b = jvp(x, y, z, u, v, w)
        return a + b

    acc, n = 0.0, int(round(T / h))
    for i in range(n):
        k1 = faug(x, y, z, u, v, w)
        k2 = faug(*(c + h2 * k for c, k in zip((x, y, z, u, v, w), k1)))
        k3 = faug(*(c + h2 * k for c, k in zip((x, y, z, u, v, w), k2)))
        k4 = faug(*(c + h * k for c, k in zip((x, y, z, u, v, w), k3)))
        x += h6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y += h6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += h6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        u += h6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        v += h6 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        w += h6 * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])
        if (i + 1) % renorm_every == 0:
            nrm = math.sqrt(u * u + v * v + w * w)
            acc += math.log(nrm)
            u, v, w = u / nrm, v / nrm, w / nrm
    return acc / T
