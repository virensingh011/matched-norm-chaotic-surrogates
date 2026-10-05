"""Fast correctness tests.  Run:  python -m pytest -q tests   (or: python tests/test_core.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jacsens import model as M
from jacsens.dynamics import lle_batch
from jacsens.perturb import (delta_of, make_perturbations, noise_iso, noise_layer, noise_support,
                             prune_global, prune_layer, quantize, rel_norm)
from jacsens.sensitivity import TrajectoryRef


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    p = M.init_params(rng)
    p["b1"] = rng.normal(size=64) * 0.1
    p["b3"] = rng.normal(size=3) * 0.1
    return rng, p


def test_backprop_matches_finite_differences():
    rng, p = _setup()
    X, Y = rng.normal(size=(32, 3)), rng.normal(size=(32, 3))
    _, g = M.loss_and_grads(p, X, Y)
    for k in M.KEYS:
        idx = tuple(rng.integers(0, s) for s in p[k].shape)
        q1, q2 = M.copy_params(p), M.copy_params(p)
        q1[k][idx] += 1e-6
        q2[k][idx] -= 1e-6
        fd = (M.loss_and_grads(q1, X, Y)[0] - M.loss_and_grads(q2, X, Y)[0]) / 2e-6
        assert abs(fd - g[k][idx]) < 1e-7, (k, fd, g[k][idx])


def test_jacobian_matches_finite_differences():
    rng, p = _setup()
    X = rng.normal(size=(10, 3))
    J = M.jac_points(p, X)
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-6
        fd = (M.forward(p, X + e) - M.forward(p, X - e)) / 2e-6
        assert np.abs(fd - J[:, :, j]).max() < 1e-7


def test_batched_step_and_jacobian_match_single():
    rng, p = _setup()
    q = noise_iso(p, 0.1, rng)
    P = M.stack_params([p, q])
    x = rng.normal(size=(2, 3))
    xn, J = M.step_and_jac_batch(P, x)
    for i, m in enumerate([p, q]):
        assert np.allclose(xn[i], M.forward(m, x[i:i + 1])[0])
        assert np.allclose(J[i], M.jac_points(m, x[i:i + 1])[0])
    S = rng.normal(size=(2, 5, 3))
    out = M.step_batch_states(P, S)
    for i, m in enumerate([p, q]):
        assert np.allclose(out[i], M.forward(m, S[i]))


def test_identity_residual_has_zero_lle():
    _, p = _setup()
    p["W3"][:] = 0
    p["b3"][:] = 0
    res = lle_batch(M.stack_params([p]), np.array([0.3, -0.2, 0.1]), 0.02,
                    n_transient=10, n_steps=500)
    assert abs(res["lle"][0]) < 1e-9


def test_matched_controls_have_exact_norms():
    rng, p = _setup()
    for q in (prune_global(p, 0.5), prune_layer(p, 0.5), quantize(p, 4)):
        rn, dl = rel_norm(p, q), delta_of(p, q)
        assert abs(rel_norm(p, noise_iso(p, rn, rng)) - rn) < 1e-12
        for ctrl in (noise_layer(p, dl, rng), noise_support(p, dl, rng)):
            assert abs(rel_norm(p, ctrl) - rn) < 1e-12
            for k in M.KEYS:
                assert abs(np.linalg.norm(ctrl[k] - p[k]) - np.linalg.norm(dl[k])) < 1e-12


def test_pruning_sparsity_and_support():
    _, p = _setup()
    q = prune_global(p, 0.5)
    w = np.concatenate([q[k].ravel() for k in M.WKEYS])
    assert abs((w == 0).mean() - 0.5) < 1e-3
    rng = np.random.default_rng(1)
    ctrl = noise_support(p, delta_of(p, q), rng)
    for k in M.WKEYS:                      # support control touches only pruned entries
        assert np.all((ctrl[k] != p[k]) <= (q[k] == 0))


def test_make_perturbations_counts_and_roles():
    rng, p = _setup()
    out = make_perturbations(p, rng, sparsities=(0.5,), bits=(8,), K=3, grid_norms=(0.1,))
    roles = [o["role"] for o in out]
    assert roles.count("primary") == 2 + 1 + 1                  # global, layer, quant, random
    assert roles.count("grid") == 3
    assert roles.count("control") == 3 * (3 + 3 + 2)             # K * (global:3 + layer:3 + quant:2)
    for o in out:
        if o["role"] == "control":
            prim = next(x for x in out if x["group"] == o["control_of"])
            assert abs(o["rel_norm"] - prim["rel_norm"]) < 1e-9


def test_s_lin_matches_frozen_orbit_for_tiny_perturbation():
    """The algebra check from sanity_check.py, miniaturised, on a random (not trained) map."""
    rng, p = _setup()
    p["W3"] *= 3.0
    T, burn, dt = 800, 100, 0.02
    X = rng.normal(size=(T, 3)) * 0.5                       # arbitrary 'orbit'; algebra is orbit-agnostic
    ref = TrajectoryRef(p, X, dt, burn=burn)
    q = noise_iso(p, 1e-5, rng)
    s_lin = ref.metrics(q)["S_lin"]

    def tang_rate(model):
        J = M.jac_points(model, X)
        u = np.random.default_rng(0).normal(size=3)
        u /= np.linalg.norm(u)
        logs = []
        for t in range(T):
            v = J[t] @ u
            n = np.linalg.norm(v)
            if t >= burn:
                logs.append(np.log(n))
            u = v / n
        return np.mean(logs) / dt

    exact = tang_rate(q) - tang_rate(p)
    assert abs(s_lin - exact) < 0.05 * abs(exact) + 1e-6, (s_lin, exact)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
