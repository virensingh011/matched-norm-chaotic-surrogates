"""Sanity checks for the Jacobian machinery and the first-order LLE prediction S_lin.

    python sanity_check.py [--system lorenz] [--epochs 50]

1. Analytic Jacobian vs central finite differences.
2. FROZEN-ORBIT test (algebra check, no statistics): for tiny random perturbations, compare
   S_lin with the exact change of the orbit-averaged tangent growth rate computed with the
   perturbed Jacobians along the SAME true orbit (tangent vector re-propagated by the
   perturbed map). These must agree at first order; if not, the formula/code is wrong.
3. OWN-ORBIT test: compare S_lin with the actual change of the LLE of the perturbed map
   (its own attractor, long rollout). Disagreement here is a genuine limit of the
   first-order/frozen-orbit approximation (or estimator noise), not a bug.
"""
import argparse

import numpy as np

from jacsens import model as M
from jacsens.dynamics import lle_batch
from jacsens.perturb import noise_iso
from jacsens.sensitivity import TrajectoryRef, reference_orbit
from jacsens.systems import SYSTEMS, initial_condition, integrate


def frozen_orbit_lle(pert, X, dt, burn, seed=0):
    J = M.jac_points(pert, X)
    u = np.random.default_rng(seed).normal(size=X.shape[1])
    u /= np.linalg.norm(u)
    logs = []
    for t in range(len(X)):
        v = J[t] @ u
        n = np.linalg.norm(v)
        if t >= burn:
            logs.append(np.log(n))
        u = v / n
    return np.mean(logs) / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="lorenz")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--n_steps", type=int, default=20000)
    args = ap.parse_args()
    dt = SYSTEMS[args.system]["dt"]
    rng = np.random.default_rng(1)

    data = integrate(args.system, initial_condition(args.system, rng), 25002, dt_sample=dt)
    mu, sd = data[:20001].mean(0), data[:20001].std(0)
    Z = (data - mu) / sd
    Xtr, Ytr = Z[:20000], Z[1:20001]
    Zte = Z[-5001:]
    parent = M.train(M.init_params(rng), Xtr, Ytr, epochs=args.epochs, rng=rng)
    print(f"parent test MSE = {M.mse(parent, Zte[:-1], Zte[1:]):.2e}")

    # 1. Jacobian vs finite differences
    X = Zte[:50]
    J = M.jac_points(parent, X)
    eps, err = 1e-6, 0.0
    for j in range(3):
        e = np.zeros(3); e[j] = eps
        fd = (M.forward(parent, X + e) - M.forward(parent, X - e)) / (2 * eps)
        err = max(err, np.abs(fd - J[:, :, j]).max())
    print(f"[1] max |analytic - finite-diff| Jacobian entry = {err:.2e}  (expect < 1e-7)")

    # 2/3. tiny-perturbation tests
    T, burn = 3000, 300
    orbit, kind = reference_orbit(parent, Zte, T)
    print(f"    reference orbit: {kind}")
    ref = TrajectoryRef(parent, orbit, dt, burn=burn)
    lam0_frozen = frozen_orbit_lle(parent, orbit, dt, burn)
    print(f"    parent tangent growth along reference orbit = {lam0_frozen:.3f}")
    norms = [1e-4, 1e-3, 3e-3, 1e-2]
    perts, tags = [], []
    for rn in norms:
        for _ in range(args.draws):
            perts.append(noise_iso(parent, rn, rng)); tags.append(rn)
    tags = np.array(tags)
    mm = ref.metrics_batch(perts)
    S_lin = np.array([m["S_lin"] for m in mm])
    S_lin0 = np.array([m["S_lin0"] for m in mm])
    d_frozen = np.array([frozen_orbit_lle(p, orbit, dt, burn) - lam0_frozen for p in perts])

    print("\n[2] frozen-orbit test (algebra check): expect corr~1, slope~1 for S_lin")
    for rn in norms:
        m = tags == rn
        c1, c0 = (np.corrcoef(S[m], d_frozen[m])[0, 1] for S in (S_lin, S_lin0))
        sl1, sl0 = (np.polyfit(S[m], d_frozen[m], 1)[0] for S in (S_lin, S_lin0))
        print(f"    rel_norm={rn:7.0e}  S_lin: corr={c1:6.3f} slope={sl1:6.3f} | "
              f"S_lin0 (direction frozen): corr={c0:6.3f} slope={sl0:6.3f}")

    P = M.stack_params([parent] + perts)
    res = lle_batch(P, Zte[100], dt, n_steps=args.n_steps)
    lle0, d_own = res["lle"][0], res["lle"][1:] - res["lle"][0]
    print(f"\n[3] own-orbit test (parent LLE = {lle0:.3f}, {args.n_steps} steps)")
    for rn in norms:
        m = (tags == rn) & res["alive"][1:]
        if m.sum() < 5:
            print(f"    rel_norm={rn:7.0e}  too few alive"); continue
        c = np.corrcoef(S_lin[m], d_own[m])[0, 1]
        print(f"    rel_norm={rn:7.0e}  corr={c:6.3f}  std(dLLE_own)={d_own[m].std():.3e}  "
              f"std(S_lin)={S_lin[m].std():.3e}  sign_agree={(np.sign(S_lin[m]) == np.sign(d_own[m])).mean():.2f}")


if __name__ == "__main__":
    main()
