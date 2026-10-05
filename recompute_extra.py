"""Deterministically rebuild every parent and its perturbations (same RNG path as
run_experiment.py) to extract quantities the main run did not store:

  * per-tensor relative displacement ||d_k||/||theta_k|| and share of ||d theta||^2 per tensor
  * parent Jacobian statistics along its own orbit (mean log max-singular-value rate, tangent rate)

No rollouts are repeated, so this is fast (~10 s per parent). Output rows align one-to-one with
the rows of results/raw/<system>_seed<k>.csv, and the script verifies the rebuild by checking the
parent's one-step MSE and every model's relative norm against the stored CSV.

    python recompute_extra.py --results results_full_30seeds
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from jacsens import model as M
from jacsens.perturb import make_perturbations
from jacsens.sensitivity import reference_orbit
from jacsens.systems import SYSTEMS, initial_condition, integrate

KEYS = M.KEYS


def rebuild(system, seed, cfg):
    dt = SYSTEMS[system]["dt"]
    rng = np.random.default_rng(10_000 * seed + sorted(SYSTEMS).index(system))
    n_total = cfg["n_train"] + cfg["n_test"] + 2
    data = integrate(system, initial_condition(system, rng), n_total, dt_sample=dt,
                     h=cfg["h"], transient=cfg["transient"])
    tr = data[:cfg["n_train"] + 1]
    Z = (data - tr.mean(0)) / tr.std(0)
    Ztr, Zte = Z[:cfg["n_train"] + 1], Z[-(cfg["n_test"] + 1):]
    parent = M.train(M.init_params(rng), Ztr[:-1], Ztr[1:], epochs=cfg["epochs"], lr=cfg["lr"],
                     batch=cfg["batch"], rng=rng)
    perts = make_perturbations(parent, rng, sparsities=tuple(cfg["sparsities"]),
                               bits=tuple(cfg["bits"]), K=cfg["K"])
    return parent, perts, Zte, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_full_30seeds")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.results, "config.json")))
    nseeds = args.seeds or cfg["seeds"]
    out_dir = os.path.join(args.results, "extra")
    os.makedirs(out_dir, exist_ok=True)

    for system in cfg["systems"]:
        for seed in range(cfg["seed0"], cfg["seed0"] + nseeds):
            path = os.path.join(out_dir, f"{system}_seed{seed}.csv")
            if os.path.exists(path):
                continue
            raw = pd.read_csv(os.path.join(args.results, "raw", f"{system}_seed{seed}.csv"))
            parent, perts, Zte, dt = rebuild(system, seed, cfg)
            assert len(raw) == len(perts) + 1, "row count mismatch"
            mse0 = M.mse(parent, Zte[:-1], Zte[1:])
            assert abs(mse0 - raw.mse.iloc[0]) < 1e-12 * max(1.0, abs(mse0)) + 1e-14, \
                f"parent MSE mismatch {mse0} vs {raw.mse.iloc[0]}"

            theta_k = {k: np.linalg.norm(parent[k]) for k in KEYS}
            rows = [dict(system=system, seed=seed, row=0, **{f"r_{k}": 0.0 for k in KEYS},
                         **{f"sh_{k}": 0.0 for k in KEYS})]
            for i, m in enumerate(perts, start=1):
                d2 = {k: float(np.sum((m["params"][k] - parent[k]) ** 2)) for k in KEYS}
                tot = sum(d2.values())
                rn = np.sqrt(tot) / np.sqrt(sum(v ** 2 for v in theta_k.values()))
                assert abs(rn - raw.rel_norm.iloc[i]) < 1e-9, f"rel_norm mismatch row {i}"
                rows.append(dict(system=system, seed=seed, row=i,
                                 **{f"r_{k}": np.sqrt(d2[k]) / theta_k[k] for k in KEYS},
                                 **{f"sh_{k}": d2[k] / tot if tot > 0 else 0.0 for k in KEYS}))
            df = pd.DataFrame(rows)

            orbit, kind = reference_orbit(parent, Zte, 3000)
            J = M.jac_points(parent, orbit)
            smax = np.linalg.norm(J, ord=2, axis=(1, 2))
            df["par_smax_rate"] = float(np.mean(np.log(smax)) / dt)       # mean log max singular value / dt
            df["par_smax_mean"] = float(np.mean(smax))
            df["par_orbit_kind"] = kind
            df.to_csv(path, index=False)
            print(f"[{system} seed {seed}] verified; smax_rate={df.par_smax_rate.iloc[0]:.3f} ({kind})",
                  flush=True)


if __name__ == "__main__":
    main()
