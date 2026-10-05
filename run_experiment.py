"""Run the matched-perturbation + Jacobian-sensitivity experiment.

Example (paper-scale, ~20-40 min on one CPU core):
    python run_experiment.py --systems lorenz rossler --seeds 30 --K 10 --out results

Quick smoke test (~1 min):
    python run_experiment.py --systems lorenz --seeds 2 --K 3 --epochs 10 \
        --n_steps 3000 --out results_smoke

Each (system, seed) is written to <out>/raw/<system>_seed<k>.csv; finished pairs are skipped
on re-run, so the script is resumable.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from jacsens import model as M
from jacsens.dynamics import lle_batch, sliced_w1, vpt_batch
from jacsens.perturb import make_perturbations
from jacsens.sensitivity import TrajectoryRef, reference_orbit
from jacsens.systems import SYSTEMS, initial_condition, integrate, reference_lle


def run_one(system, seed, args, ref_lle):
    dt = SYSTEMS[system]["dt"]
    rng = np.random.default_rng(10_000 * seed + sorted(SYSTEMS).index(system))

    # ---- data: one long trajectory, first n_train+1 states train, last n_test+1 test
    n_total = args.n_train + args.n_test + 2
    data = integrate(system, initial_condition(system, rng), n_total, dt_sample=dt,
                     h=args.h, transient=args.transient)
    tr = data[:args.n_train + 1]
    mu, sd = tr.mean(0), tr.std(0)
    Z = (data - mu) / sd
    Ztr = Z[:args.n_train + 1]
    Zte = Z[-(args.n_test + 1):]
    Xtr, Ytr, Xte, Yte = Ztr[:-1], Ztr[1:], Zte[:-1], Zte[1:]

    # ---- parent + perturbations
    parent = M.train(M.init_params(rng), Xtr, Ytr, epochs=args.epochs, lr=args.lr,
                     batch=args.batch, rng=rng)
    perts = make_perturbations(parent, rng, sparsities=tuple(args.sparsities),
                               bits=tuple(args.bits), K=args.K)
    all_models = [dict(role="parent", family="parent", level=0.0, group="parent",
                       control_of="", draw=0, params=parent, rel_norm=0.0)] + perts
    P = M.stack_params([m["params"] for m in all_models])

    # ---- metrics
    mses = np.array([M.mse(m["params"], Xte, Yte) for m in all_models])

    x0 = Xte[100]
    res = lle_batch(P, x0, dt, n_transient=args.n_transient, n_steps=args.n_steps)
    ref_idx = np.linspace(0, len(Xte) - 1, 2000).astype(int)
    sw1 = sliced_w1(res["samples"], Xte[ref_idx])
    sw1[~res["alive"]] = np.nan
    spread = res["samples"].std(axis=1).mean(axis=1)
    collapsed = res["alive"] & (spread < args.collapse_std)

    H = args.vpt_horizon
    offsets = np.linspace(0, len(Zte) - H - 2, args.n_ic).astype(int)
    vpt = vpt_batch(P, Zte, offsets, H, dt)

    orbit, orbit_kind = reference_orbit(parent, Xte, args.n_sens_pts, mode=args.sens_orbit,
                                        collapse_std=args.collapse_std)
    ref = TrajectoryRef(parent, orbit, dt, burn=args.sens_burn)
    zero = dict(S_F=0.0, S_F_rel=0.0, S_2=0.0, S_lin=0.0, S_lin0=0.0)
    sens = [zero] + ref.metrics_batch([m["params"] for m in all_models[1:]])

    rows = []
    for i, m in enumerate(all_models):
        rows.append(dict(
            system=system, seed=seed, role=m["role"], family=m["family"], level=m["level"],
            group=m["group"], control_of=m["control_of"], draw=m["draw"],
            rel_norm=m["rel_norm"], mse=mses[i], lle=res["lle"][i], lle_h1=res["lle_h1"][i],
            lle_h2=res["lle_h2"][i], alive=bool(res["alive"][i]), collapsed=bool(collapsed[i]),
            sw1=sw1[i], vpt=vpt[i], vpt_lt=vpt[i] * ref_lle, sens_orbit=orbit_kind, **sens[i]))
    return pd.DataFrame(rows), ref.parent_orbit_lle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["lorenz", "rossler"], choices=list(SYSTEMS))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--K", type=int, default=10, help="noise draws per matched control")
    ap.add_argument("--sparsities", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3])
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--h", type=float, default=0.001, help="RK4 internal step")
    ap.add_argument("--transient", type=float, default=20.0)
    ap.add_argument("--n_transient", type=int, default=1000)
    ap.add_argument("--n_steps", type=int, default=10000, help="LLE rollout length (steps)")
    ap.add_argument("--collapse_std", type=float, default=0.05)
    ap.add_argument("--vpt_horizon", type=int, default=500)
    ap.add_argument("--n_ic", type=int, default=20)
    ap.add_argument("--n_sens_pts", type=int, default=3000)
    ap.add_argument("--sens_burn", type=int, default=300)
    ap.add_argument("--sens_orbit", choices=["parent", "true"], default="parent",
                    help="orbit for Jacobian sensitivity: parent's own rollout or true data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    ref_path = os.path.join(args.out, "reference_lle.json")
    refs = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
    for system in args.systems:
        if system not in refs:
            t0 = time.time()
            refs[system] = reference_lle(system)
            print(f"[{system}] reference LLE = {refs[system]:.4f}  ({time.time() - t0:.0f}s)")
            json.dump(refs, open(ref_path, "w"), indent=2)

    for system in args.systems:
        for seed in range(args.seed0, args.seed0 + args.seeds):
            path = os.path.join(args.out, "raw", f"{system}_seed{seed}.csv")
            if os.path.exists(path):
                continue
            t0 = time.time()
            df, orbit_lle = run_one(system, seed, args, refs[system])
            df["parent_orbit_lle"] = orbit_lle
            df.to_csv(path, index=False)
            par = df[df.role == "parent"].iloc[0]
            print(f"[{system} seed {seed}] {len(df)} models | parent mse={par.mse:.2e} "
                  f"lle={par.lle:.3f} (ref {refs[system]:.3f}) | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
