"""Survival-unconditional paired comparison of structured perturbations vs matched controls.

The LLE-based comparison in analyze.py only uses models whose rollout survives, which leaves
few valid pairs at larger norms. Valid prediction time (VPT), one-step MSE and the failure
indicator are defined for EVERY model, so these comparisons use all seeds.

    python analyze_paired.py --results results

For each (system, structured group, control family) and each seed (unit of replication):
  d_vpt   = VPT_structured - median_k VPT_control      (Lyapunov times; >0: structured better)
  d_logmse= log10(MSE_structured) - median_k log10(MSE_control)   (>0: structured worse)
  d_fail  = fail_structured - mean_k fail_control      (fail = diverged or collapsed)
Reported: median over seeds, bootstrap 95% CI over seeds, Wilcoxon signed-rank p, and
Holm-adjusted p within each (system, metric) family of tests.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(0)


def boot_ci(x, fn=np.median, n=4000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    v = fn(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return tuple(np.quantile(v, [0.025, 0.975]))


def wilcoxon_p(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 6 or np.allclose(x, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--max_half_diff", type=float, default=0.25)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "raw", "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["fail"] = (~df.alive) | df.collapsed
    df["logmse"] = np.log10(df.mse)
    par = df[df.role == "parent"].copy()
    par["ok"] = par.alive & ~par.collapsed & ((par.lle_h1 - par.lle_h2).abs() <= args.max_half_diff)
    good = set(map(tuple, par[par.ok][["system", "seed"]].values))
    df = df[[(s, k) in good for s, k in zip(df.system, df.seed)]]

    rows = []
    for system, ds in df.groupby("system"):
        for g, prim in ds[ds.role == "primary"].groupby("group"):
            ctrl_all = ds[ds.control_of == g]
            for cfam, cd in ctrl_all.groupby("family"):
                d_vpt, d_mse, d_fail = [], [], []
                for seed, pr in prim.groupby("seed"):
                    pr = pr.iloc[0]
                    cs = cd[cd.seed == seed]
                    if cs.empty:
                        continue
                    d_vpt.append(pr.vpt_lt - cs.vpt_lt.median())
                    d_mse.append(pr.logmse - cs.logmse.median())
                    d_fail.append(float(pr.fail) - cs.fail.mean())
                if not d_vpt:
                    continue
                for name, v in (("d_vpt", d_vpt), ("d_logmse", d_mse), ("d_fail", d_fail)):
                    lo, hi = boot_ci(v)
                    rows.append(dict(system=system, structured=g, control=cfam, metric=name,
                                     rel_norm=float(prim.rel_norm.mean()), n_seeds=len(v),
                                     median=float(np.median(v)), ci_lo=lo, ci_hi=hi,
                                     p=wilcoxon_p(v)))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for (s, m), idx in out.groupby(["system", "metric"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].values)
    od = os.path.join(args.results, "analysis")
    os.makedirs(od, exist_ok=True)
    out.to_csv(os.path.join(od, "paired_unconditional.csv"), index=False)
    print(out.round(4).to_string(index=False))
    predictivity_vpt_all(args.results, args.max_half_diff)


def predictivity_vpt_all(results, max_half_diff):
    """Within-parent Spearman of predictors vs VPT loss over ALL perturbed models of valid
    parents (no survival filter): VPT, MSE and the Jacobian measures exist for every model."""
    from analyze import load, partial_spearman, spearman
    df = load(results, max_half_diff)
    base = df[(df.role != "parent") & df.parent_ok & np.isfinite(df.vpt_lt) & np.isfinite(df.vpt_lt_par)]
    rows = []
    for system, ds in base.groupby("system"):
        for pred in ["rel_norm", "S_F_rel", "S_2", "absS_lin", "absS_lin0", "absdMSE"]:
            per = [spearman(g[pred], g["dVPT"]) for _, g in ds.groupby("seed")]
            part = [partial_spearman(g["dVPT"].values, g[pred].values, g["rel_norm"].values)
                    for _, g in ds.groupby("seed")]
            per, part = np.array(per), np.array(part)
            lo, hi = boot_ci(per, np.nanmean)
            rows.append(dict(system=system, target="dVPT", predictor=pred, n=len(ds),
                             pooled_rho=spearman(ds[pred], ds["dVPT"]),
                             seed_rho_mean=float(np.nanmean(per)), seed_rho_lo=lo, seed_rho_hi=hi,
                             partial_rho_given_norm=float(np.nanmean(part))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(results, "analysis", "predictivity_vpt_all.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
