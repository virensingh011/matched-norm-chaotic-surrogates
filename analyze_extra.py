"""Extra analyses for the upgraded paper.  Needs recompute_extra.py output (results/extra/*.csv).

    python analyze_extra.py --results results_full_30seeds

A. Dynamical robustness radius: relative displacement r_p at which a fraction p of isotropic-noise
   perturbations fail (logistic fit of failure on log10 ||d theta||/||theta||, bootstrap over parents),
   and observed failure curves of structured families.
B. Valid-prediction-time law: VPT_time = a + b * ln(one-step MSE).  If one-step error e1 = sqrt(MSE)
   is amplified at an effective rate lam_eff, VPT ~ (1/lam_eff) ln(thr/e1), i.e. b = -1/(2 lam_eff).
   We fit b per system and per parent, compare lam_eff with the Lyapunov exponent and with the mean
   log max-singular-value rate of the parent Jacobian, and test which gives a better predictor.
C. Mechanism test: how are squared displacements distributed across tensors for each perturbation
   family, and does a layer-resolved displacement vector predict damage better than the scalar
   norm, and does perturbation family still add information once it is known?
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

from analyze import boot_ci, load

TENSORS = ["W1", "b1", "W2", "b2", "W3", "b3"]
rng = np.random.default_rng(0)


def attach_extra(df, results):
    files = sorted(glob.glob(os.path.join(results, "extra", "*.csv")))
    if not files:
        raise SystemExit("run recompute_extra.py first")
    ex = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.copy()
    df["row"] = df.groupby(["system", "seed"]).cumcount()
    return df.merge(ex, on=["system", "seed", "row"], how="inner")


def family_label(r):
    if r["role"] == "primary":
        return r["family"]
    if r["role"] in ("control", "grid"):
        return "noise"
    return "parent"


# ------------------------------------------------------------------ A
def logistic_fit(x, y):
    """MLE of P(fail) = sigmoid(a + b x); returns (a, b)."""
    def nll(p):
        z = p[0] + p[1] * x
        return np.sum(np.logaddexp(0, z) - y * z) + 1e-6 * np.sum(p ** 2)
    r = optimize.minimize(nll, np.array([0.0, 1.0]), method="BFGS")
    return r.x


def radius(a, b, p):
    return 10 ** ((np.log(p / (1 - p)) - a) / b)


def robustness(df, nboot=300):
    rows, curves = [], []
    for system, ds in df[df.parent_ok & (df.role != "parent")].groupby("system"):
        ds = ds.assign(fam=ds.apply(family_label, axis=1),
                       fail=((~ds.alive) | ds.collapsed).astype(float),
                       div=(~ds.alive).astype(float))
        iso = ds[(ds.fam == "noise") & (ds.family == "noise_iso")]
        seeds = np.array(sorted(iso.seed.unique()))
        for tag, col in [("fail (diverge or collapse)", "fail"), ("diverge only", "div")]:
            x, y = np.log10(iso.rel_norm.values), iso[col].values
            a, b = logistic_fit(x, y)
            est = {p: radius(a, b, p) for p in (0.1, 0.5, 0.9)}
            boots = {p: [] for p in est}
            by_seed = {s: g for s, g in iso.groupby("seed")}
            for _ in range(nboot):
                pick = rng.choice(seeds, len(seeds))
                g = pd.concat([by_seed[s] for s in pick])
                ab = logistic_fit(np.log10(g.rel_norm.values), g[col].values)
                for p in est:
                    boots[p].append(radius(ab[0], ab[1], p))
            for p in est:
                lo, hi = np.quantile(boots[p], [0.025, 0.975])
                rows.append(dict(system=system, outcome=tag, p=p, radius=est[p], ci_lo=lo, ci_hi=hi))
        for fam, g in ds[ds.role == "primary"].groupby("family"):
            for lvl, h in g.groupby("level"):
                curves.append(dict(system=system, family=fam, level=lvl, rel_norm=h.rel_norm.mean(),
                                   fail=h['fail'].mean(), div=h['div'].mean(), n=len(h)))
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ------------------------------------------------------------------ B
WINDOW = (-6.0, -3.0)          # log10 one-step MSE window used for the primary VPT fit


def select_law(ds, mode, lam_ref_sys=None, min_lt=0.05):
    """Primary: select on the PREDICTOR (log10 MSE in WINDOW), which does not bias the slope.
    'floor': select on the OUTCOME (VPT above floor), kept as a sensitivity check -- selecting on the
    outcome truncates the data and biases the slope toward zero."""
    if mode == "window":
        lm = np.log10(ds.mse)
        return ds[(lm >= WINDOW[0]) & (lm <= WINDOW[1])]
    return ds[ds.vpt_lt > min_lt]


def vpt_law(df, lam_ref, mode="window", min_n=30):
    out, per_parent = [], []
    for system, ds in df[df.parent_ok].groupby("system"):
        d = select_law(ds, mode).copy()
        d["t"] = d.vpt_lt / lam_ref[system]                    # VPT in time units
        d["lnmse"] = np.log(d.mse)
        r = stats.linregress(d.lnmse, d.t)
        out.append(dict(system=system, mode=mode, n=len(d), frac_at_floor=float((d.vpt_lt < 0.05).mean()), slope_time_per_ln=r.slope, intercept=r.intercept,
                        R2=r.rvalue ** 2, lam_eff=-1 / (2 * r.slope), lam_ref=lam_ref[system],
                        lam_eff_over_lam_ref=-1 / (2 * r.slope) / lam_ref[system]))
        for seed, g in d.groupby("seed"):
            if len(g) < min_n:
                continue
            rr = stats.linregress(g.lnmse, g.t)
            p = ds[(ds.seed == seed) & (ds.role == "parent")].iloc[0]
            per_parent.append(dict(system=system, seed=seed, n=len(g), slope=rr.slope, R2=rr.rvalue ** 2,
                                   lam_eff=-1 / (2 * rr.slope) if rr.slope < 0 else np.nan,
                                   lle_parent=p.lle, orbit_tangent_rate=p.parent_orbit_lle,
                                   smax_rate=p.par_smax_rate))
    return pd.DataFrame(out), pd.DataFrame(per_parent)


def law_predictors(df, lam_ref, mode="window"):
    """Leave-one-parent-out R2 of three VPT models on unsaturated models.
    M0: constant slope.  M1: slope = -1/(2 c lam_p) with lam_p = parent LLE (>0 required).
    M2: slope = -1/(2 c lam_p) with lam_p = parent mean log max-singular-value rate."""
    rows = []
    for system, ds in df[df.parent_ok].groupby("system"):
        d = select_law(ds, mode).copy()
        d["t"] = d.vpt_lt / lam_ref[system]
        d["lnmse"] = np.log(d.mse)
        seeds = sorted(d.seed.unique())
        for name, col in [("M0 constant slope", None), ("M1 lam = parent LLE", "lle_par"),
                          ("M2 lam = Jacobian smax rate", "par_smax_rate")]:
            sse = sst = 0.0
            for s in seeds:
                tr, te = d[d.seed != s], d[d.seed == s]
                if col is None:
                    r = stats.linregress(tr.lnmse, tr.t)
                    pred = r.intercept + r.slope * te.lnmse
                else:
                    ok = tr[col] > 0.02
                    trn = tr[ok]
                    if trn.empty or not (te[col] > 0.02).all():
                        continue
                    f = np.log(trn.mse) / trn[col]
                    A = np.c_[np.ones(len(trn)), -0.5 * f]
                    coef = np.linalg.lstsq(A, trn.t, rcond=None)[0]       # a, 1/c
                    pred = coef[0] + coef[1] * (-0.5 * np.log(te.mse) / te[col])
                sse += float(np.sum((te.t - pred) ** 2))
                sst += float(np.sum((te.t - tr.t.mean()) ** 2))
            rows.append(dict(system=system, model=name, loo_R2=1 - sse / sst if sst > 0 else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C
def layer_shares(df):
    d = df[df.parent_ok & (df.role != "parent")].copy()
    d["fam"] = d.apply(family_label, axis=1)
    d.loc[(d.role == "control"), "fam"] = d.family            # noise_iso / noise_layer / noise_support
    d.loc[d.role == "grid", "fam"] = "noise_iso"
    g = d.groupby(["system", "fam"])[[f"sh_{k}" for k in TENSORS]].mean().reset_index()
    return g


def layer_resolved_cv(df, nfold=5):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    rows = []
    for system, ds in df[df.parent_ok & (df.role != "parent")].groupby("system"):
        d = ds.copy()
        d["fam"] = d.apply(family_label, axis=1)
        d["vpt_lost"] = np.clip(1 - d.vpt_lt / d.vpt_lt_par, 0, 1)
        d["fail"] = ((~d.alive) | d.collapsed).astype(int)
        d["lr"] = np.log10(d.rel_norm)
        for k in TENSORS:
            d[f"lr_{k}"] = np.log10(d[f"r_{k}"] + 1e-5)
        fam = pd.get_dummies(d.fam).astype(float)
        lrel = np.log10(d.mse.values / d.mse_par.values)        # parent-relative one-step MSE
        pt = d[[f"lr_{k}" for k in TENSORS]].values
        sets = {"scalar norm": d[["lr"]].values,
                "scalar norm + family": np.c_[d[["lr"]].values, fam.values],
                "per-tensor norms": pt,
                "per-tensor norms + family": np.c_[pt, fam.values],
                "scalar norm + relative MSE": np.c_[d[["lr"]].values, lrel],
                "per-tensor + family + relative MSE": np.c_[pt, fam.values, lrel]}
        groups = d.seed.values
        for name, X in sets.items():
            pr, pc = np.zeros(len(d)), np.zeros(len(d))
            for tr, te in GroupKFold(n_splits=nfold).split(X, groups=groups):
                reg = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[tr], d.vpt_lost.values[tr])
                clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], d.fail.values[tr])
                pr[te], pc[te] = reg.predict(X[te]), clf.predict_proba(X[te])[:, 1]
            rows.append(dict(system=system, features=name, n=len(d),
                             R2_vpt_lost=r2_score(d.vpt_lost, pr), AUC_failure=roc_auc_score(d.fail, pc)))
    return pd.DataFrame(rows)


def tensor_ratio(df):
    """Per-tensor RELATIVE displacement of isotropic noise divided by that of the structured perturbation
    it is matched to (same total norm); median over structured groups."""
    rows = []
    for system, ds in df[df.parent_ok & (df.role != "parent")].groupby("system"):
        per = []
        for g, p in ds[ds.role == "primary"].groupby("group"):
            c = ds[(ds.control_of == g) & (ds.family == "noise_iso")]
            if c.empty:
                continue
            per.append({k: c[f"r_{k}"].mean() / max(p[f"r_{k}"].mean(), 1e-12) for k in ["W1", "W2", "W3"]})
        r = pd.DataFrame(per)
        rows.append(dict(system=system, **{f"ratio_{k}": float(r[k].median()) for k in r.columns},
                         n_groups=len(r)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_full_30seeds")
    ap.add_argument("--max_half_diff", type=float, default=0.25)
    ap.add_argument("--nboot", type=int, default=300)
    args = ap.parse_args()
    lam_ref = json.load(open(os.path.join(args.results, "reference_lle.json")))
    od = os.path.join(args.results, "analysis")
    os.makedirs(od, exist_ok=True)

    df = attach_extra(load(args.results, args.max_half_diff), args.results)
    print("parents with extra:", df[df.role == "parent"].groupby("system").size().to_dict())

    rob, curves = robustness(df, args.nboot)
    law, per_parent = vpt_law(df, lam_ref, "window")
    law_floor, _ = vpt_law(df, lam_ref, "floor")
    lp = law_predictors(df, lam_ref, "window")
    sh = layer_shares(df)
    cv = layer_resolved_cv(df)
    tr = tensor_ratio(df)
    for name, t in [("robustness_radius", rob), ("failure_curves", curves), ("vpt_law", law), ("vpt_law_floor", law_floor),
                    ("vpt_law_per_parent", per_parent), ("vpt_law_loo", lp),
                    ("layer_shares", sh), ("layer_resolved_cv", cv), ("tensor_ratio", tr)]:
        t.to_csv(os.path.join(od, f"extra_{name}.csv"), index=False)
        print(f"\n== {name}\n{t.round(4).to_string(index=False)}")

    if len(per_parent):
        print("\n== across-parent association of lam_eff with parent quantities (Spearman)")
        for system, g in per_parent.groupby("system"):
            g = g[np.isfinite(g.lam_eff)]
            for col in ["lle_parent", "orbit_tangent_rate", "smax_rate"]:
                print(system, col, "n=", len(g), "rho=", round(stats.spearmanr(g.lam_eff, g[col])[0], 3))


if __name__ == "__main__":
    main()
