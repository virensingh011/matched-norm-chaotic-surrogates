"""Analyse results of run_experiment.py.

    python analyze.py --results results --max_half_diff 0.25

Outputs (in <results>/analysis/):
  summary.md                      all tables below, human-readable
  matched_pairs.csv               structured perturbation vs its matched controls, per system/group
  predictivity.csv                how well each predictor ranks dynamical damage
  divergence_auc.csv              how well each predictor flags divergence
  fig_<system>_*.png              scatter plots

Statistical unit = seed (parent model). Intervals are bootstrap over seeds.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

PREDICTORS = ["rel_norm", "S_F_rel", "S_2", "absS_lin", "absS_lin0", "absdMSE"]
TARGETS = {"absdLLE": "|dLLE|", "dVPT": "VPT loss (Lyap. times)", "dSW1": "invariant-measure shift (SW1)"}
rng_global = np.random.default_rng(0)


# ------------------------------------------------------------------ helpers
def boot_ci(x, fn=np.mean, n=2000, alpha=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = rng_global.integers(0, len(x), (n, len(x)))
    v = fn(x[idx], axis=1)
    return tuple(np.quantile(v, [alpha / 2, 1 - alpha / 2]))


def wilcoxon_p(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 6 or np.allclose(x, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok])[0])


def partial_spearman(y, x, z):
    """Spearman correlation of y and x after linearly regressing out rank(z) from both ranks."""
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 8:
        return np.nan
    ry, rx, rz = (stats.rankdata(v[ok]) for v in (y, x, z))
    def resid(r):
        A = np.c_[np.ones_like(rz), rz]
        return r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
    a, b = resid(ry), resid(rx)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def fmt(v, p=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{p}g}" if isinstance(v, (float, np.floating)) else str(v)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------- load
def load(results, max_half_diff):
    files = sorted(glob.glob(os.path.join(results, "raw", "*.csv")))
    if not files:
        raise SystemExit(f"no raw CSVs in {results}/raw")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    par = df[df.role == "parent"][["system", "seed", "lle", "mse", "vpt_lt", "sw1", "alive",
                                   "collapsed", "lle_h1", "lle_h2"]]
    par = par.rename(columns={c: c + "_par" for c in par.columns if c not in ("system", "seed")})
    df = df.merge(par, on=["system", "seed"])
    df["half_diff"] = (df.lle_h1 - df.lle_h2).abs()
    df["valid"] = (df.alive & ~df.collapsed & np.isfinite(df.lle) & (df.half_diff <= max_half_diff))
    par_ok = (df.alive_par & ~df.collapsed_par & ((df.lle_h1_par - df.lle_h2_par).abs() <= max_half_diff))
    df["parent_ok"] = par_ok
    df["dLLE"] = df.lle - df.lle_par
    df["absdLLE"] = df.dLLE.abs()
    df["dMSE"] = df.mse - df.mse_par
    df["absdMSE"] = df.dMSE.abs()
    df["dVPT"] = df.vpt_lt_par - df.vpt_lt
    df["dSW1"] = df.sw1 - df.sw1_par
    df["absS_lin"] = df.S_lin.abs()
    df["absS_lin0"] = df.S_lin0.abs()
    df["diverged"] = ~df.alive
    return df


# ------------------------------------------------------- A: matched pairs
def matched_pairs(df):
    rows = []
    for system, ds in df[df.parent_ok].groupby("system"):
        for g, prim_all in ds[ds.role == "primary"].groupby("group"):
            ctrl_all = ds[ds.control_of == g]
            if ctrl_all.empty:
                continue
            for cfam, cd in ctrl_all.groupby("family"):
                diffs, absdiffs, pct, nvalid = [], [], [], []
                div_p, div_c = [], []
                for seed, pr in prim_all.groupby("seed"):
                    pr = pr.iloc[0]
                    cs = cd[cd.seed == seed]
                    div_p.append(float(pr.diverged or pr.collapsed))
                    div_c.append(float((cs.diverged | cs.collapsed).mean()))
                    cs_v = cs[cs.valid]
                    if not pr.valid or len(cs_v) < 3:
                        continue
                    diffs.append(pr.dLLE - cs_v.dLLE.median())
                    absdiffs.append(abs(pr.dLLE) - cs_v.absdLLE.median())
                    pct.append(((cs_v.dLLE < pr.dLLE).sum() + 0.5 * (cs_v.dLLE == pr.dLLE).sum()) / len(cs_v))
                    nvalid.append(len(cs_v))
                if not diffs:
                    continue
                pct = np.array(pct)
                lo, hi = boot_ci(diffs, np.median)
                plo, phi = boot_ci(pct, np.mean)
                rows.append(dict(
                    system=system, structured=g, control=cfam,
                    rel_norm=float(prim_all.rel_norm.mean()), n_seeds=len(diffs),
                    med_dLLE_diff=float(np.median(diffs)), ci_lo=lo, ci_hi=hi,
                    wilcoxon_p=wilcoxon_p(diffs),
                    med_abs_diff=float(np.median(absdiffs)),
                    mean_percentile=float(pct.mean()), pct_lo=plo, pct_hi=phi,
                    frac_seeds_structured_lower=float((np.array(diffs) < 0).mean()),
                    diverge_rate_structured=float(np.mean(div_p)),
                    diverge_rate_control=float(np.mean(div_c))))
    return pd.DataFrame(rows)


# ------------------------------------------------------- B: predictivity
def predictivity(df):
    rows = []
    base = df[(df.role != "parent") & df.parent_ok & df.valid & np.isfinite(df.vpt_lt_par)]
    for system, ds in base.groupby("system"):
        for tgt in TARGETS:
            for pred in PREDICTORS:
                pooled = spearman(ds[pred], ds[tgt])
                per_seed = [spearman(g[pred], g[tgt]) for _, g in ds.groupby("seed")]
                per_seed_partial = [partial_spearman(g[tgt].values, g[pred].values, g["rel_norm"].values)
                                    for _, g in ds.groupby("seed")]
                ps, psp = np.array(per_seed), np.array(per_seed_partial)
                lo, hi = boot_ci(ps, np.nanmean)
                rows.append(dict(
                    system=system, target=tgt, predictor=pred, n=len(ds),
                    pooled_rho=pooled, seed_rho_mean=float(np.nanmean(ps)),
                    seed_rho_lo=lo, seed_rho_hi=hi,
                    partial_rho_given_norm=float(np.nanmean(psp)) if np.isfinite(psp).any() else np.nan))
    return pd.DataFrame(rows)


def sign_check(df, small=0.1):
    rows = []
    base = df[(df.role != "parent") & df.parent_ok & df.valid]
    for system, ds in base.groupby("system"):
        for name, sub in [("all", ds), (f"rel_norm<={small}", ds[ds.rel_norm <= small])]:
            sub = sub[np.isfinite(sub.S_lin) & np.isfinite(sub.dLLE)]
            if len(sub) < 5:
                continue
            rows.append(dict(system=system, subset=name, n=len(sub),
                             pearson=float(np.corrcoef(sub.S_lin, sub.dLLE)[0, 1]),
                             spearman=spearman(sub.S_lin, sub.dLLE),
                             sign_agree=float((np.sign(sub.S_lin) == np.sign(sub.dLLE)).mean())))
    return pd.DataFrame(rows)


def divergence_auc(df):
    rows = []
    base = df[(df.role != "parent") & df.parent_ok]
    for system, ds in base.groupby("system"):
        bad = (ds.diverged | ds.collapsed).values
        for pred in PREDICTORS:
            x = ds[pred].values
            ok = np.isfinite(x)
            if bad[ok].sum() < 5 or (~bad[ok]).sum() < 5:
                continue
            u = stats.mannwhitneyu(x[ok][bad[ok]], x[ok][~bad[ok]]).statistic
            rows.append(dict(system=system, predictor=pred, n=int(ok.sum()),
                             failure_rate=float(bad[ok].mean()),
                             auc=float(u / (bad[ok].sum() * (~bad[ok]).sum()))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots
def plots(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = df[(df.role != "parent") & df.parent_ok & df.valid]
    colors = {"primary": "tab:red", "control": "tab:blue", "grid": "tab:gray"}
    for system, ds in base.groupby("system"):
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, (xcol, xlabel) in zip(axs[:2], [("rel_norm", "relative parameter displacement"),
                                                ("S_F_rel", "relative Jacobian sensitivity S_F")]):
            for role, g in ds.groupby("role"):
                ax.scatter(g[xcol], g.absdLLE + 1e-4, s=9 if role != "primary" else 28,
                           alpha=0.35 if role != "primary" else 0.9, c=colors[role],
                           label=role, zorder=3 if role == "primary" else 1)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(xlabel); ax.set_ylabel("|ΔLLE|")
            rho = spearman(ds[xcol], ds.absdLLE)
            ax.set_title(f"{system}: Spearman ρ = {rho:.2f}")
        axs[0].legend(markerscale=2)
        ax = axs[2]
        sm = ds[ds.rel_norm <= 0.2]
        ax.scatter(sm.S_lin, sm.dLLE, s=10, alpha=0.5, c="tab:green")
        lim = np.nanpercentile(np.abs(np.r_[sm.S_lin, sm.dLLE]), 98) if len(sm) else 1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("first-order prediction S_lin"); ax.set_ylabel("measured ΔLLE")
        ax.set_title("signed first-order prediction (rel_norm ≤ 0.2)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fig_{system}_predictivity.png"), dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--max_half_diff", type=float, default=0.25,
                    help="drop LLE estimates whose two halves disagree by more than this")
    args = ap.parse_args()

    df = load(args.results, args.max_half_diff)
    out = os.path.join(args.results, "analysis")
    os.makedirs(out, exist_ok=True)

    mp, pr, sg, au = matched_pairs(df), predictivity(df), sign_check(df), divergence_auc(df)
    mp.to_csv(os.path.join(out, "matched_pairs.csv"), index=False)
    pr.to_csv(os.path.join(out, "predictivity.csv"), index=False)
    au.to_csv(os.path.join(out, "divergence_auc.csv"), index=False)
    plots(df, out)

    inv = (df[df.role != "parent"].groupby("system")
           .agg(models=("valid", "size"), valid=("valid", "mean"),
                diverged=("diverged", "mean"), collapsed=("collapsed", "mean")).reset_index())
    par = df[df.role == "parent"].groupby("system").agg(
        seeds=("seed", "nunique"), parent_ok=("parent_ok", "mean"),
        parent_lle_mean=("lle", "mean"), parent_mse_median=("mse", "median")).reset_index()

    s = ["# Analysis summary", "",
         f"LLE estimates kept only if rollout finite, not collapsed, and half-run disagreement "
         f"<= {args.max_half_diff}. Intervals: bootstrap over seeds.", "",
         "## Parents", md_table(par), "", "## Model inventory (non-parent)", md_table(inv), "",
         "## A. Structured perturbation vs matched controls (ΔLLE; negative diff = structured more contracting)",
         "Each row pairs a structured perturbation with one matched-norm control family. "
         "`mean_percentile` is where the structured model's ΔLLE falls among the K control draws "
         "(0.5 = indistinguishable from noise).", "", md_table(mp) if len(mp) else "(none)", "",
         "## B. Does the predictor rank dynamical damage? (Spearman ρ)",
         "`seed_rho_mean` = mean within-parent correlation (the honest one: removes between-seed "
         "variation). `partial_rho_given_norm` = within-parent correlation after removing the "
         "rank-trend in rel_norm — the key column for the claim that sensitivity adds "
         "information beyond norm.", ""]
    for tgt, lab in TARGETS.items():
        sub = pr[pr.target == tgt].drop(columns="target") if len(pr) else pr
        s += [f"### Target: {lab}", md_table(sub) if len(sub) else "(none)", ""]
    s += ["## C. First-order sign/size check of S_lin vs measured ΔLLE",
          md_table(sg) if len(sg) else "(none)", "",
          "## D. Predicting divergence/collapse (AUC; 0.5 = chance)",
          md_table(au) if len(au) else "(none)", ""]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(s))
    print("\n".join(s))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
