"""Post hoc analysis for hypothesis H1: does attractor failure occur before one-step error rises?

For every perturbed model of a valid parent, compute the parent-relative one-step MSE
(ratio = MSE / parent MSE) and tabulate failure (diverged or collapsed) by ratio bin.
Writes results_full_30seeds/analysis/h1_failure_vs_mse_ratio.csv, h1_summary.json,
<out>/tables/tab_h1.tex and <out>/figures/fig_h1.pdf.
Usage: python analyze_h1.py --results results_full_30seeds --out paper
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results_full_30seeds")
ap.add_argument("--out", default="paper")
args = ap.parse_args()
R, OUT = args.results, args.out
HALF = 0.25  # same stability threshold as analyze.py
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "raw", "*.csv")))], ignore_index=True)
par = df[df.role == "parent"][["system", "seed", "mse", "alive", "collapsed", "lle_h1", "lle_h2"]]
par = par.rename(columns={c: c + "_par" for c in par.columns if c not in ("system", "seed")})
d = df[df.role != "parent"].merge(par, on=["system", "seed"])
d = d[d.alive_par & ~d.collapsed_par & ((d.lle_h1_par - d.lle_h2_par).abs() <= HALF)].copy()
d["fail"] = (~d.alive) | d.collapsed
d["diverge"] = ~d.alive
d["ratio"] = d.mse / d.mse_par
edges = [0, 3, 10, 30, 100, 1e3, np.inf]
labels = ["<3", "3-10", "10-30", "30-100", "100-1000", ">1000"]
d["bin"] = pd.cut(d.ratio, edges, labels=labels, right=False)

rng = np.random.default_rng(0)
rows, summ = [], {}
for s, g in d.groupby("system"):
    seeds = np.array(sorted(g.seed.unique()))
    for b in labels:
        h = g[g.bin == b]
        # parent-clustered bootstrap CI of the failure fraction
        per = h.groupby("seed").fail.agg(["sum", "size"]).reindex(seeds).fillna(0)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(seeds), len(seeds))
            sm, sz = per["sum"].values[idx].sum(), per["size"].values[idx].sum()
            boots.append(sm / sz if sz else np.nan)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append(dict(system=s, ratio_bin=b, n=len(h), seeds=h.seed.nunique(),
                         fail=h.fail.mean(), diverged=h.diverge.mean(),
                         fail_lo=lo, fail_hi=hi, median_rel_norm=h.rel_norm.median()))
    f = g[g.fail]
    summ[s] = dict(models=len(g), failed=int(len(f)),
                   failed_share_ratio_lt3=float((f.ratio < 3).mean()),
                   failed_share_ratio_lt10=float((f.ratio < 10).mean()),
                   fail_frac_given_ratio_lt3=float(g[g.ratio < 3].fail.mean()),
                   n_ratio_lt3=int((g.ratio < 3).sum()))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(R, "analysis", "h1_failure_vs_mse_ratio.csv"), index=False)
json.dump(summ, open(os.path.join(R, "analysis", "h1_summary.json"), "w"), indent=1)

# LaTeX table
lines = []
for b in labels:
    cells = [b.replace("-", "--").replace("<", "$<$").replace(">", "$>$")]
    for s in ("lorenz", "rossler"):
        r = out[(out.system == s) & (out.ratio_bin == b)].iloc[0]
        cells += [f"{int(r.n):,}".replace(",", "{,}"),
                  f"{100*r.fail:.0f} [{100*r.fail_lo:.0f}, {100*r.fail_hi:.0f}]",
                  f"{100*r.diverged:.0f}"]
    lines.append(" & ".join(cells) + r" \\")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True); os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
lines[-1] = lines[-1].rstrip().rstrip("\\").rstrip()  # main.tex appends the final row terminator
open(os.path.join(OUT, "tables", "tab_h1.tex"), "w").write("\n".join(lines))

# Figure
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
x = np.arange(len(labels))
for a, (s, name) in zip(ax, [("lorenz", "Lorenz-63"), ("rossler", "R\u00f6ssler")]):
    o = out[out.system == s]
    a.errorbar(x, 100 * o.fail, yerr=[100 * (o.fail - o.fail_lo), 100 * (o.fail_hi - o.fail)],
               fmt="o-", color="k", capsize=3, label="diverged or collapsed")
    a.plot(x, 100 * o.diverged, "s--", color="tab:red", label="diverged")
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=7.5)
    a.set_xlabel("one-step MSE / parent MSE"); a.set_title(name, fontsize=10)
    a.grid(alpha=.3); a.set_ylim(0, 100)
ax[0].set_ylabel("failed models (%)"); ax[0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "fig_h1.pdf")); print(out.round(3).to_string()); print(json.dumps(summ, indent=1))
