"""Generate every figure and LaTeX table in the paper from the raw results.

    python make_paper_assets.py --results results_full_30seeds --out paper

Requires analyze.py outputs (predictivity.csv, divergence_auc.csv) and analyze_paired.py output
(paired_unconditional.csv) to exist in <results>/analysis/.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import load
from analyze_extra import attach_extra, family_label, logistic_fit
from scipy import stats

SYS_NAME = {"lorenz": "Lorenz-63", "rossler": "R\\\"ossler"}
SYS_PLOT = {"lorenz": "Lorenz-63", "rossler": "Rössler"}
FAM_STYLE = {"prune_global": ("Global pruning", "s", "tab:red"),
             "prune_layer": ("Layer-wise pruning", "^", "tab:orange"),
             "prune_random": ("Random pruning", "D", "tab:purple"),
             "quant": ("Quantization", "o", "tab:green")}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})


def pretty_group(g):
    fam, lvl = g.split("@")
    if fam == "quant":
        return f"{int(float(lvl))}-bit quantization"
    name = {"prune_global": "Global pruning", "prune_layer": "Layer-wise pruning",
            "prune_random": "Random pruning"}[fam]
    return f"{name} {int(round(float(lvl) * 100))}\\%"


def group_order(groups):
    def key(g):
        fam, lvl = g.split("@")
        f = {"prune_global": 0, "prune_layer": 1, "quant": 2, "prune_random": 3}[fam]
        return (f, -float(lvl) if fam == "quant" else float(lvl))
    return sorted(groups, key=key)


def fmt(x, p=2):
    return "--" if not np.isfinite(x) else f"{x:.{p}f}"


# ------------------------------------------------------------------- figures
def fig_failure(df, out):
    nonp = df[(df.role != "parent") & df.parent_ok].copy()
    nonp["div"] = nonp.diverged.astype(float)
    nonp["col"] = (nonp.collapsed & nonp.alive).astype(float)
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)
    for r, system in enumerate(["lorenz", "rossler"]):
        ds = nonp[nonp.system == system]
        grid = ds[ds.role == "grid"].groupby("level").agg(x=("rel_norm", "mean"), div=("div", "mean"),
                                                        col=("col", "mean")).reset_index()
        prim = ds[ds.role == "primary"].groupby(["family", "level"]).agg(
            x=("rel_norm", "mean"), div=("div", "mean"), col=("col", "mean")).reset_index()
        for c, (key, title) in enumerate([("div", "diverged (left bounded region)"), ("col", "collapsed to near-constant state")]):
            ax = axs[r, c]
            ax.plot(grid.x, grid[key], "k-o", ms=3, lw=1, label="isotropic noise (grid)")
            for fam, (lab, mk, col) in FAM_STYLE.items():
                p = prim[prim.family == fam]
                ax.scatter(p.x, p[key], marker=mk, c=col, s=34, label=lab, zorder=3, edgecolor="k", lw=0.4)
            ax.set_xscale("log")
            ax.set_title(f"{SYS_PLOT[system]}: {title}")
            ax.set_ylim(-0.03, 1.03)
            if r == 1:
                ax.set_xlabel("relative parameter displacement $\\|\\Delta\\theta\\|/\\|\\theta\\|$")
            if c == 0:
                ax.set_ylabel("fraction of perturbed models")
            ax.grid(alpha=0.25)
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_failure.pdf"))
    plt.close(fig)


def fig_scatter_vpt(df, out):
    ds = df[(df.role != "parent") & df.parent_ok & np.isfinite(df.vpt_lt_par)].copy()
    ds["cls"] = np.where(ds.role == "primary", "structured", "isotropic / matched noise")
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    for ax, system in zip(axs, ["lorenz", "rossler"]):
        d = ds[ds.system == system]
        for cls, col, s, a, z in [("isotropic / matched noise", "tab:blue", 5, 0.18, 1),
                                  ("structured", "tab:red", 20, 0.85, 3)]:
            g = d[d.cls == cls]
            ax.scatter(g.rel_norm, g.dVPT, c=col, s=s, alpha=a, label=cls, zorder=z, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("relative parameter displacement $\\|\\Delta\\theta\\|/\\|\\theta\\|$")
        ax.set_title(SYS_PLOT[system])
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("loss of valid prediction time (Lyapunov times)")
    axs[0].legend(fontsize=7, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_vpt_vs_norm.pdf"))
    plt.close(fig)


def fig_predictivity(pred, out):
    targets = [("absdLLE", "$|\\Delta$LLE$|$"), ("dVPT", "VPT loss"), ("dSW1", "invariant-measure shift")]
    preds = [("rel_norm", "$\\|\\Delta\\theta\\|/\\|\\theta\\|$", "tab:gray"),
             ("S_F_rel", "$S_F$ (rel. Jacobian change)", "tab:blue"),
             ("absS_lin", "$|S_{lin}|$ (first order)", "tab:cyan"),
             ("absdMSE", "$|\\Delta$ one-step MSE$|$", "tab:red")]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    w = 0.2
    for ax, system in zip(axs, ["lorenz", "rossler"]):
        for j, (pk, plab, col) in enumerate(preds):
            vals, lo, hi = [], [], []
            for tk, _ in targets:
                r = pred[(pred.system == system) & (pred.target == tk) & (pred.predictor == pk)].iloc[0]
                vals.append(r.seed_rho_mean); lo.append(r.seed_rho_mean - r.seed_rho_lo)
                hi.append(r.seed_rho_hi - r.seed_rho_mean)
            x = np.arange(len(targets)) + (j - 1.5) * w
            ax.bar(x, vals, w, yerr=[lo, hi], color=col, label=plab, capsize=2, lw=0.5, edgecolor="k")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([t[1] for t in targets])
        ax.set_title(SYS_PLOT[system])
        ax.axhline(0, c="k", lw=0.6)
        ax.grid(axis="y", alpha=0.25)
    axs[0].set_ylabel("within-parent Spearman $\\rho$ (95% CI over seeds)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(out, "fig_predictivity.pdf"))
    plt.close(fig)


def fig_forest(pair, out):
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 8.4), sharey="row")
    metrics = [("d_vpt", "VPT difference, structured $-$ control\n(Lyapunov times; $>0$: structured better)"),
               ("d_logmse", "$\\log_{10}$ MSE difference, structured $-$ control\n($<0$: structured lower error)")]
    for r, system in enumerate(["lorenz", "rossler"]):
        groups = group_order(pair[(pair.system == system)].structured.unique())
        ypos = {g: i for i, g in enumerate(groups[::-1])}
        for c, (mk, mlab) in enumerate(metrics):
            ax = axs[r, c]
            for ctrl, col, off, lab in [("noise_iso", "tab:blue", 0.17, "isotropic noise"),
                                        ("noise_layer", "tab:orange", -0.17, "layer-norm-matched noise")]:
                d = pair[(pair.system == system) & (pair.metric == mk) & (pair.control == ctrl)]
                for _, row in d.iterrows():
                    y = ypos[row.structured] + off
                    ax.plot([row.ci_lo, row.ci_hi], [y, y], c=col, lw=1.4)
                    ax.plot(row["median"], y, "o", c=col, ms=3.5, label=lab if row.structured == groups[0] else None)
            ax.axvline(0, c="k", lw=0.7)
            ax.set_yticks(list(ypos.values()))
            ax.set_yticklabels([pretty_group(g).replace("\\%", "%") for g in ypos.keys()], fontsize=7)
            ax.set_title(f"{SYS_PLOT[system]}")
            ax.set_xlabel(mlab, fontsize=8)
            ax.grid(axis="x", alpha=0.25)
    axs[0, 0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_forest.pdf"))
    plt.close(fig)


def fig_slin_validity(out):
    # values printed by sanity_check.py (system=lorenz, 40 draws per norm, 20000-step own-orbit LLE)
    norms = np.array([1e-4, 1e-3, 3e-3, 1e-2])
    frozen = np.array([0.998, 0.868, 0.376, 0.030])
    frozen0 = np.array([0.018, 0.279, 0.321, 0.510])
    own = np.array([0.030, 0.190, -0.081, 0.454])
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(norms, frozen, "o-", c="tab:blue", label="$S_{lin}$ vs exact frozen-orbit change")
    ax.plot(norms, frozen0, "s--", c="tab:gray", label="$S_{lin}^{(0)}$ (direction frozen) vs frozen-orbit change")
    ax.plot(norms, own, "^-", c="tab:red", label="$S_{lin}$ vs actual $\\Delta$LLE (own orbit)")
    ax.set_xscale("log"); ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("relative parameter displacement"); ax.set_ylabel("correlation")
    ax.legend(fontsize=6.5, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_slin_validity.pdf"))
    plt.close(fig)


# -------------------------------------------------------------------- tables
def table_failure(df, out):
    nonp = df[(df.role != "parent") & df.parent_ok].copy()
    nonp["div"] = nonp.diverged.astype(float)
    nonp["col"] = (nonp.collapsed & nonp.alive).astype(float)
    lines = []
    for system in ["lorenz", "rossler"]:
        ds = nonp[nonp.system == system]
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{SYS_NAME[system]}}} ({ds.seed.nunique()} valid parents)}} \\\\")
        prim = ds[(ds.role == "primary") & (ds.family != "prune_random")]
        for g in group_order(prim.group.unique()):
            p = prim[prim.group == g]
            c = ds[(ds.control_of == g) & (ds.family == "noise_iso")]
            lines.append(" & ".join([pretty_group(g), fmt(p.rel_norm.mean(), 3),
                                     fmt(p["div"].mean()), fmt(p["col"].mean()),
                                     fmt(c["div"].mean()), fmt(c["col"].mean())]) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    open(os.path.join(out, "tab_failure.tex"), "w").write("\n".join(lines) + "\n")


def table_predict(pred, out):
    names = [("rel_norm", "$\\|\\Delta\\theta\\|/\\|\\theta\\|$"), ("S_F_rel", "$S_F$ (relative)"),
             ("S_2", "$S_2$"), ("absS_lin", "$|S_{lin}|$"), ("absS_lin0", "$|S_{lin}^{(0)}|$"),
             ("absdMSE", "$|\\Delta$ one-step MSE$|$")]
    tg = [("absdLLE", "$|\\Delta$LLE$|$"), ("dVPT", "VPT loss"), ("dSW1", "SW$_1$ shift")]
    for kind, fname in [("rho", "tab_predict.tex"), ("partial", "tab_partial.tex")]:
        lines = []
        for system in ["lorenz", "rossler"]:
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{SYS_NAME[system]}}}}} \\\\")
            for pk, pn in names:
                cells = [pn]
                for tk, _ in tg:
                    r = pred[(pred.system == system) & (pred.target == tk) & (pred.predictor == pk)].iloc[0]
                    if kind == "rho":
                        cells.append(f"{r.seed_rho_mean:.2f} [{r.seed_rho_lo:.2f}, {r.seed_rho_hi:.2f}]")
                    else:
                        cells.append(f"{r.partial_rho_given_norm:.2f}")
                lines.append(" & ".join(cells) + " \\\\")
            lines.append("\\midrule")
        open(os.path.join(out, fname), "w").write("\n".join(lines[:-1]) + "\n")


def table_auc(auc, out):
    names = [("rel_norm", "$\\|\\Delta\\theta\\|/\\|\\theta\\|$"), ("S_F_rel", "$S_F$ (relative)"),
             ("S_2", "$S_2$"), ("absS_lin", "$|S_{lin}|$"), ("absS_lin0", "$|S_{lin}^{(0)}|$"),
             ("absdMSE", "$|\\Delta$ one-step MSE$|$")]
    lines = []
    for pk, pn in names:
        cells = [pn]
        for system in ["lorenz", "rossler"]:
            r = auc[(auc.system == system) & (auc.predictor == pk)].iloc[0]
            cells.append(f"{r.auc:.3f}")
        lines.append(" & ".join(cells) + " \\\\")
    open(os.path.join(out, "tab_auc.tex"), "w").write("\n".join(lines) + "\n")


def table_paired(pair, out):
    def cell(r):
        return f"{r['median']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", r.p_holm

    for mk, fname, p in [("d_vpt", "tab_paired_vpt.tex", 3), ("d_logmse", "tab_paired_mse.tex", 2)]:
        lines = []
        for system in ["lorenz", "rossler"]:
            d = pair[(pair.system == system) & (pair.metric == mk)]
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{SYS_NAME[system]}}}}} \\\\")
            for g in group_order(d.structured.unique()):
                row = [pretty_group(g), fmt(d[d.structured == g].rel_norm.iloc[0], 3),
                       str(int(d[d.structured == g].n_seeds.iloc[0]))]
                for ctrl in ["noise_iso", "noise_layer"]:
                    x = d[(d.structured == g) & (d.control == ctrl)]
                    if x.empty:
                        row += ["--", "--"]
                        continue
                    cstr, ph = cell(x.iloc[0].copy())
                    cstr = cstr.replace(f"{x.iloc[0]['median']:+.3f}", f"{x.iloc[0]['median']:+.{p}f}")
                    row += [f"{x.iloc[0]['median']:+.{p}f} [{x.iloc[0].ci_lo:+.{p}f}, {x.iloc[0].ci_hi:+.{p}f}]",
                            "$<$0.001" if ph < 1e-3 else f"{ph:.3f}"]
                lines.append(" & ".join(row) + " \\\\")
            lines.append("\\midrule")
        open(os.path.join(out, fname), "w").write("\n".join(lines[:-1]) + "\n")


def fix_headers(tabdir):
    """\\input after \\midrule breaks a leading \\multicolumn; rewrite section-header rows as plain cells."""
    import re
    for f in os.listdir(tabdir):
        if not f.endswith(".tex"):
            continue
        p = os.path.join(tabdir, f)
        out = []
        for line in open(p).read().split("\n"):
            m = re.match(r"\\multicolumn\{(\d+)\}\{l\}\{(.*)\} \\\\$", line)
            if m:
                line = m.group(2) + " &" * (int(m.group(1)) - 1) + " \\\\"
            out.append(line)
        txt = "\n".join(out).rstrip()
        if txt.endswith("\\\\"):
            txt = txt[:-2].rstrip()           # last row's \\ is supplied by main.tex after \input
        open(p, "w").write(txt)


def key_numbers(df, out):
    res = {}
    par = df[df.role == "parent"]
    for system, p in par.groupby("system"):
        ok = p[p.parent_ok]
        res[system] = dict(parents=len(p), valid_parents=len(ok),
                           parent_lle_mean_valid=float(ok.lle.mean()), parent_lle_sd_valid=float(ok.lle.std()),
                           parent_mse_median=float(p.mse.median()),
                           parent_mse_min=float(p.mse.min()), parent_mse_max=float(p.mse.max()))
    nonp = df[(df.role != "parent") & df.parent_ok]
    for system, d in nonp.groupby("system"):
        res[system].update(models=len(d), diverged=float(d.diverged.mean()),
                           collapsed=float((d.collapsed & d.alive).mean()),
                           usable_lle=float(d.valid.mean()))
        res[system]["n_models_valid_parents"] = int(len(d))
    json.dump(res, open(os.path.join(out, "key_numbers.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))



# ---------------------------------------------------- upgrade: robustness, law, mechanism
def fig_robustness(dfx, rad, out):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    xs = np.logspace(-2.7, 0, 200)
    for ax, system in zip(axs, ["lorenz", "rossler"]):
        ds = dfx[dfx.parent_ok & (dfx.role != "parent") & (dfx.system == system)].copy()
        ds["fail"] = ((~ds.alive) | ds.collapsed).astype(float)
        iso = ds[(ds.role.isin(["control", "grid"])) & (ds.family == "noise_iso")]
        a, b = logistic_fit(np.log10(iso.rel_norm.values), iso["fail"].values)
        ax.plot(xs, 1 / (1 + np.exp(-(a + b * np.log10(xs)))), "k-", lw=1.4, label="isotropic noise (logistic fit)")
        grid = ds[ds.role == "grid"].groupby("level").agg(x=("rel_norm", "mean"), f=("fail", "mean"))
        ax.plot(grid.x, grid.f, "ko", ms=3, label="isotropic noise (grid)")
        r = rad[(rad.system == system) & (rad.outcome.str.startswith("fail")) & (rad.p == 0.5)].iloc[0]
        ax.axvspan(r.ci_lo, r.ci_hi, color="gray", alpha=0.25)
        ax.axvline(r.radius, color="gray", ls="--", lw=1)
        ax.text(r.radius * 1.08, 0.04, f"$r_{{50}}$={r.radius:.3f}", fontsize=8)
        for fam, (lab, mk, col) in FAM_STYLE.items():
            g = ds[(ds.role == "primary") & (ds.family == fam)].groupby("level").agg(x=("rel_norm", "mean"), f=("fail", "mean"))
            ax.scatter(g.x, g.f, marker=mk, c=col, s=34, label=lab, edgecolor="k", lw=0.4, zorder=3)
        ax.set_xscale("log"); ax.set_title(SYS_PLOT[system]); ax.grid(alpha=0.25)
        ax.set_xlabel("relative parameter displacement $\\|\\Delta\\theta\\|/\\|\\theta\\|$")
    axs[0].set_ylabel("fraction of perturbed models that fail")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(out, "fig_robustness.pdf"))
    plt.close(fig)


def fig_vpt_law(dfx, law, lam_ref, out):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, system in zip(axs, ["lorenz", "rossler"]):
        d = dfx[dfx.parent_ok & (dfx.system == system)]
        lm = np.log10(d.mse)
        ax.axvspan(-6, -3, color="gray", alpha=0.12, lw=0)
        ax.scatter(lm, d.vpt_lt, s=3, alpha=0.2, c="tab:blue", lw=0)
        r = law[law.system == system].iloc[0]
        xx = np.linspace(-6, -3, 50)
        yy = (r.intercept + r.slope_time_per_ln * np.log(10 ** xx)) * lam_ref[system]
        ax.plot(xx, yy, "r-", lw=1.8)
        ax.set_xlim(-7.2, -0.8)
        ax.set_title(f"{SYS_PLOT[system]}: window fit $R^2$={r.R2:.2f}, slope {r.slope_time_per_ln * np.log(10) * lam_ref[system]:.2f} Lyap. times/decade", fontsize=8.5)
        ax.set_xlabel("$\\log_{10}$ one-step MSE"); ax.grid(alpha=0.25)
    axs[0].set_ylabel("valid prediction time (Lyapunov times)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_vpt_law.pdf"))
    plt.close(fig)


def table_extra(an, lam_ref, out):
    rad = pd.read_csv(os.path.join(an, "extra_robustness_radius.csv"))
    law = pd.read_csv(os.path.join(an, "extra_vpt_law.csv"))
    pp = pd.read_csv(os.path.join(an, "extra_vpt_law_per_parent.csv"))
    loo = pd.read_csv(os.path.join(an, "extra_vpt_law_loo.csv"))
    sh = pd.read_csv(os.path.join(an, "extra_layer_shares.csv"))
    cv = pd.read_csv(os.path.join(an, "extra_layer_resolved_cv.csv"))
    tr = pd.read_csv(os.path.join(an, "extra_tensor_ratio.csv"))

    def g2(x):
        return f"{x:.2g}"

    L = []
    for system in ["lorenz", "rossler"]:
        cells = [SYS_NAME[system]]
        for p in (0.1, 0.5, 0.9):
            r = rad[(rad.system == system) & rad.outcome.str.startswith("fail") & (rad.p == p)].iloc[0]
            cells.append(f"{g2(r.radius)} [{g2(r.ci_lo)}, {g2(r.ci_hi)}]")
        r = rad[(rad.system == system) & (rad.outcome == "diverge only") & (rad.p == 0.5)].iloc[0]
        cells.append(f"{g2(r.radius)} [{g2(r.ci_lo)}, {g2(r.ci_hi)}]")
        L.append(" & ".join(cells) + " \\\\")
    open(os.path.join(out, "tab_radius.tex"), "w").write("\n".join(L))

    lawf = pd.read_csv(os.path.join(an, "extra_vpt_law_floor.csv"))
    L = []
    for system in ["lorenz", "rossler"]:
        r = law[law.system == system].iloc[0]
        rf = lawf[lawf.system == system].iloc[0]
        p = pp[pp.system == system].dropna(subset=["lam_eff"])
        rho = [stats.spearmanr(p.lam_eff, p[c])[0] for c in ["lle_parent", "orbit_tangent_rate", "smax_rate"]]
        for tag, rr, last in [("$\\log_{10}$MSE in $[-6,-3]$", r, f"{rho[0]:+.2f}, {rho[1]:+.2f}, {rho[2]:+.2f}"),
                              ("VPT $>0.05$ (sensitivity)", rf, "--")]:
            pr = f"{p.R2.min():.2f}--{p.R2.max():.2f}" if rr is r else "--"
            slope_dec = rr.slope_time_per_ln * np.log(10) * lam_ref[system]
            L.append(" & ".join([SYS_NAME[system], tag, str(int(rr.n)), f"{rr.R2:.2f}", pr, f"{slope_dec:.2f}",
                                 f"{rr.lam_eff:.2f}", f"{rr.lam_eff_over_lam_ref:.1f}", last]) + " \\\\")
    open(os.path.join(out, "tab_law.tex"), "w").write("\n".join(L))

    L = []
    names = {"M0 constant slope": "Constant slope per system",
             "M1 lam = parent LLE": "Slope from parent Lyapunov exponent",
             "M2 lam = Jacobian smax rate": "Slope from parent Jacobian expansion rate"}
    for m, nm in names.items():
        a = loo[(loo.system == "lorenz") & (loo.model == m)].loo_R2.iloc[0]
        b = loo[(loo.system == "rossler") & (loo.model == m)].loo_R2.iloc[0]
        L.append(f"{nm} & {a:.2f} & {b:.2f} \\\\")
    open(os.path.join(out, "tab_law_loo.tex"), "w").write("\n".join(L))

    L = []
    for system in ["lorenz", "rossler"]:
        L.append(f"\\textit{{{SYS_NAME[system]}}} & & & & \\\\")
        for fam, nm in [("noise_iso", "Isotropic noise"), ("noise_layer", "Layer-matched noise"),
                        ("prune_global", "Global pruning"), ("prune_layer", "Layer-wise pruning"),
                        ("quant", "Quantization")]:
            r = sh[(sh.system == system) & (sh.fam == fam)].iloc[0]
            bias = r.sh_b1 + r.sh_b2 + r.sh_b3
            L.append(f"{nm} & {100 * r.sh_W1:.1f} & {100 * r.sh_W2:.1f} & {100 * r.sh_W3:.1f} & {100 * bias:.1f} \\\\")
    open(os.path.join(out, "tab_shares.tex"), "w").write("\n".join(L))

    L = []
    order = ["scalar norm", "scalar norm + family", "per-tensor norms", "per-tensor norms + family",
             "scalar norm + relative MSE", "per-tensor + family + relative MSE"]
    for f in order:
        c = [f]
        for system in ["lorenz", "rossler"]:
            r = cv[(cv.system == system) & (cv.features == f)].iloc[0]
            c += [f"{r.R2_vpt_lost:.3f}", f"{r.AUC_failure:.3f}"]
        L.append(" & ".join(c) + " \\\\")
    open(os.path.join(out, "tab_cv.tex"), "w").write("\n".join(L))
    return rad, law


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_full_30seeds")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--max_half_diff", type=float, default=0.25)
    args = ap.parse_args()
    figdir, tabdir = os.path.join(args.out, "figures"), os.path.join(args.out, "tables")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)
    df = load(args.results, args.max_half_diff)
    an = os.path.join(args.results, "analysis")
    pred = pd.read_csv(os.path.join(an, "predictivity.csv"))
    auc = pd.read_csv(os.path.join(an, "divergence_auc.csv"))
    pair = pd.read_csv(os.path.join(an, "paired_unconditional.csv"))
    # VPT-loss predictivity uses ALL perturbed models (no survival filter); LLE and SW1 targets
    # are only defined for non-failed models and stay survival-conditional.
    pv = pd.read_csv(os.path.join(an, "predictivity_vpt_all.csv"))
    pred = pd.concat([pred[pred.target != "dVPT"], pv], ignore_index=True)

    fig_failure(df, figdir)
    fig_scatter_vpt(df, figdir)
    fig_predictivity(pred, figdir)
    fig_forest(pair, figdir)
    fig_slin_validity(figdir)
    table_failure(df, tabdir)
    table_predict(pred, tabdir)
    table_auc(auc, tabdir)
    table_paired(pair, tabdir)
    dfx = attach_extra(df, args.results)
    lam_ref = json.load(open(os.path.join(args.results, "reference_lle.json")))
    rad, law = table_extra(an, lam_ref, tabdir)
    fig_robustness(dfx, rad, figdir)
    fig_vpt_law(dfx, law, lam_ref, figdir)
    fix_headers(tabdir)
    key_numbers(df, args.out)


if __name__ == "__main__":
    main()
