# How much compression can a learned chaotic surrogate survive?
### A matched-control study of pruning, quantization and attractor failure in Lorenz-63 and Rössler emulators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23157081.svg)](https://doi.org/10.5281/zenodo.23157081)

Author: Viren Singh ([ORCID 0009-0001-6618-0126](https://orcid.org/0009-0001-6618-0126))

Code, raw results and manuscript source for a controlled study of how pruning and quantization
affect learned surrogates of the Lorenz-63 and Rössler flows. Every structured perturbation is paired
with random perturbations of exactly the same total relative norm, the same per-tensor norms, and
(for pruning) the same support, so the roles of magnitude, distribution across layers, and which
entries are touched can be separated.

Pure NumPy / SciPy / pandas / Matplotlib. No GPU or deep-learning framework needed.

## Main findings (30 seeds per system, 21,180 perturbed models)
* **Failure dominates.** 68% (Lorenz) and 79% (Rössler) of perturbed surrogates diverge or collapse
  to a near-constant state, so Lyapunov-exponent comparisons are strongly survival-conditional.
* **Robustness radius.** Half of isotropic-noise perturbations destroy the attractor at a relative
  parameter displacement of 4.4% (Lorenz) and 1.8% (Rössler); 6-bit quantization sits at this scale.
* **Equal norm is not equal dynamics, but the differences are modest.** Against isotropic noise of
  the same norm, structured perturbations have longer valid prediction time (17/20 comparisons)
  and lower one-step error (18/20); matching per-tensor norms leaves 5/20 and 5/20 significant.
  Isotropic noise displaces the small output layer 2.5-2.8x more in relative terms. Failure *mode*
  differs (global 50% pruning diverges in 80% of Lorenz surrogates; matched noise mostly collapses them).
* **Failure is rarely silent.** 98-99% of failed surrogates already have one-step error at least
  10x that of their parent (a post hoc, cross-sectional test of hypothesis H1; `analyze_h1.py`).
  High one-step error is sensitive but not specific, since ~3/4 of survivors exceed it too, and
  a minority of Rössler models fail at near-parent error.
* **Predictors.** Norm orders damage across scales; parent-relative one-step error adds the most
  predictive power for valid-prediction-time loss (cross-validated R2 0.77 -> 0.85 Lorenz,
  0.74 -> 0.88 Rössler); a trajectory-weighted Jacobian-change measure does not improve on it.
  Valid prediction time is approximately log-linear in one-step error (R2 0.64 / 0.79 in a window
  of moderate errors) with a slope not explained by Lyapunov or Jacobian statistics.
  Long-run Lyapunov-exponent change is poorly predicted by every quantity tested, and first-order
  Lyapunov response is valid only for relative displacements below about 1e-3.

Scope and limitations: two 3-D systems, one small residual MLP (3-64-64-3), post-training
compression only, valid prediction time is a short-horizon metric, and the Rössler surrogates are
only weakly chaotic. See `paper/main.tex` for details.

## Quick start
```bash
pip install -r requirements.txt
python tests/test_core.py        # 8 correctness tests (gradients, Jacobians, matched norms, ...)
python sanity_check.py           # Jacobian and first-order-response checks (~1 min)
```

## Reproduce the experiment (~1 hour on one CPU core)
```bash
python run_experiment.py --systems lorenz rossler --seeds 30 --K 10 --out results
python analyze.py --results results
python analyze_paired.py --results results
python recompute_extra.py --results results     # rebuilds parents (~10 min); adds per-tensor displacements
python analyze_extra.py --results results       # robustness radius, VPT law, layer-resolved checks
python analyze_h1.py --results results --out paper   # failure vs parent-relative one-step error
python make_paper_assets.py --results results --out paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
The runner is resumable: finished (system, seed) pairs are skipped on re-run. The raw per-model
results used in the paper are already included in `results_full_30seeds/`, so you can run the
analysis and figure scripts without re-running the experiment:
```bash
python analyze.py --results results_full_30seeds
python analyze_paired.py --results results_full_30seeds
python analyze_extra.py --results results_full_30seeds
python analyze_h1.py --results results_full_30seeds --out paper
python make_paper_assets.py --results results_full_30seeds --out paper
```

## Layout
```
jacsens/
  systems.py       Lorenz-63 / Rössler, RK4 integration, reference Lyapunov exponent
  model.py         residual MLP, manual backprop + Adam, analytic and batched Jacobians
  perturb.py       pruning, quantization, matched random controls
  dynamics.py      batched LLE, valid prediction time, sliced Wasserstein distance
  sensitivity.py   trajectory-weighted Jacobian-change measures, first-order LLE response
run_experiment.py  recompute_extra.py  analyze.py  analyze_paired.py  analyze_extra.py  analyze_h1.py
make_paper_assets.py  sanity_check.py
tests/test_core.py
results_full_30seeds/{raw,extra,analysis,config.json,reference_lle.json}
paper/{main.tex,refs.bib,figures,tables}
```

## What is measured
Per model: one-step test MSE; largest Lyapunov exponent (analytic Jacobians, tangent-vector
renormalisation, first/second-half estimates as a stability check); valid prediction time in
Lyapunov times; sliced-Wasserstein distance to the true invariant measure; divergence and collapse
flags; and the Jacobian-change measures `S_F`, `S_2`, `S_lin`, `S_lin0` (see
`jacsens/sensitivity.py` for definitions and the first-order derivation).

## License
Code: MIT (see `LICENSE`).

## Citation
See `CITATION.cff`. Archived release: [doi:10.5281/zenodo.23157081](https://doi.org/10.5281/zenodo.23157081).
