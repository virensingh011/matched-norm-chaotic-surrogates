# Analysis summary

LLE estimates kept only if rollout finite, not collapsed, and half-run disagreement <= 0.25. Intervals: bootstrap over seeds.

## Parents
| system | seeds | parent_ok | parent_lle_mean | parent_mse_median |
|---|---|---|---|---|
| lorenz | 30 | 1 | 0.863 | 4.94e-06 |
| rossler | 30 | 0.733 | 0.0306 | 1.55e-06 |

## Model inventory (non-parent)
| system | models | valid | diverged | collapsed |
|---|---|---|---|---|
| lorenz | 10590 | 0.303 | 0.143 | 0.541 |
| rossler | 10590 | 0.199 | 0.595 | 0.207 |

## A. Structured perturbation vs matched controls (ΔLLE; negative diff = structured more contracting)
Each row pairs a structured perturbation with one matched-norm control family. `mean_percentile` is where the structured model's ΔLLE falls among the K control draws (0.5 = indistinguishable from noise).

| system | structured | control | rel_norm | n_seeds | med_dLLE_diff | ci_lo | ci_hi | wilcoxon_p | med_abs_diff | mean_percentile | pct_lo | pct_hi | frac_seeds_structured_lower | diverge_rate_structured | diverge_rate_control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lorenz | prune_global@0.25 | noise_iso | 0.102 | 1 | 0.186 | n/a | n/a | n/a | -0.186 | 0.667 | n/a | n/a | 0 | 0.9 | 0.76 |
| lorenz | prune_global@0.25 | noise_layer | 0.102 | 2 | 0.00493 | -0.0171 | 0.027 | n/a | -0.00493 | 0.375 | 0.25 | 0.5 | 0.5 | 0.9 | 0.747 |
| lorenz | prune_global@0.25 | noise_support | 0.102 | 1 | -0.0101 | n/a | n/a | n/a | 0.0101 | 0.2 | n/a | n/a | 1 | 0.9 | 0.757 |
| lorenz | prune_global@0.5 | noise_iso | 0.296 | 3 | -0.0182 | -0.0227 | -0.0106 | n/a | 0.0182 | 0 | 0 | 0 | 1 | 0.8 | 0.767 |
| lorenz | prune_global@0.5 | noise_layer | 0.296 | 3 | -0.011 | -0.0117 | -0.000739 | n/a | 0.011 | 0.167 | 0 | 0.333 | 1 | 0.8 | 0.717 |
| lorenz | prune_global@0.5 | noise_support | 0.296 | 1 | -0.00809 | n/a | n/a | n/a | 0.00809 | 0 | n/a | n/a | 1 | 0.8 | 0.817 |
| lorenz | prune_layer@0.25 | noise_iso | 0.108 | 5 | 0.188 | -0.271 | 0.715 | n/a | -0.188 | 0.74 | 0.56 | 0.92 | 0.2 | 0.633 | 0.727 |
| lorenz | prune_layer@0.25 | noise_layer | 0.108 | 8 | 0.0554 | -0.553 | 0.582 | 0.844 | -0.0157 | 0.543 | 0.25 | 0.816 | 0.5 | 0.633 | 0.697 |
| lorenz | prune_layer@0.25 | noise_support | 0.108 | 6 | 0.158 | -0.398 | 0.652 | 0.562 | -0.0908 | 0.667 | 0.333 | 0.944 | 0.333 | 0.633 | 0.637 |
| lorenz | prune_layer@0.5 | noise_iso | 0.306 | 2 | 0.31 | -0.00725 | 0.628 | n/a | -0.31 | 0.625 | 0.25 | 1 | 0.5 | 0.933 | 0.747 |
| lorenz | prune_layer@0.5 | noise_layer | 0.306 | 1 | 0.623 | n/a | n/a | n/a | -0.623 | 0.75 | n/a | n/a | 0 | 0.933 | 0.683 |
| lorenz | prune_layer@0.5 | noise_support | 0.306 | 1 | -0.00832 | n/a | n/a | n/a | 0.00832 | 0.2 | n/a | n/a | 1 | 0.933 | 0.753 |
| lorenz | prune_layer@0.75 | noise_iso | 0.568 | 1 | -0.0103 | n/a | n/a | n/a | -0.0103 | 0.333 | n/a | n/a | 1 | 0.933 | 0.873 |
| lorenz | quant@3 | noise_iso | 0.402 | 2 | 0.491 | 0.00665 | 0.976 | n/a | -0.357 | 1 | 1 | 1 | 0 | 0.733 | 0.82 |
| lorenz | quant@3 | noise_layer | 0.402 | 5 | 0.00471 | -0.0162 | 0.402 | n/a | -0.00471 | 0.613 | 0.28 | 0.933 | 0.4 | 0.733 | 0.743 |
| lorenz | quant@4 | noise_iso | 0.172 | 6 | -0.00571 | -0.252 | 0.774 | 0.844 | 0.00571 | 0.458 | 0.167 | 0.75 | 0.5 | 0.7 | 0.747 |
| lorenz | quant@4 | noise_layer | 0.172 | 4 | -0.32 | -0.893 | 0.413 | n/a | 0.306 | 0.5 | 0 | 1 | 0.5 | 0.7 | 0.683 |
| lorenz | quant@6 | noise_iso | 0.0386 | 12 | -0.015 | -0.185 | 0.194 | 0.97 | -0.0471 | 0.53 | 0.354 | 0.722 | 0.5 | 0.467 | 0.653 |
| lorenz | quant@6 | noise_layer | 0.0386 | 12 | -0.0326 | -0.122 | 0.0627 | 0.38 | 0.00848 | 0.509 | 0.316 | 0.686 | 0.583 | 0.467 | 0.383 |
| lorenz | quant@8 | noise_iso | 0.00945 | 26 | 0.0189 | -0.00132 | 0.0362 | 0.237 | -0.0203 | 0.598 | 0.493 | 0.701 | 0.308 | 0.0667 | 0.103 |
| lorenz | quant@8 | noise_layer | 0.00945 | 26 | 0.00594 | -0.022 | 0.0152 | 0.803 | -0.00498 | 0.509 | 0.396 | 0.624 | 0.385 | 0.0667 | 0.0133 |
| rossler | prune_global@0.25 | noise_layer | 0.101 | 1 | 0.000394 | n/a | n/a | n/a | -0.0142 | 0.5 | n/a | n/a | 0 | 0.955 | 0.882 |
| rossler | prune_layer@0.25 | noise_iso | 0.117 | 2 | 0.0123 | 0.00882 | 0.0157 | n/a | -0.0112 | 0.708 | 0.667 | 0.75 | 0 | 0.455 | 0.859 |
| rossler | prune_layer@0.25 | noise_layer | 0.117 | 11 | -0.000351 | -0.0101 | 0.0495 | 0.898 | -0.00647 | 0.546 | 0.327 | 0.76 | 0.545 | 0.455 | 0.559 |
| rossler | prune_layer@0.25 | noise_support | 0.117 | 12 | -0.00665 | -0.0503 | 0.00306 | 0.38 | -0.00677 | 0.357 | 0.165 | 0.571 | 0.75 | 0.455 | 0.482 |
| rossler | prune_layer@0.5 | noise_support | 0.334 | 1 | -0.0954 | n/a | n/a | n/a | 0.0909 | 0 | n/a | n/a | 1 | 0.818 | 0.841 |
| rossler | quant@4 | noise_layer | 0.122 | 6 | -0.0185 | -0.0507 | 0.0333 | 0.562 | -0.0116 | 0.444 | 0.222 | 0.722 | 0.667 | 0.682 | 0.755 |
| rossler | quant@6 | noise_iso | 0.0276 | 7 | 0.00271 | -0.00532 | 0.0146 | 0.688 | 0.00379 | 0.476 | 0.214 | 0.738 | 0.429 | 0.545 | 0.645 |
| rossler | quant@6 | noise_layer | 0.0276 | 8 | 0.000732 | -0.00571 | 0.107 | 0.547 | 0.000199 | 0.599 | 0.382 | 0.821 | 0.375 | 0.545 | 0.418 |
| rossler | quant@8 | noise_iso | 0.0067 | 20 | 0.00461 | -0.00175 | 0.0446 | 0.105 | -0.0044 | 0.565 | 0.415 | 0.7 | 0.45 | 0.0909 | 0.227 |
| rossler | quant@8 | noise_layer | 0.0067 | 19 | -0.00197 | -0.00767 | 0.0344 | 0.623 | -0.00389 | 0.485 | 0.318 | 0.648 | 0.579 | 0.0909 | 0.132 |

## B. Does the predictor rank dynamical damage? (Spearman ρ)
`seed_rho_mean` = mean within-parent correlation (the honest one: removes between-seed variation). `partial_rho_given_norm` = within-parent correlation after removing the rank-trend in rel_norm — the key column for the claim that sensitivity adds information beyond norm.

### Target: |dLLE|
| system | predictor | n | pooled_rho | seed_rho_mean | seed_rho_lo | seed_rho_hi | partial_rho_given_norm |
|---|---|---|---|---|---|---|---|
| lorenz | rel_norm | 3208 | 0.633 | 0.613 | 0.527 | 0.672 | 0.000579 |
| lorenz | S_F_rel | 3208 | 0.657 | 0.637 | 0.539 | 0.701 | 0.237 |
| lorenz | S_2 | 3208 | 0.656 | 0.637 | 0.544 | 0.699 | 0.235 |
| lorenz | absS_lin | 3208 | 0.555 | 0.541 | 0.462 | 0.599 | 0.108 |
| lorenz | absS_lin0 | 3208 | 0.536 | 0.525 | 0.448 | 0.578 | 0.0719 |
| lorenz | absdMSE | 3208 | 0.665 | 0.648 | 0.546 | 0.709 | 0.286 |
| rossler | rel_norm | 1634 | 0.0681 | 0.0841 | -0.01 | 0.171 | 0.00248 |
| rossler | S_F_rel | 1634 | 0.09 | 0.0867 | -0.0127 | 0.181 | 0.0302 |
| rossler | S_2 | 1634 | 0.0907 | 0.0878 | -0.00163 | 0.18 | 0.0329 |
| rossler | absS_lin | 1634 | 0.0778 | 0.104 | 0.0172 | 0.184 | 0.0725 |
| rossler | absS_lin0 | 1634 | 0.0828 | 0.0748 | -0.00693 | 0.161 | 0.0274 |
| rossler | absdMSE | 1634 | 0.0721 | 0.0746 | -0.0224 | 0.165 | 0.00192 |

### Target: VPT loss (Lyap. times)
| system | predictor | n | pooled_rho | seed_rho_mean | seed_rho_lo | seed_rho_hi | partial_rho_given_norm |
|---|---|---|---|---|---|---|---|
| lorenz | rel_norm | 3208 | 0.681 | 0.904 | 0.898 | 0.91 | -0.0061 |
| lorenz | S_F_rel | 3208 | 0.695 | 0.94 | 0.936 | 0.944 | 0.605 |
| lorenz | S_2 | 3208 | 0.694 | 0.938 | 0.934 | 0.942 | 0.592 |
| lorenz | absS_lin | 3208 | 0.583 | 0.782 | 0.769 | 0.796 | 0.219 |
| lorenz | absS_lin0 | 3208 | 0.607 | 0.8 | 0.786 | 0.812 | 0.276 |
| lorenz | absdMSE | 3208 | 0.706 | 0.975 | 0.973 | 0.977 | 0.854 |
| rossler | rel_norm | 1634 | 0.667 | 0.831 | 0.815 | 0.846 | -0.0537 |
| rossler | S_F_rel | 1634 | 0.756 | 0.925 | 0.898 | 0.943 | 0.739 |
| rossler | S_2 | 1634 | 0.753 | 0.921 | 0.891 | 0.939 | 0.722 |
| rossler | absS_lin | 1634 | 0.624 | 0.794 | 0.767 | 0.817 | 0.45 |
| rossler | absS_lin0 | 1634 | 0.61 | 0.767 | 0.734 | 0.798 | 0.457 |
| rossler | absdMSE | 1634 | 0.767 | 0.947 | 0.937 | 0.955 | 0.825 |

### Target: invariant-measure shift (SW1)
| system | predictor | n | pooled_rho | seed_rho_mean | seed_rho_lo | seed_rho_hi | partial_rho_given_norm |
|---|---|---|---|---|---|---|---|
| lorenz | rel_norm | 3208 | 0.739 | 0.755 | 0.734 | 0.775 | 0.00197 |
| lorenz | S_F_rel | 3208 | 0.766 | 0.787 | 0.77 | 0.804 | 0.343 |
| lorenz | S_2 | 3208 | 0.764 | 0.785 | 0.768 | 0.802 | 0.335 |
| lorenz | absS_lin | 3208 | 0.627 | 0.656 | 0.632 | 0.68 | 0.127 |
| lorenz | absS_lin0 | 3208 | 0.623 | 0.655 | 0.629 | 0.679 | 0.12 |
| lorenz | absdMSE | 3208 | 0.783 | 0.806 | 0.79 | 0.822 | 0.435 |
| rossler | rel_norm | 1634 | 0.555 | 0.61 | 0.535 | 0.675 | -0.0254 |
| rossler | S_F_rel | 1634 | 0.663 | 0.68 | 0.591 | 0.743 | 0.398 |
| rossler | S_2 | 1634 | 0.658 | 0.678 | 0.594 | 0.742 | 0.389 |
| rossler | absS_lin | 1634 | 0.547 | 0.587 | 0.528 | 0.632 | 0.249 |
| rossler | absS_lin0 | 1634 | 0.597 | 0.588 | 0.509 | 0.651 | 0.288 |
| rossler | absdMSE | 1634 | 0.643 | 0.686 | 0.61 | 0.744 | 0.412 |

## C. First-order sign/size check of S_lin vs measured ΔLLE
| system | subset | n | pearson | spearman | sign_agree |
|---|---|---|---|---|---|
| lorenz | all | 3208 | 0.175 | 0.162 | 0.544 |
| lorenz | rel_norm<=0.1 | 1473 | 0.0453 | 0.103 | 0.519 |
| rossler | all | 1634 | 0.0313 | 0.164 | 0.583 |
| rossler | rel_norm<=0.1 | 960 | 0.103 | 0.245 | 0.596 |

## D. Predicting divergence/collapse (AUC; 0.5 = chance)
| system | predictor | n | failure_rate | auc |
|---|---|---|---|---|
| lorenz | rel_norm | 10590 | 0.684 | 0.727 |
| lorenz | S_F_rel | 10590 | 0.684 | 0.735 |
| lorenz | S_2 | 10590 | 0.684 | 0.733 |
| lorenz | absS_lin | 10590 | 0.684 | 0.723 |
| lorenz | absS_lin0 | 10590 | 0.684 | 0.702 |
| lorenz | absdMSE | 10590 | 0.684 | 0.737 |
| rossler | rel_norm | 7766 | 0.79 | 0.787 |
| rossler | S_F_rel | 7766 | 0.79 | 0.831 |
| rossler | S_2 | 7766 | 0.79 | 0.829 |
| rossler | absS_lin | 7766 | 0.79 | 0.77 |
| rossler | absS_lin0 | 7766 | 0.79 | 0.811 |
| rossler | absdMSE | 7766 | 0.79 | 0.838 |
