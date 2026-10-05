"""Structured perturbations (pruning, quantization) and matched random controls.

Relative displacement is ||theta' - theta|| / ||theta|| over ALL parameters
(weights + biases). Pruning touches weights only; quantization touches all tensors.

Control families for a structured perturbation with displacement `delta`:
  noise_iso      isotropic Gaussian, same total norm             (the paper's control)
  noise_layer    Gaussian per tensor, same per-tensor norms      (layer-wise matched)
  noise_support  Gaussian only on entries `delta` touched, same per-tensor norms
                 (isolates *which values* are written, holding the support fixed)
"""
import numpy as np

from .model import KEYS, WKEYS, copy_params


def flat(p):
    return np.concatenate([p[k].ravel() for k in KEYS])


def delta_of(p, q):
    return {k: q[k] - p[k] for k in KEYS}


def rel_norm(p, q):
    return float(np.linalg.norm(flat(q) - flat(p)) / np.linalg.norm(flat(p)))


def add(p, delta):
    return {k: p[k] + delta[k] for k in KEYS}


# ---------------------------------------------------------------- pruning
def prune_global(p, sparsity):
    q = copy_params(p)
    allw = np.concatenate([np.abs(p[k]).ravel() for k in WKEYS])
    kth = int(round(sparsity * allw.size))
    if kth == 0:
        return q
    order = np.argsort(allw, kind="stable")[:kth]
    mask = np.ones(allw.size, bool)
    mask[order] = False
    i = 0
    for k in WKEYS:
        n = p[k].size
        q[k] = (p[k].ravel() * mask[i:i + n]).reshape(p[k].shape)
        i += n
    return q


def prune_layer(p, sparsity):
    q = copy_params(p)
    for k in WKEYS:
        w = np.abs(p[k]).ravel()
        kth = int(round(sparsity * w.size))
        mask = np.ones(w.size, bool)
        mask[np.argsort(w, kind="stable")[:kth]] = False
        q[k] = (p[k].ravel() * mask).reshape(p[k].shape)
    return q


def prune_random(p, sparsity, rng):
    q = copy_params(p)
    for k in WKEYS:
        mask = rng.random(p[k].shape) >= sparsity
        q[k] = p[k] * mask
    return q


# ------------------------------------------------------------ quantization
def quantize(p, bits):
    """Post-training per-tensor symmetric uniform quantization (weights and biases)."""
    q = {}
    qmax = 2 ** (bits - 1) - 1
    for k in KEYS:
        amax = np.abs(p[k]).max()
        scale = amax / qmax if amax > 0 else 1.0
        q[k] = np.clip(np.round(p[k] / scale), -qmax, qmax) * scale
    return q


# --------------------------------------------------------- matched controls
def noise_iso(p, rel, rng):
    z = rng.normal(size=flat(p).size)
    z *= rel * np.linalg.norm(flat(p)) / np.linalg.norm(z)
    out, i = {}, 0
    for k in KEYS:
        n = p[k].size
        out[k] = p[k] + z[i:i + n].reshape(p[k].shape)
        i += n
    return out


def _scaled_like(ref_delta, rng, support_only):
    d = {}
    for k in KEYS:
        nk = np.linalg.norm(ref_delta[k])
        if nk == 0:
            d[k] = np.zeros_like(ref_delta[k])
            continue
        z = rng.normal(size=ref_delta[k].shape)
        if support_only:
            z = z * (ref_delta[k] != 0)
        nz = np.linalg.norm(z)
        d[k] = z * (nk / nz) if nz > 0 else np.zeros_like(z)
    return d


def noise_layer(p, ref_delta, rng):
    return add(p, _scaled_like(ref_delta, rng, support_only=False))


def noise_support(p, ref_delta, rng):
    return add(p, _scaled_like(ref_delta, rng, support_only=True))


# ------------------------------------------------------------ the full set
def make_perturbations(parent, rng, sparsities=(0.25, 0.5, 0.75), bits=(8, 6, 4, 3),
                       K=10, grid_norms=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6)):
    """Return a list of dicts: role/family/level/group/control_of/draw/params."""
    out = []

    def rec(role, family, level, group, control_of, draw, params):
        out.append(dict(role=role, family=family, level=level, group=group,
                        control_of=control_of, draw=draw, params=params,
                        rel_norm=rel_norm(parent, params)))

    primaries = []
    for s in sparsities:
        primaries.append(("prune_global", s, prune_global(parent, s)))
        primaries.append(("prune_layer", s, prune_layer(parent, s)))
    for b in bits:
        primaries.append(("quant", b, quantize(parent, b)))

    for fam, lvl, q in primaries:
        g = f"{fam}@{lvl}"
        rec("primary", fam, lvl, g, "", 0, q)
        rn = rel_norm(parent, q)
        dl = delta_of(parent, q)
        for k in range(K):
            rec("control", "noise_iso", lvl, f"noise_iso[{g}]", g, k, noise_iso(parent, rn, rng))
            rec("control", "noise_layer", lvl, f"noise_layer[{g}]", g, k,
                noise_layer(parent, dl, rng))
            if fam.startswith("prune"):
                rec("control", "noise_support", lvl, f"noise_support[{g}]", g, k,
                    noise_support(parent, dl, rng))

    for s in sparsities:                      # random pruning at equal sparsity (own family)
        rec("primary", "prune_random", s, f"prune_random@{s}", "", 0, prune_random(parent, s, rng))

    for rn in grid_norms:                     # calibration cloud for predictivity analysis
        for k in range(K):
            rec("grid", "noise_iso", rn, f"grid@{rn}", "", k, noise_iso(parent, rn, rng))
    return out
