"""Long-horizon dynamical diagnostics for many learned maps at once."""
import numpy as np

from .model import step_and_jac_batch, step_batch_states


def lle_batch(P, x0, dt, n_transient=1000, n_steps=10000, bound=50.0,
              keep_stride=5, seed=0):
    """Largest Lyapunov exponent (per unit time) of M learned maps, from analytic
    Jacobians + tangent-vector renormalisation (Benettin).

    Returns dict with
      lle, lle_h1, lle_h2 : (M,)   full-run and first/second-half estimates
      alive               : (M,)   False if the rollout left |x|<=bound or went non-finite
      samples             : (M, n_steps//keep_stride, d)  post-transient states
    """
    M = P["W1"].shape[0]
    d = x0.shape[0]
    rng = np.random.default_rng(seed)
    x = np.tile(x0[None, :], (M, 1))
    alive = np.ones(M, bool)

    def check(x, alive):
        bad = ~np.isfinite(x).all(1) | (np.abs(x).max(1, initial=0) > bound)
        alive = alive & ~bad
        x = np.where(alive[:, None], x, 0.0)
        return x, alive

    for _ in range(n_transient):
        x, _J = step_and_jac_batch(P, x)
        x, alive = check(x, alive)

    u = rng.normal(size=(M, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    half = n_steps // 2
    logs = np.zeros((M, 2))
    n_keep = n_steps // keep_stride
    samples = np.zeros((M, n_keep, d))
    for t in range(n_steps):
        if t % keep_stride == 0 and t // keep_stride < n_keep:
            samples[:, t // keep_stride] = x
        xn, J = step_and_jac_batch(P, x)
        v = (J @ u[:, :, None])[:, :, 0]
        nrm = np.linalg.norm(v, axis=1)
        ok = alive & np.isfinite(nrm) & (nrm > 0)
        logs[:, 0 if t < half else 1] += np.where(ok, np.log(np.where(ok, nrm, 1.0)), 0.0)
        u = np.where(ok[:, None], v / np.where(ok, nrm, 1.0)[:, None], u)
        x, alive = check(xn, alive)
    lle_h1 = logs[:, 0] / (half * dt)
    lle_h2 = logs[:, 1] / ((n_steps - half) * dt)
    lle = logs.sum(1) / (n_steps * dt)
    for a in (lle, lle_h1, lle_h2):
        a[~alive] = np.nan
    return dict(lle=lle, lle_h1=lle_h1, lle_h2=lle_h2, alive=alive, samples=samples)


def sliced_w1(samples, ref, n_proj=32, seed=0, chunk=40):
    """Sliced 1-Wasserstein distance between each model's rollout samples (M,n,d)
    and a reference cloud (n,d); an invariant-measure distance (no binning)."""
    M, n, d = samples.shape
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rp = np.sort(ref @ dirs.T, axis=0)
    # equalise sample counts by quantile-subsampling the reference
    idx = np.linspace(0, len(ref) - 1, n).astype(int)
    rp = rp[idx]
    out = np.empty(M)
    for i in range(0, M, chunk):
        sp = np.sort(samples[i:i + chunk] @ dirs.T, axis=1)      # (c,n,P)
        out[i:i + chunk] = np.abs(sp - rp[None]).mean(axis=(1, 2))
    return out


def vpt_batch(P, traj, offsets, horizon, dt, thresh=0.4):
    """Valid prediction time (time units) of M models, averaged over initial conditions.

    traj    : (N,d) true standardised trajectory sampled every dt
    offsets : initial-condition indices into traj
    Error is ||x_hat - x|| / sqrt(d); standardised coords => unit-variance scale.
    """
    M = P["W1"].shape[0]
    d = traj.shape[1]
    segs = np.stack([traj[o:o + horizon + 1] for o in offsets])      # (nIC,H+1,d)
    S = np.tile(traj[offsets][None], (M, 1, 1))                      # (M,nIC,d)
    first = np.full((M, len(offsets)), horizon, dtype=float)
    done = np.zeros((M, len(offsets)), bool)
    for t in range(1, horizon + 1):
        S = step_batch_states(P, S)
        err = np.linalg.norm(S - segs[None, :, t], axis=2) / np.sqrt(d)
        exceed = ~done & (~np.isfinite(err) | (err > thresh))
        first[exceed] = t - 1
        done |= exceed
        S = np.where(np.isfinite(S), S, 1e6)
        S = np.clip(S, -1e6, 1e6)
    return first.mean(1) * dt
