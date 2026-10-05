"""Trajectory-weighted Jacobian sensitivity of a perturbation.

Setting: parent map f_theta, perturbed map f_theta' with Delta J(x) = J_theta'(x) - J_theta(x),
evaluated along a reference orbit x_0..x_{T-1} (default: the parent's OWN autonomous orbit,
i.e. a sample of its invariant measure, which is what linear-response theory calls for).

Unsigned size-of-change proxies
  S_F      = mean_t ||Delta J(x_t)||_F
  S_F_rel  = S_F / mean_t ||J_theta(x_t)||_F          (scale-free)
  S_2      = mean_t ||Delta J(x_t)||_2

Signed first-order prediction of Delta LLE
  Parent tangent recursion along the orbit:   n_t = ||J_t u_t||,  v = J_t u_t / n_t,  u_{t+1} = v.
  Perturbed:  u'_t = u_t + du_t.  To first order
      log ||J'_t u'_t||  =  log n_t + v^T (dJ_t u_t + J_t du_t) / n_t
      du_{t+1}           =  (I - v v^T)(dJ_t u_t + J_t du_t) / n_t          (direction response)
  so   S_lin  = (1/dt) mean_t  v^T (dJ_t u_t + J_t du_t) / n_t .

  S_lin0 drops the J_t du_t term (tangent direction frozen). It is NOT a consistent first-order
  quantity -- the Lyapunov direction rotates at the same order as dJ -- and is kept only so
  the difference is visible (sanity_check.py, test 2, demonstrates this).

Caveat: S_lin is a linearisation about a FIXED orbit. It ignores the shift of the invariant
measure itself and the non-smoothness of the LLE under perturbation, so it is expected to
degrade for large perturbations; analyze.py quantifies where.
"""
import numpy as np

from .model import forward, jac_points


def rollout(p, x0, n, bound=50.0):
    """Autonomous orbit (n,d) of a single model; returns None if it leaves |x|<=bound."""
    x = x0.copy()
    out = np.empty((n, x0.size))
    for t in range(n):
        out[t] = x
        x = forward(p, x[None])[0]
        if not np.all(np.isfinite(x)) or np.abs(x).max() > bound:
            return None
    return out


def reference_orbit(parent, X_true, T, n_transient=1000, collapse_std=0.05, mode="parent"):
    """Parent's own orbit if it is finite and non-collapsed, else the true orbit X_true[:T]."""
    if mode == "parent":
        orb = rollout(parent, X_true[100], n_transient + T)
        if orb is not None and orb[n_transient:].std(0).mean() > collapse_std:
            return orb[n_transient:], "parent"
    return X_true[:T], "true"


class TrajectoryRef:
    """Parent Jacobians and tangent vectors along a reference orbit (computed once)."""

    def __init__(self, parent, X, dt, burn=300, seed=0):
        self.parent, self.X, self.dt, self.burn = parent, X, dt, burn
        T, d = X.shape
        J = jac_points(parent, X)
        u = np.random.default_rng(seed).normal(size=d)
        u /= np.linalg.norm(u)
        U, V, N = np.empty((T, d)), np.empty((T, d)), np.empty(T)
        for t in range(T):
            U[t] = u
            v = J[t] @ u
            N[t] = np.linalg.norm(v)
            V[t] = v / N[t]
            u = V[t]
        self.J, self.U, self.V, self.N = J, U, V, N
        self.J_fro = float(np.mean(np.linalg.norm(J, axis=(1, 2))))
        self.parent_orbit_lle = float(np.mean(np.log(N[burn:])) / dt)

    def metrics_batch(self, perts):
        """Sensitivity metrics for a list of perturbed parameter dicts."""
        T, d = self.X.shape
        M = len(perts)
        dJ = np.empty((T, M, d, d))
        for m, p in enumerate(perts):
            dJ[:, m] = jac_points(p, self.X) - self.J
        flat = dJ.transpose(1, 0, 2, 3)                              # (M,T,d,d)
        S_F = np.linalg.norm(flat, axis=(2, 3)).mean(1)
        S_2 = np.linalg.norm(flat, ord=2, axis=(2, 3)).mean(1)

        du = np.zeros((M, d))
        acc0, acc1 = np.zeros(M), np.zeros(M)
        for t in range(T):
            dJu = dJ[t] @ self.U[t]                                  # (M,d)
            w = dJu + du @ self.J[t].T
            if t >= self.burn:
                acc0 += dJu @ self.V[t] / self.N[t]
                acc1 += w @ self.V[t] / self.N[t]
            du = (w - np.outer(w @ self.V[t], self.V[t])) / self.N[t]
        k = (T - self.burn) * self.dt
        return [dict(S_F=S_F[m], S_F_rel=S_F[m] / self.J_fro, S_2=S_2[m],
                     S_lin=acc1[m] / k, S_lin0=acc0[m] / k) for m in range(M)]

    def metrics(self, pert):
        return self.metrics_batch([pert])[0]
