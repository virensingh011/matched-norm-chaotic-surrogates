"""Residual MLP  f(x) = x + W3 tanh(W2 tanh(W1 x + b1) + b2) + b3  in pure numpy.

No torch dependency: manual backprop + Adam for training, analytic Jacobians,
and *batched* versions (leading axis M = many models) so hundreds of perturbed
models can be rolled out simultaneously.
"""
import numpy as np

KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")
WKEYS = ("W1", "W2", "W3")


def init_params(rng, d=3, H=64):
    def xavier(o, i):
        lim = np.sqrt(6.0 / (i + o))
        return rng.uniform(-lim, lim, (o, i))
    return {"W1": xavier(H, d), "b1": np.zeros(H),
            "W2": xavier(H, H), "b2": np.zeros(H),
            "W3": 0.1 * xavier(d, H), "b3": np.zeros(d)}


def copy_params(p):
    return {k: v.copy() for k, v in p.items()}


def stack_params(plist):
    return {k: np.stack([p[k] for p in plist]) for k in KEYS}


def unstack_params(P, i):
    return {k: P[k][i] for k in KEYS}


# ---------------------------------------------------------------- single model
def forward(p, X):
    h1 = np.tanh(X @ p["W1"].T + p["b1"])
    h2 = np.tanh(h1 @ p["W2"].T + p["b2"])
    return X + h2 @ p["W3"].T + p["b3"]


def mse(p, X, Y):
    return float(np.mean((forward(p, X) - Y) ** 2))


def loss_and_grads(p, X, Y):
    h1 = np.tanh(X @ p["W1"].T + p["b1"])
    h2 = np.tanh(h1 @ p["W2"].T + p["b2"])
    pred = X + h2 @ p["W3"].T + p["b3"]
    err = pred - Y
    B, d = err.shape
    loss = float(np.mean(err ** 2))
    dg = 2.0 * err / (B * d)
    g = {"W3": dg.T @ h2, "b3": dg.sum(0)}
    da2 = (dg @ p["W3"]) * (1 - h2 ** 2)
    g["W2"], g["b2"] = da2.T @ h1, da2.sum(0)
    da1 = (da2 @ p["W2"]) * (1 - h1 ** 2)
    g["W1"], g["b1"] = da1.T @ X, da1.sum(0)
    return loss, g


def train(p, X, Y, epochs=50, lr=1e-3, batch=128, rng=None, b1=0.9, b2=0.999, eps=1e-8):
    rng = rng or np.random.default_rng(0)
    p = copy_params(p)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    n, t = len(X), 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            _, g = loss_and_grads(p, X[idx], Y[idx])
            t += 1
            for k in KEYS:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                s[k] = b2 * s[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** t)
                sh = s[k] / (1 - b2 ** t)
                p[k] -= lr * mh / (np.sqrt(sh) + eps)
    return p


def jac_points(p, X):
    """Analytic Jacobians df/dx at each row of X.  Returns (n, d, d)."""
    d = X.shape[1]
    h1 = np.tanh(X @ p["W1"].T + p["b1"])
    h2 = np.tanh(h1 @ p["W2"].T + p["b2"])
    A = p["W3"][None, :, :] * (1 - h2 ** 2)[:, None, :]      # (n,d,H)
    B = A @ p["W2"]                                          # (n,d,H)
    C = B * (1 - h1 ** 2)[:, None, :]
    return np.eye(d)[None] + C @ p["W1"]                     # (n,d,d)


# ----------------------------------------------------------------- batched
def step_and_jac_batch(P, x):
    """One map step and Jacobian for M models at M states.  x: (M,d)."""
    d = x.shape[1]
    xc = x[:, :, None]
    h1 = np.tanh((P["W1"] @ xc)[:, :, 0] + P["b1"])
    h2 = np.tanh((P["W2"] @ h1[:, :, None])[:, :, 0] + P["b2"])
    xn = x + (P["W3"] @ h2[:, :, None])[:, :, 0] + P["b3"]
    A = P["W3"] * (1 - h2 ** 2)[:, None, :]
    B = A @ P["W2"]
    C = B * (1 - h1 ** 2)[:, None, :]
    J = C @ P["W1"] + np.eye(d)[None]
    return xn, J


def step_batch_states(P, S):
    """Map step for M models at n states each.  S: (M,n,d) -> (M,n,d)."""
    h1 = np.tanh(S @ P["W1"].transpose(0, 2, 1) + P["b1"][:, None, :])
    h2 = np.tanh(h1 @ P["W2"].transpose(0, 2, 1) + P["b2"][:, None, :])
    return S + h2 @ P["W3"].transpose(0, 2, 1) + P["b3"][:, None, :]
