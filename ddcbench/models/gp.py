"""Exact Gaussian-process regression in NumPy, with a CasADi mean.

One independent GP per output, squared-exponential kernel with one
length-scale per input (ARD). Hyperparameters maximise the log marginal
likelihood (L-BFGS-B, a few random restarts). To keep the MPC cheap the GP is
trained on a subset of ``n_max`` points of the dataset; the subset is drawn
at random with a fixed seed. This is a subset-of-data approximation, not a
sparse GP with inducing variables.
"""
import casadi as ca
import numpy as np
from scipy.optimize import minimize


def _sqdist(A, B, ell):
    A = A / ell
    B = B / ell
    return (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T).clip(0)


class GP1D:
    """GP for one scalar output on standardised inputs."""

    def fit(self, Z, y, restarts=3, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        self.Z = Z
        self.ym = y.mean()
        self.ys = y.std() + 1e-12
        yn = (y - self.ym) / self.ys
        d = Z.shape[1]

        def nlml(theta):
            ell = np.exp(theta[:d])
            sf2 = np.exp(2 * theta[d])
            sn2 = np.exp(2 * theta[d + 1]) + 1e-8
            K = sf2 * np.exp(-0.5 * _sqdist(Z, Z, ell)) + sn2 * np.eye(len(Z))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            a = np.linalg.solve(L.T, np.linalg.solve(L, yn))
            return 0.5 * yn @ a + np.sum(np.log(np.diag(L)))

        best = None
        for i in range(restarts):
            th0 = np.concatenate([np.log(rng.uniform(0.5, 2.0, d)),
                                  [0.0], [np.log(rng.uniform(0.05, 0.3))]])
            bounds = [(-3, 3)] * d + [(-3, 3), (-7, 0)]
            res = minimize(nlml, th0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        th = best.x
        self.ell = np.exp(th[:d])
        self.sf2 = float(np.exp(2 * th[d]))
        self.sn2 = float(np.exp(2 * th[d + 1]) + 1e-8)
        K = self.sf2 * np.exp(-0.5 * _sqdist(Z, Z, self.ell)) + self.sn2 * np.eye(len(Z))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yn))
        return self

    def predict(self, Zs):
        Ks = self.sf2 * np.exp(-0.5 * _sqdist(Zs, self.Z, self.ell))
        mean = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.sf2 - np.sum(v ** 2, 0), 1e-12)
        return mean * self.ys + self.ym, var * self.ys ** 2, self.sn2 * self.ys ** 2


class GPModel:
    """Multi-output GP: inputs z (raw), outputs y (raw), each output its own GP."""

    def __init__(self, n_max=150, restarts=3, seed=0):
        self.n_max = n_max
        self.restarts = restarts
        self.seed = seed

    def fit(self, Z, Y):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(Z), size=min(self.n_max, len(Z)), replace=False)
        self.zm = Z.mean(0)
        self.zs = Z.std(0) + 1e-12
        Zn = (Z[idx] - self.zm) / self.zs
        self.gps = [GP1D().fit(Zn, Y[idx, j], self.restarts, rng) for j in range(Y.shape[1])]
        return self

    def predict(self, Z):
        """Mean, latent variance and noise variance, each of shape (n, ny)."""
        Zn = (np.atleast_2d(Z) - self.zm) / self.zs
        out = [g.predict(Zn) for g in self.gps]
        mean = np.stack([o[0] for o in out], 1)
        var = np.stack([o[1] for o in out], 1)
        noise = np.array([o[2] for o in out])
        return mean, var, noise

    def casadi_mean(self):
        """CasADi Function z -> GP mean (column), for use inside the MPC."""
        d = len(self.zm)
        z = ca.SX.sym("z", d)
        zn = (z - ca.DM(self.zm)) / ca.DM(self.zs)
        outs = []
        for g in self.gps:
            Zd = ca.DM(g.Z / g.ell)                       # (M, d)
            q = zn / ca.DM(g.ell)
            diff = Zd - ca.repmat(q.T, Zd.shape[0], 1)
            k = g.sf2 * ca.exp(-0.5 * ca.sum2(diff ** 2))  # (M, 1)
            outs.append(ca.dot(k, ca.DM(g.alpha)) * g.ys + g.ym)
        return ca.Function("gp_mean", [z], [ca.vertcat(*outs)])
