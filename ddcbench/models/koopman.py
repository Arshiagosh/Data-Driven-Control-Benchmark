"""Koopman model by extended dynamic mode decomposition (EDMD) with inputs.

Lifted state psi(x) = [x, 1, rbf_1(x), ..., rbf_m(x)], where the radial basis
functions use only the translation-invariant features ``plant.feature_idx``
(standardised). The model z_{k+1} = A z_k + B u_k is fitted by ridge least
squares on measured data, and x = C z with C = [I 0] because x is part of psi.
The lifted model is linear in z and u, so the MPC on it is a QP.
"""
import numpy as np


class KoopmanModel:
    def __init__(self, plant, n_rbf=40, ridge=1e-6, seed=0):
        self.plant = plant
        self.n_rbf = n_rbf
        self.ridge = ridge
        self.seed = seed

    def lift(self, X):
        X = np.atleast_2d(X)
        if self.n_rbf == 0:
            return np.hstack([X, np.ones((len(X), 1))])
        F = (X[:, self.plant.feature_idx] - self.fm) / self.fs
        d2 = ((F[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1)
        rbf = np.exp(-d2 / (2 * self.width ** 2))
        return np.hstack([X, np.ones((len(X), 1)), rbf])

    def fit(self, dataset):
        Xk, Uk, Xk1 = dataset.transitions()
        rng = np.random.default_rng(self.seed)
        F = Xk[:, self.plant.feature_idx]
        self.fm, self.fs = F.mean(0), F.std(0) + 1e-12
        Fn = (F - self.fm) / self.fs
        self.centres = Fn[rng.choice(len(Fn), self.n_rbf, replace=False)]
        dc = np.sqrt(((self.centres[:, None] - self.centres[None]) ** 2).sum(-1))
        self.width = np.median(dc[dc > 0]) if self.n_rbf > 1 else 1.0
        Zk, Zk1 = self.lift(Xk), self.lift(Xk1)
        G = np.hstack([Zk, Uk])
        W = np.linalg.solve(G.T @ G + self.ridge * len(G) * np.eye(G.shape[1]), G.T @ Zk1)
        nz = Zk.shape[1]
        self.A = W[:nz].T
        self.B = W[nz:].T
        self.C = np.hstack([np.eye(self.plant.nx), np.zeros((self.plant.nx, nz - self.plant.nx))])
        self.nz = nz
        return self

    def predict(self, x0, U):
        """Open-loop state prediction for an input sequence U (H, nu)."""
        z = self.lift(x0)[0]
        out = []
        for u in U:
            z = self.A @ z + self.B @ u
            out.append(self.C @ z)
        return np.array(out)
