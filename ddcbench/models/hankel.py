"""Hankel (behavioural) data matrices for DeePC.

Convention: the sample w_k = (u_k, y_{k+1}) pairs an input with the
measurement it produces, y = x (full state). A window of L = Tini + N samples
splits into a past part (Tini samples: the last Tini inputs and the outputs
up to the current measurement) and a future part (the N inputs to choose and
the N outputs they produce). Several episodes are combined column-wise
(a mosaic Hankel matrix), which is valid because each column is a trajectory
of the same system on its own.
"""
import numpy as np


def hankel_columns(U, Y1, L):
    """Columns of length L*(nu+ny) from one episode.
    U: (T, nu) inputs u_0..u_{T-1};  Y1: (T, ny) outputs y_1..y_T."""
    T = len(U)
    cols = []
    for j in range(T - L + 1):
        cols.append(np.concatenate([U[j:j + L].ravel(), Y1[j:j + L].ravel()]))
    return np.array(cols).T


class HankelData:
    def __init__(self, dataset, Tini, N, max_cols=None, seed=0):
        self.Tini, self.N = Tini, N
        L = Tini + N
        ep = dataset.episodes[0]
        self.nu = ep["U"].shape[1]
        self.ny = ep["Xm"].shape[1]
        Hu, Hy = [], []
        for e in dataset.episodes:
            H = hankel_columns(e["U"], e["Xm"][1:], L)
            Hu.append(H[:L * self.nu])
            Hy.append(H[L * self.nu:])
        Hu = np.hstack(Hu)
        Hy = np.hstack(Hy)
        if max_cols is not None and Hu.shape[1] > max_cols:
            idx = np.sort(np.random.default_rng(seed).choice(Hu.shape[1], max_cols, replace=False))
            Hu, Hy = Hu[:, idx], Hy[:, idx]
        nu, ny = self.nu, self.ny
        self.Up, self.Uf = Hu[:Tini * nu], Hu[Tini * nu:]
        self.Yp, self.Yf = Hy[:Tini * ny], Hy[Tini * ny:]
        self.ncol = Hu.shape[1]

    def min_columns(self, n):
        """Columns needed for persistency of excitation of order L + n."""
        return (self.Tini + self.N) * self.nu + n

    def predict(self, uini, yini, uf, lam=1e-6):
        """Least-squares (ridge) output prediction from past data and future inputs."""
        A = np.vstack([self.Up, self.Yp, self.Uf])
        b = np.concatenate([uini.ravel(), yini.ravel(), uf.ravel()])
        g = np.linalg.solve(A.T @ A + lam * np.eye(self.ncol), A.T @ b)
        return (self.Yf @ g).reshape(self.N, self.ny)
