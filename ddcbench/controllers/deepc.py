"""Regularised DeePC (data-enabled predictive control), solved as a QP with OSQP.

Decision variables: g (Hankel weights), u (N inputs), y (N outputs),
sigma (slack on the past outputs) and s (soft state-constraint slack).

    min  sum (y_k - r_k)' Q (y_k - r_k) + (u_k - u_ref)' R (u_k - u_ref)
         + lam_g ||g||^2 + lam_s ||sigma||^2 + rho1 sum s + rho2 ||s||^2
    s.t. Up g = u_ini,  Yp g = y_ini + sigma,  Uf g = u,  Yf g = y,
         u_min <= u <= u_max,  soft box on the constrained outputs.

For a noise-free LTI plant with persistently exciting data, lam_g -> 0 and
sigma = 0 recover MPC on the true model (the fundamental lemma). For noisy
data and the nonlinear plants the quadratic regularisation is a heuristic.
"""
import numpy as np
import osqp
import scipy.sparse as sp

from ..models.hankel import HankelData
from .nmpc import Controller


class DeePC(Controller):
    def __init__(self, task, dataset, Tini=3, lam_g=1.0, lam_s=1e4, rho1=1e3,
                 rho2=1e4, max_cols=800, seed=0, name="DeePC"):
        super().__init__(task)
        self.name = name
        p = task.plant
        N, nu, ny = task.N, p.nu, p.nx
        self.Tini = Tini
        self.H = H = HankelData(dataset, Tini, N, max_cols=max_cols, seed=seed)
        ng = H.ncol
        nc = len(task.con_idx)
        self.dims = (N, nu, ny, nc, ng)
        nz = ng + N * nu + N * ny + Tini * ny + N * nc
        self.off = dict(g=0, u=ng, y=ng + N * nu, sig=ng + N * nu + N * ny,
                        s=ng + N * nu + N * ny + Tini * ny)

        Qs = [task.Q] * (N - 1) + [task.Qf]
        Pdiag = [sp.eye(ng) * lam_g, sp.kron(sp.eye(N), sp.csc_matrix(task.R)),
                 sp.block_diag([sp.csc_matrix(q) for q in Qs]),
                 sp.eye(Tini * ny) * lam_s, sp.eye(N * nc) * rho2]
        P = 2 * sp.block_diag(Pdiag, format="csc")

        Z = lambda r, c: sp.csc_matrix((r, c))
        I = sp.eye
        Cc = sp.csc_matrix(np.eye(ny)[task.con_idx])
        CcN = sp.kron(I(N), Cc)
        rows = [
            sp.hstack([sp.csc_matrix(H.Up), Z(Tini * nu, N * nu), Z(Tini * nu, N * ny),
                       Z(Tini * nu, Tini * ny), Z(Tini * nu, N * nc)]),
            sp.hstack([sp.csc_matrix(H.Yp), Z(Tini * ny, N * nu), Z(Tini * ny, N * ny),
                       -I(Tini * ny), Z(Tini * ny, N * nc)]),
            sp.hstack([sp.csc_matrix(H.Uf), -I(N * nu), Z(N * nu, N * ny),
                       Z(N * nu, Tini * ny), Z(N * nu, N * nc)]),
            sp.hstack([sp.csc_matrix(H.Yf), Z(N * ny, N * nu), -I(N * ny),
                       Z(N * ny, Tini * ny), Z(N * ny, N * nc)]),
            sp.hstack([Z(N * nu, ng), I(N * nu), Z(N * nu, N * ny),
                       Z(N * nu, Tini * ny), Z(N * nu, N * nc)]),
            sp.hstack([Z(N * nc, ng), Z(N * nc, N * nu), CcN,
                       Z(N * nc, Tini * ny), -I(N * nc)]),
            sp.hstack([Z(N * nc, ng), Z(N * nc, N * nu), CcN,
                       Z(N * nc, Tini * ny), I(N * nc)]),
            sp.hstack([Z(N * nc, ng), Z(N * nc, N * nu), Z(N * nc, N * ny),
                       Z(N * nc, Tini * ny), I(N * nc)]),
        ]
        A = sp.vstack(rows, format="csc")
        inf = np.inf
        self.n_eq_u = Tini * nu
        self.n_eq_y = Tini * ny
        l_rest = np.concatenate([np.zeros(N * nu + N * ny), np.tile(p.u_min, N),
                                 -inf * np.ones(N * nc), np.tile(task.x_min_c, N),
                                 np.zeros(N * nc)])
        u_rest = np.concatenate([np.zeros(N * nu + N * ny), np.tile(p.u_max, N),
                                 np.tile(task.x_max_c, N), inf * np.ones(N * nc),
                                 inf * np.ones(N * nc)])
        self.l_rest, self.u_rest = l_rest, u_rest
        self.rho1 = rho1
        self.nz = nz
        self.prob = osqp.OSQP()
        q0 = np.zeros(nz)
        b0 = np.zeros(Tini * nu + Tini * ny)
        self.prob.setup(sp.triu(P, format="csc"), q0, A,
                        np.concatenate([b0, l_rest]), np.concatenate([b0, u_rest]),
                        verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000,
                        polishing=False, warm_starting=True)
        self.reset()

    def reset(self):
        super().reset()
        self.u_hist = None
        self.y_hist = None
        self.u_last = None

    def control(self, y, t):
        N, nu, ny, nc, ng = self.dims
        task, p = self.task, self.task.plant
        r = self.ref[t + 1:t + N + 1]
        u_ref = p.equilibrium_input(self.ref[t], p.prior_params)
        if self.u_hist is None:
            # no past data at the start: assume the plant rested at y under u_ref
            self.u_hist = [u_ref.copy() for _ in range(self.Tini)]
            self.y_hist = [y.copy() for _ in range(self.Tini)]
        else:
            self.y_hist = self.y_hist[1:] + [y.copy()]
        uini = np.concatenate(self.u_hist)
        yini = np.concatenate(self.y_hist)
        q = np.zeros(self.nz)
        Qs = [task.Q] * (N - 1) + [task.Qf]
        q[self.off["u"]:self.off["y"]] = np.concatenate([-2 * task.R @ u_ref] * N)
        q[self.off["y"]:self.off["sig"]] = np.concatenate(
            [-2 * Qs[k] @ r[k] for k in range(N)])
        q[self.off["s"]:] = self.rho1
        b = np.concatenate([uini, yini])
        self.prob.update(q=q, l=np.concatenate([b, self.l_rest]),
                         u=np.concatenate([b, self.u_rest]))
        res = self.prob.solve(raise_error=False)
        self.last_ok = "solved" in str(res.info.status).lower() and \
            "infeasible" not in str(res.info.status).lower()
        if self.last_ok:
            u_seq = res.x[self.off["u"]:self.off["y"]].reshape(N, nu)
            self.u_plan = u_seq
            u = u_seq[0]
        else:
            u = u_ref if self.u_last is None else self.u_last
        u = np.clip(u, p.u_min, p.u_max)
        self.u_hist = self.u_hist[1:] + [u.copy()]
        self.u_last = u
        return u
