"""Linear MPC on the lifted Koopman model (condensed QP, OSQP).

Prediction: x_k = C (A^k z_0 + sum_j A^(k-1-j) B u_j), k = 1..N, so the
stacked states are X = Phi z_0 + Gamma U. Only z_0 = psi(y_t) changes between
steps, so P and the constraint matrix are fixed and only q, l, u are updated.
"""
import numpy as np
import osqp
import scipy.sparse as sp

from .nmpc import Controller


class KoopmanMPC(Controller):
    def __init__(self, task, model, rho1=1e3, rho2=1e4, name="Koopman MPC"):
        super().__init__(task)
        self.name = name
        self.model = model
        p = task.plant
        N, nx, nu = task.N, p.nx, p.nu
        nc = len(task.con_idx)
        A, B, C = model.A, model.B, model.C
        nz = A.shape[0]
        Phi = np.zeros((N * nx, nz))
        Gam = np.zeros((N * nx, N * nu))
        Ak = np.eye(nz)
        powers = [np.eye(nz)]
        for k in range(1, N + 1):
            Ak = A @ Ak
            powers.append(Ak)
        for k in range(1, N + 1):
            Phi[(k - 1) * nx:k * nx] = C @ powers[k]
            for j in range(k):
                Gam[(k - 1) * nx:k * nx, j * nu:(j + 1) * nu] = C @ powers[k - 1 - j] @ B
        self.Phi, self.Gam = Phi, Gam
        Qbar = np.kron(np.eye(N), task.Q)
        Qbar[-nx:, -nx:] = task.Qf
        Rbar = np.kron(np.eye(N), task.R)
        self.Qbar, self.Rbar = Qbar, Rbar
        H = Gam.T @ Qbar @ Gam + Rbar
        H = 0.5 * (H + H.T)
        P = sp.block_diag([sp.csc_matrix(2 * H), sp.eye(N * nc) * 2 * rho2], format="csc")
        Cc = np.kron(np.eye(N), np.eye(nx)[task.con_idx])
        self.CcG = Cc @ Gam
        self.CcPhi = Cc @ Phi
        Acon = sp.vstack([
            sp.hstack([sp.eye(N * nu), sp.csc_matrix((N * nu, N * nc))]),
            sp.hstack([sp.csc_matrix(self.CcG), -sp.eye(N * nc)]),
            sp.hstack([sp.csc_matrix(self.CcG), sp.eye(N * nc)]),
            sp.hstack([sp.csc_matrix((N * nc, N * nu)), sp.eye(N * nc)]),
        ], format="csc")
        self.dims = (N, nx, nu, nc)
        self.rho1 = rho1
        self.prob = osqp.OSQP()
        self.prob.setup(sp.triu(P, format="csc"), np.zeros(N * nu + N * nc), Acon,
                        -np.inf * np.ones(Acon.shape[0]), np.inf * np.ones(Acon.shape[0]),
                        verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000,
                        polishing=False, warm_starting=True)
        self.reset()

    def reset(self):
        super().reset()
        self.u_last = None

    def control(self, y, t):
        N, nx, nu, nc = self.dims
        task, p = self.task, self.task.plant
        r = self.ref[t + 1:t + N + 1].ravel()
        u_ref = p.equilibrium_input(self.ref[t], p.prior_params)
        z0 = self.model.lift(y)[0]
        free = self.Phi @ z0
        qU = 2 * self.Gam.T @ self.Qbar @ (free - r) - 2 * self.Rbar @ np.tile(u_ref, N)
        q = np.concatenate([qU, self.rho1 * np.ones(N * nc)])
        cfree = self.CcPhi @ z0
        inf = np.inf
        l = np.concatenate([np.tile(p.u_min, N), -inf * np.ones(N * nc),
                            np.tile(task.x_min_c, N) - cfree, np.zeros(N * nc)])
        u = np.concatenate([np.tile(p.u_max, N), np.tile(task.x_max_c, N) - cfree,
                            inf * np.ones(N * nc), inf * np.ones(N * nc)])
        self.prob.update(q=q, l=l, u=u)
        res = self.prob.solve(raise_error=False)
        status = str(res.info.status).lower()
        self.last_ok = "solved" in status and "infeasible" not in status
        if self.last_ok:
            uk = res.x[:nu]
        else:
            uk = u_ref if self.u_last is None else self.u_last
        uk = np.clip(uk, p.u_min, p.u_max)
        self.u_last = uk
        return uk
