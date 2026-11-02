"""Nonlinear MPC on a discrete-time CasADi model (multiple shooting, IPOPT).

Used three times: with the prior model (nominal MPC), with the true parameters
(oracle MPC) and with the prior model plus a GP residual (GP-MPC).

State constraints are soft (exact L1 penalty plus a small quadratic term), so
the solver never reports infeasibility; violations are measured on the true
plant afterwards. An optional per-stage tightening (a parameter of the NLP)
shrinks the constraint set; GP-MPC uses it.
"""
import casadi as ca
import numpy as np

from ..plants import rk4_step


class Controller:
    name = "controller"

    def __init__(self, task):
        self.task = task
        self.ref = task.reference()
        self.last_ok = True

    def reset(self):
        self.last_ok = True

    def control(self, y, t):
        raise NotImplementedError


def model_step_function(plant, params, substeps=1):
    """CasADi Function (x, u) -> x_next for a parameter set."""
    x = ca.SX.sym("x", plant.nx)
    u = ca.SX.sym("u", plant.nu)
    xn = rk4_step(plant, x, u, params, plant.dt, substeps, lib=ca)
    return ca.Function("F", [x, u], [xn])


class NMPC(Controller):
    def __init__(self, task, step_fn, ueq_params, name="nmpc", rho1=1e3, rho2=1e4,
                 max_iter=100):
        super().__init__(task)
        self.name = name
        self.F = step_fn
        # input reference: equilibrium of the current set-point under ueq_params
        self.ueq_params = ueq_params
        p = task.plant
        N, nx, nu = task.N, p.nx, p.nu
        nc = len(task.con_idx)
        self.nc = nc

        X = ca.MX.sym("X", nx, N + 1)
        U = ca.MX.sym("U", nu, N)
        S = ca.MX.sym("S", nc, N)
        x0 = ca.MX.sym("x0", nx)
        R_ = ca.MX.sym("r", nx, N + 1)
        ur = ca.MX.sym("ur", nu)
        tig = ca.MX.sym("tig", nc, N)

        Q, Rw, Qf = (ca.DM(m) for m in (task.Q, task.R, task.Qf))
        cost = 0
        g = [X[:, 0] - x0]
        h = []
        for k in range(N):
            e = X[:, k] - R_[:, k]
            du = U[:, k] - ur
            cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([du.T, Rw, du])
            g.append(X[:, k + 1] - self.F(X[:, k], U[:, k]))
            xc = X[task.con_idx, k + 1]
            h.append(xc - task.x_max_c + tig[:, k] - S[:, k])
            h.append(task.x_min_c + tig[:, k] - xc - S[:, k])
            cost += rho1 * ca.sum1(S[:, k]) + rho2 * ca.sumsqr(S[:, k])
        eN = X[:, N] - R_[:, N]
        cost += ca.mtimes([eN.T, Qf, eN])

        w = ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(S))
        par = ca.vertcat(x0, ca.vec(R_), ur, ca.vec(tig))
        G = ca.vertcat(*g, *h)
        self.n_eq = nx * (N + 1)
        self.n_ineq = 2 * nc * N
        opts = {"expand": True, "print_time": False, "ipopt.print_level": 0,
                "ipopt.max_iter": max_iter, "ipopt.tol": 1e-6,
                "ipopt.acceptable_tol": 1e-4, "ipopt.sb": "yes"}
        self.solver = ca.nlpsol("mpc", "ipopt",
                                {"x": w, "f": cost, "g": G, "p": par}, opts)
        inf = np.inf
        self.lbw = np.concatenate([-inf * np.ones(nx * (N + 1)),
                                   np.tile(p.u_min, N), np.zeros(nc * N)])
        self.ubw = np.concatenate([inf * np.ones(nx * (N + 1)),
                                   np.tile(p.u_max, N), inf * np.ones(nc * N)])
        self.lbg = np.concatenate([np.zeros(self.n_eq), -inf * np.ones(self.n_ineq)])
        self.ubg = np.zeros(self.n_eq + self.n_ineq)
        self.dims = (N, nx, nu, nc)
        self.reset()

    def reset(self):
        super().reset()
        self.w0 = None
        self.lam_g = None
        self.X_pred = None
        self.U_pred = None

    def tightening(self, t):
        """Per-stage constraint tightening, shape (nc, N). Zero by default."""
        N, _, _, nc = self.dims
        return np.zeros((nc, N))

    def control(self, y, t):
        N, nx, nu, nc = self.dims
        r = self.ref[t:t + N + 1]
        u_ref = self.task.plant.equilibrium_input(r[0], self.ueq_params)
        if self.w0 is None:
            Xg = np.tile(y, (N + 1, 1)).T
            Ug = np.tile(u_ref, (N, 1)).T
            w0 = np.concatenate([Xg.ravel("F"), Ug.ravel("F"), np.zeros(nc * N)])
        else:
            w0 = self.w0
        par = np.concatenate([y, r.T.ravel("F"), u_ref,
                              self.tightening(t).ravel("F")])
        args = dict(x0=w0, p=par, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        sol = self.solver(**args)
        stats = self.solver.stats()
        self.last_ok = bool(stats["success"]) or stats["return_status"] in (
            "Solved_To_Acceptable_Level", "Maximum_Iterations_Exceeded")
        wopt = np.array(sol["x"]).ravel()
        Xo = wopt[:nx * (N + 1)].reshape((nx, N + 1), order="F")
        Uo = wopt[nx * (N + 1):nx * (N + 1) + nu * N].reshape((nu, N), order="F")
        So = wopt[nx * (N + 1) + nu * N:].reshape((nc, N), order="F")
        self.X_pred, self.U_pred = Xo, Uo
        # warm start: shift by one step
        Xs = np.hstack([Xo[:, 1:], Xo[:, -1:]])
        Us = np.hstack([Uo[:, 1:], Uo[:, -1:]])
        Ss = np.hstack([So[:, 1:], So[:, -1:]])
        self.w0 = np.concatenate([Xs.ravel("F"), Us.ravel("F"), Ss.ravel("F")])
        return Uo[:, 0]


def nominal_mpc(task):
    p = task.plant
    F = model_step_function(p, p.prior_params)
    return NMPC(task, F, p.prior_params, name="Nominal MPC")


def oracle_mpc(task):
    p = task.plant
    F = model_step_function(p, p.true_params)
    return NMPC(task, F, p.true_params, name="Oracle MPC")
