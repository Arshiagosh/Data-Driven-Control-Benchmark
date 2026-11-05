"""GP-MPC: nominal MPC on the prior model plus the GP mean, with constraint
tightening from the GP variance.

Tightening (an approximation, stated as such in the README): the latent GP
variance is evaluated along the previous optimal trajectory (shifted by one
step), velocity variances are summed over the horizon, and position-like
states integrate the velocity standard deviation with the sampling time. The
propagation ignores the Jacobian of the dynamics, so it is a heuristic
margin, not a certified bound. The constraint is shrunk by beta standard
deviations, capped at half the constraint width.
"""
import numpy as np

from .nmpc import NMPC


class GPMPC(NMPC):
    def __init__(self, task, residual_model, beta=2.0, name="GP-MPC"):
        self.res = residual_model
        self.beta = beta
        p = task.plant
        super().__init__(task, residual_model.step_function(), p.prior_params, name=name)

    def tightening(self, t):
        N, nx, nu, nc = self.dims
        task, p = self.task, self.task.plant
        if self.X_pred is None:
            return np.zeros((nc, N))
        Xs = np.hstack([self.X_pred[:, 1:], self.X_pred[:, -1:]])[:, :N]
        Us = np.hstack([self.U_pred[:, 1:], self.U_pred[:, -1:]])
        _, var = self.res.predict_step(Xs.T, Us.T)        # (N, n_vel)
        sv = np.sqrt(np.cumsum(var, axis=0))               # velocity std per stage
        sp = np.cumsum(sv, axis=0) * p.dt                  # position std per stage
        tig = np.zeros((nc, N))
        for i, c in enumerate(task.con_idx):
            if c in p.vel_idx:
                tig[i] = self.beta * sv[:, p.vel_idx.index(c)]
            elif c in p.pos_idx:
                tig[i] = self.beta * sp[:, p.pos_idx.index(c)]
            cap = 0.25 * (task.x_max_c[i] - task.x_min_c[i])
            tig[i] = np.minimum(tig[i], cap)
        self.last_tightening = tig
        return tig
