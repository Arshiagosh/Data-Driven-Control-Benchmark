"""Prior model plus a GP residual on the velocity states.

The mismatch in every plant of this benchmark enters through the
accelerations, so the GP learns the one-step velocity error
    dv = v_{k+1} - F_prior(x_k, u_k)_v
from features z = [x_k[feature_idx], u_k]. The matching position correction
uses the constant-acceleration approximation dp = dt/2 * dv.
"""
import casadi as ca
import numpy as np

from ..controllers.nmpc import model_step_function
from .gp import GPModel


class ResidualGP:
    def __init__(self, plant, n_max=150, seed=0):
        self.plant = plant
        self.F_prior = model_step_function(plant, plant.prior_params)
        self.gp = GPModel(n_max=n_max, seed=seed)
        E = np.zeros((plant.nx, len(plant.vel_idx)))
        for j, (ip, iv) in enumerate(zip(plant.pos_idx, plant.vel_idx)):
            E[ip, j] = 0.5 * plant.dt
            E[iv, j] = 1.0
        self.E = E

    def features(self, X, U):
        return np.hstack([np.atleast_2d(X)[:, self.plant.feature_idx], np.atleast_2d(U)])

    def prior_batch(self, X, U):
        Fm = self.F_prior.map(len(X))
        return np.array(Fm(X.T, U.T)).T

    def targets(self, Xk, Uk, Xk1):
        return (Xk1 - self.prior_batch(Xk, Uk))[:, self.plant.vel_idx]

    def fit(self, dataset):
        Xk, Uk, Xk1 = dataset.transitions()
        self.gp.fit(self.features(Xk, Uk), self.targets(Xk, Uk, Xk1))
        return self

    def predict_step(self, X, U):
        """Mean next state (batch) and latent velocity variance."""
        m, var, _ = self.gp.predict(self.features(X, U))
        return self.prior_batch(X, U) + m @ self.E.T, var

    def step_function(self):
        p = self.plant
        x = ca.SX.sym("x", p.nx)
        u = ca.SX.sym("u", p.nu)
        gp_mean = self.gp.casadi_mean()
        z = ca.vertcat(x[p.feature_idx], u)
        xn = self.F_prior(x, u) + ca.DM(self.E) @ gp_mean(z)
        return ca.Function("F_gp", [x, u], [xn])
