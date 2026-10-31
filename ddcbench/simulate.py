"""Closed-loop simulation of the true plant with measurement noise."""
import time

import numpy as np

from .plants import rk4_step

SIM_SUBSTEPS = 10  # the true plant is integrated finer than any model uses


def true_step(plant, x, u):
    return rk4_step(plant, x, u, plant.true_params, plant.dt, SIM_SUBSTEPS)


def run_closed_loop(task, controller, seed=0):
    """Run one episode. Returns a dict with true states, inputs and timings."""
    rng = np.random.default_rng(seed)
    p = task.plant
    x = task.x0.astype(float).copy()
    X = np.zeros((task.T + 1, p.nx))
    U = np.zeros((task.T, p.nu))
    times = np.zeros(task.T)
    fails = 0
    X[0] = x
    controller.reset()
    for t in range(task.T):
        y = x + task.noise_std * rng.standard_normal(p.nx)
        t0 = time.perf_counter()
        u = controller.control(y, t)
        times[t] = time.perf_counter() - t0
        fails += int(not controller.last_ok)
        u = np.clip(u, p.u_min, p.u_max)
        x = true_step(p, x, u)
        U[t] = u
        X[t + 1] = x
        if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1e3):
            X[t + 1:] = np.nan
            U[t + 1:] = np.nan
            break
    return dict(X=X, U=U, times=times, fails=fails)
