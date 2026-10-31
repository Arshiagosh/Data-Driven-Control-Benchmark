"""Closed-loop tasks: reference, constraints, cost weights and noise per plant.

Every controller in the benchmark sees the same task, so the comparison is fair:
same horizon, same weights, same input limits, same state constraints.
"""
from dataclasses import dataclass

import numpy as np

from .plants import make_plant


@dataclass
class Task:
    plant: object
    T: int                 # closed-loop steps
    N: int                 # prediction horizon
    x0: np.ndarray
    ref_points: list       # [(start_step, x_ref), ...]
    Q: np.ndarray
    R: np.ndarray
    Qf: np.ndarray
    con_idx: list          # constrained state indices
    x_min_c: np.ndarray
    x_max_c: np.ndarray
    noise_std: np.ndarray  # measurement noise (per state)
    # data collection: range of random set-points and dither size
    setpoint_low: np.ndarray = None
    setpoint_high: np.ndarray = None
    dither_std: np.ndarray = None
    abort_box: np.ndarray = None   # |x| > abort_box ends a data episode
    track_idx: list = None         # states used for the steady-state error

    def reference(self, steps=None):
        """Reference trajectory of length T + N + 1 (last value held)."""
        steps = self.T + self.N + 1 if steps is None else steps
        r = np.zeros((steps, self.plant.nx))
        for start, xr in self.ref_points:
            r[start:] = xr
        return r


def make_task(name):
    p = make_plant(name)
    if name == "msd":
        return Task(
            plant=p, T=200, N=20, x0=np.array([0.0, 0.0]),
            ref_points=[(0, np.array([1.0, 0.0])), (100, np.array([-0.8, 0.0]))],
            Q=np.diag([10.0, 0.1]), R=np.diag([0.01]), Qf=np.diag([10.0, 0.1]),
            con_idx=[1], x_min_c=np.array([-1.0]), x_max_c=np.array([1.0]),
            noise_std=np.array([0.002, 0.005]),
            setpoint_low=np.array([-1.2, 0.0]), setpoint_high=np.array([1.2, 0.0]),
            dither_std=np.array([0.8]), abort_box=np.array([3.0, 4.0]), track_idx=[0],
        )
    if name == "pendulum":
        return Task(
            plant=p, T=160, N=20, x0=np.array([0.5, 0.0]),
            ref_points=[(0, np.array([0.0, 0.0])), (60, np.array([0.3, 0.0])),
                        (120, np.array([0.0, 0.0]))],
            Q=np.diag([20.0, 0.1]), R=np.diag([0.01]), Qf=np.diag([20.0, 0.1]),
            con_idx=[1], x_min_c=np.array([-1.2]), x_max_c=np.array([1.2]),
            noise_std=np.array([0.002, 0.01]),
            setpoint_low=np.array([-0.45, 0.0]), setpoint_high=np.array([0.45, 0.0]),
            dither_std=np.array([0.5]), abort_box=np.array([1.2, 6.0]), track_idx=[0],
        )
    if name == "quadrotor":
        return Task(
            plant=p, T=160, N=20, x0=np.zeros(6),
            ref_points=[(0, np.array([1.0, 0.5, 0, 0, 0, 0])),
                        (80, np.array([0.0, 1.0, 0, 0, 0, 0]))],
            Q=np.diag([10.0, 10.0, 1.0, 0.5, 0.5, 0.1]), R=np.diag([0.01, 0.01]),
            Qf=np.diag([10.0, 10.0, 1.0, 0.5, 0.5, 0.1]),
            con_idx=[2, 3], x_min_c=np.array([-0.35, -0.8]), x_max_c=np.array([0.35, 0.8]),
            noise_std=np.array([0.002, 0.002, 0.003, 0.005, 0.005, 0.01]),
            setpoint_low=np.array([-0.5, 0.0, 0, 0, 0, 0]),
            setpoint_high=np.array([1.5, 1.5, 0, 0, 0, 0]),
            dither_std=np.array([0.6, 0.6]),
            abort_box=np.array([4.0, 4.0, 0.8, 3.0, 3.0, 8.0]), track_idx=[0, 1],
        )
    raise ValueError(f"unknown task {name}")


TASK_NAMES = ["msd", "pendulum", "quadrotor"]
