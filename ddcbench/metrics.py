"""Closed-loop metrics, computed on the true (noise-free) state."""
import numpy as np


def evaluate(task, run):
    X, U = run["X"], run["U"]
    T = task.T
    ref = task.reference()[:T + 1]
    diverged = bool(np.any(~np.isfinite(X)))
    if diverged:
        return dict(cost=float("nan"), ss_err=float("nan"), viol_max=float("nan"), viol_steps=int(T),
                    t_mean_ms=float(np.mean(run["times"]) * 1e3),
                    t_max_ms=float(np.max(run["times"]) * 1e3),
                    fails=int(run["fails"]), diverged=True)
    E = X[1:] - ref[1:]
    cost = float(np.mean(np.einsum("ti,ij,tj->t", E, task.Q, E)))
    xc = X[1:, task.con_idx]
    viol = np.maximum(xc - task.x_max_c, 0) + np.maximum(task.x_min_c - xc, 0)
    return dict(cost=cost, ss_err=steady_state_error(task, run), viol_max=float(viol.max()),
                viol_steps=int(np.sum(viol.max(axis=1) > 1e-6)),
                t_mean_ms=float(np.mean(run["times"]) * 1e3),
                t_max_ms=float(np.max(run["times"]) * 1e3),
                fails=int(run["fails"]), diverged=False)


def steady_state_error(task, run, window=20):
    """Mean norm of the position error over the last ``window`` steps of
    every reference segment (shows the offset a wrong model leaves)."""
    X = run["X"]
    if np.any(~np.isfinite(X)):
        return float("nan")
    ref = task.reference()
    # The MPC previews the reference N steps ahead, so a segment that ends in
    # a set-point switch is measured before the preview reaches the switch.
    ends = [s - task.N for s, _ in task.ref_points[1:]] + [task.T]
    errs = []
    for e in ends:
        seg = slice(e - window, e)
        d = X[seg][:, task.track_idx] - ref[seg][:, task.track_idx]
        errs.append(np.linalg.norm(d, axis=1).mean())
    return float(np.mean(errs))
