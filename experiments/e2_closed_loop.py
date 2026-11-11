"""E2: closed-loop comparison, 3 plants x 10 seeds x 5 controllers.

Writes results/closed_loop.json (every run's metrics plus mean and std),
results/traj_<plant>.npz (seed 0 trajectories) and the summary figures.
"""
import common  # noqa: F401
import multiprocessing as mp
import sys

import numpy as np

from common import CONTROLLERS, EVAL_SEEDS, RESULTS, build_controllers, dataset, \
    save_json, tuned_lambda
from ddcbench.metrics import evaluate
from ddcbench.simulate import run_closed_loop
from ddcbench.tasks import TASK_NAMES, make_task


def job(args):
    name, s = args
    task = make_task(name)
    ds = dataset(task, s)
    ctrls = build_controllers(task, ds, s, tuned_lambda(name))
    rows, trajs = {}, {}
    for c in ctrls:
        run = run_closed_loop(task, c, seed=500 + s)
        rows[c.name] = evaluate(task, run)
        if s == 0:
            trajs[c.name] = (run["X"], run["U"])
    print(name, s, {k: round(v["cost"], 3) for k, v in rows.items()}, flush=True)
    return name, s, rows, trajs


def summarise(runs):
    keys = ["cost", "ss_err", "viol_max", "viol_steps", "t_mean_ms", "t_max_ms", "fails"]
    out = {}
    for c in CONTROLLERS:
        out[c] = {}
        for k in keys:
            vals = np.array([r[c][k] for r in runs], dtype=float)
            out[c][k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                         "max": float(np.nanmax(vals))}
        out[c]["diverged_runs"] = int(sum(r[c]["diverged"] for r in runs))
    return out


def main(processes=4):
    jobs = [(n, s) for n in TASK_NAMES for s in EVAL_SEEDS]
    with mp.Pool(processes) as pool:
        results = pool.map(job, jobs)
    out = {"seeds": EVAL_SEEDS, "controllers": CONTROLLERS, "plants": {}}
    for name in TASK_NAMES:
        runs = [r for (n, s, r, _) in sorted(results, key=lambda z: z[1]) if n == name]
        out["plants"][name] = {"deepc_lambda_g": tuned_lambda(name),
                               "summary": summarise(runs), "runs": runs}
        trajs = [t for (n, s, _, t) in results if n == name and s == 0][0]
        np.savez(f"{RESULTS}/traj_{name}.npz",
                 **{f"{c}|X": trajs[c][0] for c in trajs},
                 **{f"{c}|U": trajs[c][1] for c in trajs})
    save_json(out, "closed_loop.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
